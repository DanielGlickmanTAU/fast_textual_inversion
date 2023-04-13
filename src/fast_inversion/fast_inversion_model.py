from torch import nn

from src.fast_inversion.config import get_config
from src.misc import compute
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor, CLIPImageProcessor, CLIPModel
from diffusers.utils.import_utils import is_xformers_available

embedding_size = 768
clip_output_size = 1024
diffusion_model_name = 'runwayml/stable-diffusion-v1-5'
placeholder_token = 'my_new_token'
cache_dir = compute.get_cache_dir()
device = compute.get_device()


class SimpleModel(torch.nn.Module):
    def __init__(self, num_steps, image_encoder):
        # num_steps == len(dataset.steps)
        super().__init__()
        self.image_encoder = image_encoder
        self.num_steps = num_steps

        self.embedding_step_dim = embedding_size // 2
        self.step_embedding = torch.nn.Embedding(num_steps, self.embedding_step_dim)
        self.embedding_step_with_timestep_dim = embedding_size + self.embedding_step_dim
        self.embedding_update = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_step_with_timestep_dim,
                            (self.embedding_step_with_timestep_dim) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((self.embedding_step_with_timestep_dim) // 2, embedding_size)

        )

    def forward(self, images, x_emb, step):
        bs = x_emb.shape[0]
        timestep = self.step_embedding(step.to(x_emb.device))
        emb_with_timestep = torch.cat((x_emb, timestep.expand(bs, -1)), dim=1)
        emb_update = self.embedding_update(emb_with_timestep)

        return emb_update + x_emb

    @torch.no_grad()
    def encode_images(self, images):
        return images


class SimpleCrossAttentionModel(torch.nn.Module):
    def __init__(self, num_steps, image_encoder, step_time_scale=False, embedding_hidden_multiplier=0.5,
                 project_patches_dim=0):
        # num_steps == len(dataset.steps)
        super().__init__()
        self.project_patches_dim = project_patches_dim
        self.image_encoder = image_encoder
        self.num_steps = num_steps
        self.step_time_scale = step_time_scale
        if step_time_scale:
            self.time_scale = torch.nn.Parameter(torch.ones(self.num_steps, 1))

        self.step_embedding = torch.nn.Embedding(num_steps, embedding_size)
        self.embedding_update = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, int(embedding_hidden_multiplier * embedding_size)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(embedding_hidden_multiplier * embedding_size), embedding_size)

        )
        if self.project_patches_dim:
            self.patch_projection = torch.nn.Linear(clip_output_size, self.project_patches_dim)

        kv_dim = self.project_patches_dim if self.project_patches_dim else clip_output_size
        self.attn = torch.nn.MultiheadAttention(embed_dim=embedding_size, kdim=kv_dim,
                                                vdim=kv_dim,
                                                num_heads=4, batch_first=True)

    def forward(self, images, x_emb, step):
        bs = x_emb.shape[0]
        images = images.view(bs, -1, images.shape[-1])

        if self.project_patches_dim:
            images = self.patch_projection(images)

        step = step.to(x_emb.device).expand(bs)
        timestep = self.step_embedding(step)
        # emb_with_timestep = torch.cat((x_emb, timestep), dim=1)
        emb_with_timestep = x_emb + timestep

        emb_new, attn = self.attn(emb_with_timestep.unsqueeze(1), images, images, need_weights=False)
        emb_new = emb_new.squeeze(1)

        emb_new = emb_new + emb_with_timestep
        emb_update = self.embedding_update(emb_new)

        if self.step_time_scale:
            emb_update = emb_update * torch.nn.functional.embedding(step, self.time_scale)

        return emb_update + x_emb

    @torch.no_grad()
    def encode_images(self, images):
        return self.image_encoder(images)


def get_unet():
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="unet", )
    unet.requires_grad_(False)
    # if torch.cuda.is_available() and is_xformers_available():
    #     unet.enable_xformers_memory_efficient_attention()
    return unet.to(generation_device())


def get_vae():
    vae = AutoencoderKL.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="vae", )
    vae.requires_grad_(False)
    return vae.to(generation_device())


def get_clip_text():
    clip = CLIPTextModel.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="text_encoder", )
    clip.requires_grad_(False)
    return clip.to(generation_device())


def get_clip_tokenizer():
    return CLIPTokenizer.from_pretrained(diffusion_model_name, cache_dir=cache_dir,
                                         subfolder="tokenizer")


def get_clip_image():
    def _image_processor_wrapper(image):
        pixel_values_ = processor(image)['pixel_values']
        assert len(pixel_values_) == 1, f'len must be one, got {pixel_values_}'
        return torch.from_numpy(pixel_values_[0])

    def _clip_image_wrapper(batched_images):
        B, N, C, H, W = batched_images.shape
        batched_images = batched_images.view(-1, C, H, W)

        data = clip(batched_images, output_hidden_states=True)
        hidden_state = data.hidden_states[-2]

        a, n, d = hidden_state.shape
        return hidden_state.view(B, N, n, d)

    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)

    clip.requires_grad_(False)
    clip = clip.to(device)
    return _clip_image_wrapper, _image_processor_wrapper


def set_embedding_in_text_encoder(embedding, text_encoder, tokenizer):
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    assert num_added_tokens == 1
    embedding = embedding.to(text_encoder.device)

    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = embedding


def get_diffusion_pipeline(text_encoder, tokenizer, unet, vae):
    pipeline = DiffusionPipeline.from_pretrained(
        diffusion_model_name, cache_dir=cache_dir,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,

    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None
    return pipeline


def generation_device():
    cfg = get_config()
    return 'cpu' if cfg.validate_images_on_cpu else device


class VAEEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.image_size = 64  # or whatever vae outsize is
        self.patch_size = 2  # idk

        self.patch_embedding = nn.Conv2d(
            in_channels=4,  # if i use first layer of unet, it is 320...
            out_channels=self.embed_dim,  # 720 in clip..
            kernel_size=self.patch_size,  # 32 in clip
            stride=self.patch_size,  # this makes non overlapping patches
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
