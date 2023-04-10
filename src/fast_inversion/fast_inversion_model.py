from src.fast_inversion.config import get_config
from src.misc import compute
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor, CLIPImageProcessor, CLIPModel
from diffusers.utils.import_utils import is_xformers_available

embedding_size = 768
diffusion_model_name = 'runwayml/stable-diffusion-v1-5'
placeholder_token = 'my_new_token'
cache_dir = compute.get_cache_dir()
device = compute.get_device()


class SimpleModel(torch.nn.Module):
    def __init__(self, num_steps):
        # num_steps == len(dataset.steps)
        super().__init__()
        self.num_steps = num_steps

        self.embedding_step_dim = embedding_size // 2
        self.step_embedding = torch.nn.Embedding(num_steps, self.embedding_step_dim)
        self.embedding_update = torch.nn.Sequential(
            torch.nn.Linear(embedding_size + self.embedding_step_dim,
                            (embedding_size + self.embedding_step_dim) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((embedding_size + self.embedding_step_dim) // 2, embedding_size)

        )

    def forward(self, images, x_emb, step):
        bs = x_emb.shape[0]
        timestep = self.step_embedding(step.to(x_emb.device))
        emb_with_timestep = torch.cat((x_emb, timestep.expand(bs, -1)), dim=1)
        emb_update = self.embedding_update(emb_with_timestep)

        return emb_update + x_emb


def get_unet():
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="unet", )
    unet.requires_grad_(False)
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
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

    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)

    clip.requires_grad_(False)
    return clip.to(generation_device()), _image_processor_wrapper


def set_embedding_in_text_encoder(embedding, text_encoder, tokenizer):
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    assert num_added_tokens == 1

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
