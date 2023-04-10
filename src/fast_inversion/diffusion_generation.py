import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from src.fast_inversion.config import TrainConfig
from src.misc import compute
import wandb

diffusion_model_name = 'runwayml/stable-diffusion-v1-5'
num_inference_steps = 25
# num_inference_steps = 10
placeholder_token = 'my_new_token'
cache_dir = compute.get_cache_dir()


def generate_images(embedding, experiment, config: TrainConfig):
    # TODO:
    # make sure loading learned embedding into model... can look at others code..
    # probably should overwrite pipeline to use my own embeddings with fast embedder

    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name, cache_dir=cache_dir,
                                              subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        diffusion_model_name, cache_dir=cache_dir, subfolder="text_encoder", )
    vae = AutoencoderKL.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="vae", )

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_name, cache_dir=cache_dir, subfolder="unet", )

    set_embedding_in_text_encoder(embedding, text_encoder, tokenizer)

    validation_prompt = f"An image of {placeholder_token}"

    # create pipeline (note: unet and vae are loaded again in float32)
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

    prompt = config.num_images_per_person_to_log * [validation_prompt]
    images = pipeline(prompt, num_inference_steps=num_inference_steps).images
    experiment.log(
        {
            "validation": [
                wandb.Image(image, caption=f"{i}: {validation_prompt}")
                for i, image in enumerate(images)
            ]
        })

    del text_encoder
    del vae
    del unet
    del pipeline
    torch.cuda.empty_cache()


def set_embedding_in_text_encoder(embedding, text_encoder, tokenizer):
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    assert num_added_tokens == 1

    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = embedding
