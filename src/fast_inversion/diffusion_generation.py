import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from src.misc import compute
import wandb


def generate_images(embedding, experiment):
    # TODO:
    # make sure loading learned embedding into model... can look at others code..
    # probably should overwrite pipeline to use my own embeddings with fast embedder

    diffusion_model_name = 'runwayml/stable-diffusion-v1-5'
    num_validation_images = 4
    cache_dir = compute.get_cache_dir()

    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name, cache_dir=cache_dir,
                                              subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        diffusion_model_name, cache_dir=cache_dir, subfolder="text_encoder", )
    vae = AutoencoderKL.from_pretrained(diffusion_model_name, cache_dir=cache_dir, subfolder="vae", )

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_name, cache_dir=cache_dir, subfolder="unet", )

    validation_prompt = "An image of my_new_token"

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

    prompt = num_validation_images * [validation_prompt]
    images = pipeline(prompt, num_inference_steps=25).images
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
