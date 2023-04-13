import torch

from src.fast_inversion.config import TrainConfig
from src.fast_inversion.fast_inversion_model import get_clip_tokenizer, get_clip_text, get_vae, get_unet, \
    set_embedding_in_text_encoder, get_diffusion_pipeline, placeholder_token
from src.misc import compute
import wandb

# num_inference_steps = 25
num_inference_steps = 10


@torch.no_grad()
def generate_images(embedding, path, experiment, config: TrainConfig):
    # TODO:
    # make sure loading learned embedding into model... can look at others code..
    # probably should overwrite pipeline to use my own embeddings with fast embedder

    tokenizer = get_clip_tokenizer()

    text_encoder = get_clip_text()
    vae = get_vae()

    unet = get_unet()

    set_embedding_in_text_encoder(embedding, text_encoder, tokenizer)

    validation_prompt = f"An image of {placeholder_token}"

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = get_diffusion_pipeline(text_encoder, tokenizer, unet, vae)

    prompt = config.num_images_per_person_to_log * [validation_prompt]
    images = pipeline(prompt, num_inference_steps=num_inference_steps).images
    if experiment is not None:
        print('logging image experiment')
        experiment.log(
            {
                "validation": [
                    wandb.Image(image, caption=f"path: {path.replace('/', '_')}. {i}: {validation_prompt}")
                    for i, image in enumerate(images)
                ]
            })

    del text_encoder
    del vae
    del unet
    del pipeline
    torch.cuda.empty_cache()
