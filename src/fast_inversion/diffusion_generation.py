import torch

from src.misc import compute


def generate_images(args,fast_embedder):
    # TODO:
    # make sure loading learned embedding into model... can look at others code..
    # probably should overwrite pipeline to use my own embeddings with fast embedder

    cache_dir = compute.get_cache_dir()
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, cache_dir=cache_dir,
                                              subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="vae",
                                        revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="unet", revision=args.revision
    )

    args.validation_prompt = args.validation_prompt.replace('{}', args.placeholder_token)
    if args.placeholder_token not in args.validation_prompt:
        args.validation_prompt = args.validation_prompt.strip() + ' ' + args.placeholder_token

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir,
        text_encoder=accelerator.unwrap_model(text_encoder),
        unet=accelerator.unwrap_model(unet),
        vae=accelerator.unwrap_model(vae),
        tokenizer=tokenizer,
        revision=args.revision,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None
    # run inference
    generator = (
        None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    prompt = args.num_validation_images * [args.validation_prompt]
    images = pipeline(prompt, num_inference_steps=25, generator=generator).images
    for tracker in accelerator.trackers:
        # if tracker.name == "tensorboard":
        #     np_images = np.stack([np.asarray(img) for img in images])
        #     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del text_encoder
    del vae
    del unet
    del pipeline
    torch.cuda.empty_cache()
