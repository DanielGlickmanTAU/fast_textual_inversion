#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys
import os

from src.CustomDiffusionPipeline import CustomDiffusionPipeline
from src.UnetCrossAttentionWrapper import UnetCrossAttentionWrapper
from src.tracing_text_encoder import TracingTextEncoder

sys.path.append(os.path.abspath('..'))
from src import run_utils
from src.data import utils
from src.misc import compute
import wandb
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler, UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, loss, distance_loss=None):
    logger.info(f"Saving embeddings to {save_path}")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds = learned_embeds.clone().detach().cpu()
    learned_embeds_dict = {args.placeholder_token: learned_embeds}
    if loss:
        learned_embeds_dict['loss'] = loss
    if distance_loss:
        learned_embeds_dict['distance_loss'] = distance_loss
    torch.save(learned_embeds_dict, save_path)
    return learned_embeds


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--mode', type=str, default='cross')

    parser.add_argument('--left_side', type=str, default='a woman in a red shirt')
    parser.add_argument('--right_side', type=str, default='and a man in a blue shirt')

    parser.add_argument('--steps_to_repeat', nargs='*', default=[])
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )

    parser.add_argument('--times_gen', type=int, default=100)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--wandb_project', type=str, default='composability tracing', help='wandb project name')

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        '--as_json',
        action='store_true',
        default=False,
        help="wheter the train_data_dir is a json with datasetstate object"

    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        '--num_images', type=int, default=0, help="Number of images"
    )
    parser.add_argument('--label_used', type=str, default='',
                        help="Label of the images in the datasets. used for analysis and debug")
    parser.add_argument('--dataset', type=str, help='name of dataset')

    parser.add_argument('--s3_upload', action='store_true')
    parser.add_argument('--start_runner', action='store_true')
    parser.add_argument('--mark_done', type=str, default=None)

    parser.add_argument('--distance_loss_alpha', type=float, default=0.)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


first_epoch = 0
global_step = 0


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, "../.gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = compute.get_cache_dir()
    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name, cache_dir=cache_dir, )

    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, cache_dir=cache_dir,
                                                  subfolder="tokenizer")

    # Load scheduler and models
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, cache_dir=cache_dir,
    #                                                 subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="text_encoder", revision=args.revision,
        # tokenizer=tokenizer
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="vae",
                                        revision=args.revision)
    # save some memory..
    vae.encoder = None
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir, subfolder="unet", revision=args.revision
    )

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # train_dataset = TextualInversionDataset(
    #     data_root=images,
    #     tokenizer=tokenizer,
    #     size=args.resolution,
    #     placeholder_token=args.placeholder_token,
    #     repeats=args.repeats,
    #     learnable_property=args.learnable_property,
    #     center_crop=args.center_crop,
    #     set="train",
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
    #     pin_memory=True
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     # args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.

    # text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     text_encoder, optimizer, train_dataloader, lr_scheduler
    # )

    text_encoder = accelerator.prepare(
        text_encoder
    )

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and 'wandb' in args.report_to:
        accelerator.init_trackers(args.wandb_project, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global global_step
    global first_epoch
    # Potentially load in the weights and states from a previous save
    resume_step = None
    if args.resume_from_checkpoint:
        first_epoch, global_step, resume_step = resume_from_checkpoint(accelerator, args, first_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    do_validation(accelerator, args, cache_dir, text_encoder, unet, vae, tokenizer, times_gen=args.times_gen)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

    if args.mark_done:
        run_utils.mark_id_done(args.mark_done)
    if args.start_runner:
        run_utils.run_if_not_running()

    if args.s3_upload:
        _, zipname = os.path.split(args.output_dir)
        print(f'uploading to s3 {zipname}')
        utils.s3_zip_and_upload(args.output_dir, zipname + '.zip')
        print('done uploading')

    exit()


def do_validation(accelerator, args, cache_dir, text_encoder, unet, vae, tokenizer, times_gen=1):
    # args.validation_prompt = "a woman in a red shirt and a man in a blue shirt"
    mode = args.mode

    # if mode == 'cross' or mode == 'causal':
    if mode == 'cross':
        args.validation_prompt = args.left_side + ' ' + args.right_side

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    pipeline = CustomDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir=cache_dir,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        vae=accelerator.unwrap_model(vae),
        tokenizer=tokenizer,
        revision=args.revision,
        requires_safety_checker=False,
        num_repeats=3,
        # steps_to_repeat=[23]
        steps_to_repeat=[int(x) for x in args.steps_to_repeat],
        std=args.std,
    )
    pipeline.text_encoder = TracingTextEncoder(pipeline.text_encoder, tokenizer, mode, args.left_side, args.right_side)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None
    # run inference
    generator = (
        None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    prompt = args.num_validation_images * [args.validation_prompt]

    for _ in range(times_gen):
        images = pipeline(prompt, num_inference_steps=25, generator=generator).images
        # images = pipeline(prompt, num_inference_steps=5, generator=generator).images
        for tracker in accelerator.trackers:
            # if tracker.name == "tensorboard":
            #     np_images = np.stack([np.asarray(img) for img in images])
            #     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: ")
                            for i, image in enumerate(images)
                        ]
                    }
                )
    del pipeline
    torch.cuda.empty_cache()


def resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch):
    global first_epoch
    global global_step
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    return first_epoch, global_step, resume_step


if __name__ == "__main__":
    main()
