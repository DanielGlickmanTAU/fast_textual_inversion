import random

import torch.nn.functional as F
import torch

from src.data.images_to_embedding_dataset import ImageEmbeddingInput
from src.fast_inversion import diffusion_generation
from src.fast_inversion.config import TrainConfig
from src.fast_inversion.fast_inversion_model import get_clip_image
from src.fast_inversion.wandb_helper import init_wandb
import tqdm

from src.misc import compute

init_emb = None

device = compute.get_device()


def set_init_emb(init_emb_):
    global init_emb
    init_emb = init_emb_


def get_embedding_for_image(model, sample):
    images = torch.stack(sample['images']).unsqueeze(0)
    steps = len(sample['embeddings'])
    return eval_model(images, model, steps, init_emb.unsqueeze(0))


def train(model, train_loader, eval_dataloader, args: TrainConfig):
    model = model.to(device)
    wandb = init_wandb(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        if wandb:
            wandb.log({'epoch': epoch})
        teacher_force = random.uniform(0, 1) > (
                epoch / args.epochs) if args.teacher_force == 'linear' else bool(args.teacher_force)
        train_epoch(model, train_loader, optimizer, wandb, teacher_force)
        if args.validate_loss:
            eval_model_epoch(model, eval_dataloader, wandb)
        if (epoch + 1) % args.log_images_every_n_epochs == 0 and wandb:
            for i in range(args.num_persons_images_to_log):
                sample = eval_dataloader.dataset[i]
                emb = get_embedding_for_image(model, sample)
                torch.cuda.empty_cache()
                diffusion_generation.generate_images(emb, sample['path'], wandb, args)


def train_epoch(model, data_loader, optimizer, wandb, teacher_force=True):
    # images: (B,n, d) where n is num images
    # embeddings: (B,k,d) where k = 5000/n_steps
    for batch in tqdm.tqdm(data_loader):
        train_step(model, batch, optimizer, wandb, teacher_force)


def train_step(model, input: ImageEmbeddingInput, optimizer, wandb, teacher_force):
    stats = {}

    input = input.to(device)
    images, embeddings, n_steps, is_real = input.images, input.embeddings, len(input.embeddings), input.is_real
    encoded_images = model.encode_images(images)
    for step in range(n_steps - 1):
        if teacher_force or step == 0:
            x_emb = embeddings[step]
        else:
            x_emb = emb_predicted.detach().clone()

        emb_predicted = model(encoded_images, x_emb, torch.tensor(step, device=device), is_real=is_real)
        emb_target = embeddings[step + 1]

        loss = F.mse_loss(emb_predicted.float(), emb_target.float(), reduction="mean")
        if loss <= 1.:
            loss.backward()
            optimizer.step()
        else:
            print(f"Loss too high: {loss}. paths: {input.paths}")
        optimizer.zero_grad()

        stats[f'loss_step{step}'] = loss.item()
    if wandb:
        loss_avg = sum(stats.values()) / len(stats)
        stats['loss_avg'] = loss_avg
        wandb.log(stats)


def eval_model_epoch(model, loader, wandb):
    total_loss = 0
    total_items = 0
    for batch in loader:
        batch = batch.to(device)
        loss = eval_loss(model, batch)
        n = len(batch)
        total_loss += loss.item() * n
        total_items += n
    total_loss = total_loss / total_items
    if wandb:
        wandb.log({'eval_final_loss': total_loss})


def eval_loss(model, input: ImageEmbeddingInput):
    images, embeddings, n_steps = input.images, input.embeddings, len(input.embeddings)
    bs = images.shape[0]
    x_emb = init_emb.expand(bs, -1)
    x_emb = eval_model(images, model, n_steps, x_emb)
    emb_target = embeddings[-1]
    loss = F.mse_loss(x_emb.float(), emb_target.float(), reduction="mean")
    return loss


@torch.no_grad()
def eval_model(images, model, n_steps, x_emb):
    x_emb = x_emb.to(device)
    images = images.to(device)
    encoded_images = model.encode_images(images)
    for step in range(n_steps - 1):
        emb_predicted = model(encoded_images, x_emb, torch.tensor(step, device=device))
        x_emb = emb_predicted.detach().clone()
    return x_emb
