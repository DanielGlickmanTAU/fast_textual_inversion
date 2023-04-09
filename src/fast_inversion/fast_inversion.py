from collections import defaultdict

import torch.nn.functional as F
import torch

from src.data.images_to_embedding_dataset import ImageEmbeddingInput
from src.fast_inversion.wandb_helper import init_wandb
import tqdm


def train(model, data_loader, args):
    wandb = init_wandb(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        train_epoch(model, data_loader, optimizer, wandb)


def train_epoch(model, data_loader, optimizer, wandb, teacher_force=True):
    # images: (B,n, d) where n is num images
    # embeddings: (B,k,d) where k = 5000/n_steps
    for batch in tqdm.tqdm(data_loader):
        # todo: here encode images with clip etc
        train_step(model, batch, optimizer, wandb, teacher_force)


def train_step(model, input: ImageEmbeddingInput, optimizer, wandb, teacher_force):
    images, embeddings, n_steps = input.images, input.embeddings, len(input.embeddings)
    stats = {}
    for step in range(n_steps - 1):
        if teacher_force or step == 0:
            x_emb = embeddings[step]
        else:
            x_emb = emb_predicted.detach().clone()

        emb_predicted = model(images, x_emb, torch.tensor(step))
        emb_target = embeddings[step + 1]

        loss = F.mse_loss(emb_predicted.float(), emb_target.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        stats[f'loss_step{step}'] = loss.item()
    if wandb:
        loss_avg = sum(stats.values()) / len(stats)
        stats['loss_avg'] = loss_avg
        wandb.log(stats)


def eval(model, images, n_steps):
    embedding = init_emb  # save batch.embeddings[0][0]
    for step in range(n_steps - 1):
        embedding = model(images, embedding, step)
