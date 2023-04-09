import torch.nn.functional as F
import torch


def train(model, data_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


def train_epoch(model, data_loader, teacher_force=True):
    # images: (B,n, d) where n is num images
    # embeddings: (B,k,d) where k = 5000/n_steps
    for batch in data_loader:
        n_steps = len(batch.embeddings)
        # todo: here encode images with clip etc
        train_step(model, batch.images, batch.embeddings, n_steps, teacher_force)


def train_step(model, images, embeddings, n_steps, teacher_force):
    for step in range(n_steps - 1):
        if teacher_force or step == 0:
            x_emb = embeddings[step]
        else:
            x_emb = emb_predicted.detach().clone()

        emb_target = embeddings[step + 1]
        emb_predicted = model(images, x_emb, step)

        loss = F.mse_loss(emb_predicted.float(), emb_target.float(), reduction="mean")
    # accelerator.clip_grad_norm_(model.parameters(), 1.0)
    # optimizer.step()
    # lr_scheduler.step()
    # optimizer.zero_grad()


def eval(model, images, n_steps):
    embedding = init_emb  # save batch.embeddings[0][0]
    for step in range(n_steps - 1):
        embedding = model(images, embedding, step)
