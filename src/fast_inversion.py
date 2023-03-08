import torch.nn.functional as F
import torch

# torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
#         pin_memory=True
#     )
init_emb = None


def train_epoch(model, data_loader, n_steps, teacher_force=True):
    # images: (B,n, d) where n is num images
    # embeddings: (B,k,d) where k = 5000/n_steps
    for images, embeddings in data_loader:
        train_step(model, images, embeddings, n_steps, teacher_force)


def train_step(model, images, embeddings, n_steps, teacher_force):
    for step in range(n_steps - 1):
        if teacher_force or step == 0:
            x_emb = embeddings[:, step, :]
        else:
            x_emb = emb_predicted.detach().clone()

        emb_predicted = model(images, x_emb, step)
        emb_target = embeddings[:, step + 1, :]

        loss = F.mse_loss(emb_predicted.float(), emb_target.float(), reduction="mean")


def eval(model, images, n_steps):
    embedding = init_emb  # learned vector
    for step in range(n_steps - 1):
        embedding = model(images, embedding, step)
