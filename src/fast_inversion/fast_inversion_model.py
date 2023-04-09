import torch

embedding_size = 768


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
