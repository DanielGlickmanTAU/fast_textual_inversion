import torch

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImageEmbeddingInput, ImagesEmbeddingDataloader

# celebhq_flow()

# splits = create_splits()
# print(splits)
# create split file
# json.dump(splits,open('celebhq_dataset/split.json','w'))
# s3_upload('celebhq_dataset/', 'dataset_celebhq.zip')
from src.fast_inversion import train_epoch

ds = ImagesEmbeddingDataset(
    steps=[0, 40, 100, 180, 280, 400, 520, 660, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200,
           2400,
           2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]
)
ds[0]
# loader = torch.utils.data.DataLoader(
#     ds, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate
# )

loader = ImagesEmbeddingDataloader(ds, batch_size=32, shuffle=True, pin_memory=True)

import matplotlib.pyplot as plt


def show(x):
    diffs = [(a - b).norm(2, dim=-1).mean().item() for a, b in zip(x.embeddings[:-1], x.embeddings[1:])]
    plt.plot(diffs)
    plt.show()


train_epoch(None, loader)

for x in loader:
    print([(a - b).norm(2, dim=-1).mean() for a, b in zip(x.embeddings[:-1], x.embeddings[1:])])
