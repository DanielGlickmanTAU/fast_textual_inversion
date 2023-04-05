import dataclasses
from collections import OrderedDict
from typing import List

import torch

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset
from src.data.utils import celebhq_flow, create_splits, s3_upload, celebhq_dir

# celebhq_flow()
import json


# splits = create_splits()
# print(splits)
# create split file
# json.dump(splits,open('celebhq_dataset/split.json','w'))
# s3_upload('celebhq_dataset/', 'dataset_celebhq.zip')

def pad_images(seq, max_length):
    pad_image = torch.zeros_like(seq[0])
    new_seq = seq + [pad_image] * (max_length - len(seq))
    is_real = torch.zeros(max_length, dtype=int)
    is_real[:len(seq)] = 1
    return torch.stack(new_seq), is_real


def batch_embeddings(batch):
    num_embeddings = len(batch[0]['embeddings'])
    embeddings = []
    for embedding_index in range(num_embeddings):
        embs = [x['embeddings'][embedding_index] for x in batch]
        embeddings.append(torch.stack(embs))
    return embeddings


# Custom collate function
def custom_collate(batch):
    max_length = max([len(item['images']) for item in batch])
    padded_input = [pad_images(item['images'], max_length) for item in batch]
    padded_image, is_real = zip(*padded_input)
    padded_images = torch.stack(padded_image)
    is_real = torch.stack(is_real)

    embeddings_to_step = batch_embeddings(batch)
    # return {'images': padded_images, 'is_real': is_real, 'embeddings': embeddings_to_step}
    return ImageEmbeddingInput(padded_images, is_real, embeddings_to_step)


@dataclasses.dataclass
class ImageEmbeddingInput(OrderedDict):
    # shape (B,max_images, 3,512,512)
    images: torch.Tensor
    # shape (B,max_images)
    is_real: torch.Tensor
    # list of size num_embeddings(0 entry is the initial embedding, i.e "person"). Each entry is size (B,d)
    embeddings: List[torch.Tensor]


ds = ImagesEmbeddingDataset(steps=[*range(5001)][::20][:2000])
ds[0]
loader = torch.utils.data.DataLoader(
    ds, batch_size=200, shuffle=True, pin_memory=True, collate_fn=custom_collate
)

for x in loader:
    print(x)
    print([(a - b).norm(2, dim=-1).mean() for a, b in zip(x.embeddings[:-1], x.embeddings[1:])])
