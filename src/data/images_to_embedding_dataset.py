import dataclasses
import json
import os
import re
from collections import OrderedDict
from typing import List

import PIL
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.data import concepts_datasets
from src.data.concepts_datasets import get_project_dir
from src.data.utils import extract_zip_to_path


def get_celeb_dirs(celeb_parent_dir=get_project_dir() + '/celebhq'):
    return [os.path.join(celeb_parent_dir, file_path) for file_path in os.listdir(celeb_parent_dir)]


def get_celeb_embedding_from_train_data_dir(train_data_dir):
    def find_embedding_dir(train_data_dir):
        inner_dir = [os.path.join(train_data_dir, d) for d in os.listdir(train_data_dir) if
                     os.path.isdir(os.path.join(train_data_dir, d))]
        assert len(inner_dir) == 1, f'{train_data_dir} has more than one embedding dirs'
        inner_dir = inner_dir[0]
        return inner_dir

    train_data_dir = train_data_dir + '/training_output'
    embedding_dir = find_embedding_dir(train_data_dir)

    learned_embeds_files = []
    for f in os.listdir(embedding_dir):
        match = re.match(r'learned_embeds-steps-(\d+).bin', f)
        if match:
            number = match.group(1)
            learned_embeds_files.append((os.path.join(embedding_dir, f), int(number)))
    return sorted(learned_embeds_files, key=lambda x: x[1])


def embedding_bin_file_path_to_tensor(path):
    ckpt = torch.load(path) if torch.cuda.is_available() else torch.load(path, map_location=torch.device('cpu'))
    return ckpt['my_new_token']


class ImagesEmbeddingDataset(Dataset):
    def __init__(self, split='train', base_dir='celebhq_dataset/', image_size=512, flip_p=0.5, steps=None,
                 download=False):
        assert split in ['train', 'eval', 'test']
        self.image_size = image_size
        self.base_dir = base_dir
        if download:
            self.download()
        self.split = json.load(open(base_dir + 'split.json', 'r'))[split]
        self.paths = self.get_dirs()
        if steps is not None:
            self.steps = steps
        else:
            self.steps = [0, 40, 100, 180, 280, 400, 520, 660, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200,
                          2400,
                          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]
        self.flip_transform = transforms.RandomHorizontalFlip(p=flip_p)
        # todo: some embeddings will be None until all jobs are finished
        self.init_embd = self[0]['embeddings'][0].detach().clone()

    def get_dirs(self, ):
        datadir = os.path.join(self.base_dir, 'data')
        all_dirs = [os.path.join(datadir, instance_id) for instance_id in self.split]
        dirs_with_embeddings = [dir for dir in all_dirs if os.path.exists(dir + '/embeddings')]
        print(f'WARNING: only {len(dirs_with_embeddings)} out of {len(all_dirs)} instances have embeddings')
        print(f'WARNING: only {len(dirs_with_embeddings)} out of {len(all_dirs)} instances have embeddings')
        print(f'WARNING: only {len(dirs_with_embeddings)} out of {len(all_dirs)} instances have embeddings')
        return dirs_with_embeddings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        images = [self.load_image(image_path) for image_path in self.get_images_path(path)]
        embeddings = [self.load_embedding(embd_path) for embd_path in self.get_embeddings(path)]

        return {'images': images, 'path': path, 'embeddings': embeddings}

    @staticmethod
    def get_images_path(instance_dir, as_json=False):
        if as_json:
            state = json.load(open(instance_dir, 'r'))
            images = concepts_datasets.get_images_from_dataset_state(state)
        else:
            images = [os.path.join(instance_dir, file_path) for file_path in os.listdir(instance_dir) if
                      'png' in file_path or 'jpg' in file_path or 'jpeg' in file_path]

        return images

    def load_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        image = image.resize((self.image_size, self.image_size), resample=PIL.Image.Resampling.BICUBIC)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)

    def get_embeddings(self, instance_dir):
        embeddings_dir = os.path.join(instance_dir, 'embeddings')
        return [os.path.join(embeddings_dir, f'learned_embeds-steps-{step}.bin') for step in self.steps]

    def load_embedding(self, embd_path):
        return torch.load(embd_path, map_location=torch.device('cpu'))['my_new_token']

    def download(self):
        link = 'https://fast-inversion.s3.amazonaws.com/dataset_celebhq.zip'
        zip = 'dataset_celebhq.zip'
        os.system(f'curl --get {link} >> {zip}')
        extract_zip_to_path(zip, self.base_dir)


@dataclasses.dataclass
class ImageEmbeddingInput(OrderedDict):
    # shape (B,max_images, 3,512,512)
    images: torch.Tensor
    # shape (B,max_images)
    is_real: torch.Tensor
    # list of size num_embeddings(0 entry is the initial embedding, i.e "person"). Each entry is size (B,d)
    embeddings: List[torch.Tensor]

    def __len__(self):
        return len(self.images)

    def to(self, device):
        return ImageEmbeddingInput(images=self.images.to(device),
                                   is_real=self.is_real.to(device),
                                   embeddings=[emb.to(device) for emb in self.embeddings]
                                   )


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


class ImagesEmbeddingDataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(ImagesEmbeddingDataloader, self).__init__(collate_fn=custom_collate, *args, **kwargs)
