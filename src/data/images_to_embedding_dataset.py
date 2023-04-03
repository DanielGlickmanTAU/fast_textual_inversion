import json
import re

import PIL
from PIL import Image
from torchvision.transforms import transforms

from src.data import concepts_datasets
from src.data.concepts_datasets import get_project_dir
from src.misc import compute
import wandb
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset


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
    def __init__(self, split='train', base_dir='celebhq_dataset/', image_size=512, flip_p=0.5):
        self.image_size = image_size
        self.base_dir = base_dir
        self.split = json.load(open(base_dir + 'split.json', 'r'))[split]
        self.paths = self.get_dirs()

        self.flip_transform = transforms.RandomHorizontalFlip(p=flip_p)
        # images = [self.get_images(p, False) for p in paths]
        # embeddings = [get_celeb_embedding_from_train_data_dir(p) for p in paths]
        # todo: some embeddings will be None until all jobs are finished

    def get_dirs(self, ):
        datadir = os.path.join(self.base_dir, 'data')
        return [os.path.join(datadir, instance_id) for instance_id in self.split]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        images = [self.load_image(image_path) for image_path in self.get_images_path(path)]
        steps = None
        embeddings = [self.load_embedding(embd_path) for embd_path in self.get_embeddings(path, steps)]

        return {'images': images, 'path': path}

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

    def get_embeddings(self, instance_dir, steps):
        # todo filter right embeddings file names with steps
        # todo, dont list, just create exact names(-0.bin, -1.bin etc)
        embeddings_dir = os.path.join(instance_dir, 'embeddings')
        return [os.path.join(embeddings_dir, emb_file_name) for emb_file_name in os.listdir(embeddings_dir)]

    def load_embedding(self, embd_path):
        return torch.load(embd_path, map_location=torch.device('cpu'))['my_new_token']
