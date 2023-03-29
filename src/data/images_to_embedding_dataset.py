import json
import re

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
    def __init__(self, train_data_parent_dir, embedding_dir, as_json, flip_p=0.5):
        # dict with train,eval,test list of indicies
        d = json.load(open('split.json', 'r'))


        self.flip_transform = transforms.RandomHorizontalFlip(p=flip_p)
        paths = get_celeb_dirs(train_data_parent_dir)
        images = [self.get_images(p, False) for p in paths]
        embeddings = [get_celeb_embedding_from_train_data_dir(p) for p in paths]
        # todo: some embeddings will be None until all jobs are finished

    @staticmethod
    def get_images(train_data_dir, as_json):
        if as_json:
            state = json.load(open(train_data_dir, 'r'))
            images = concepts_datasets.get_images_from_dataset_state(state)
        else:
            images = [os.path.join(train_data_dir, file_path) for file_path in os.listdir(train_data_dir) if
                      'png' in file_path or 'jpg' in file_path or 'jpeg' in file_path]

        return images
