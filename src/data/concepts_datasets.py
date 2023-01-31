import os
from dataclasses import dataclass
from typing import Union

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy
import torchvision.transforms as transforms
from collections import defaultdict
import random


def get_project_dir():
    pwd = os.popen('pwd').read()
    # no / at the end
    return pwd[:pwd.index('/fast_textual_inversion') + len('/fast_textual_inversion')]


def get_cars_ds():
    ds = torchvision.datasets.StanfordCars(root=f'{get_project_dir()}/cars_data',
                                           download=True)  # ,transform=transforms.ToTensor())
    return ds


def get_food_dir():
    return f'{get_project_dir()}/food_data'


def get_food_ds(split='test'):
    return torchvision.datasets.Food101(root=get_food_dir(),
                                        download=True, split=split)  # ,transform=transforms.ToTensor())


def get_celeba_ds():
    return torchvision.datasets.CelebA(root=f'{get_project_dir()}/celeba_data', download=True)


@dataclass
class DatasetState:
    name: str
    indices: list
    split: str = 'train'


# quick is for fast debugging
def select_k_random_indices_with_label(dataset, label, k, min_allowed_examples=3, quick=False):
    label_index = []
    for i, data in enumerate(dataset):
        if data[1] == label:
            label_index.append(i)
            if quick and len(label_index) == k:
                break
    k = min(k, len(label_index))
    if k < min_allowed_examples:
        raise ValueError(
            f'needs at least {min_allowed_examples} images with label {label}, but have only {len(label_index)}')
    selected_index = random.sample(label_index, k)
    return selected_index


def get_datasetstate_with_k_random_indices_with_label(ds_name, label, k, min_allowed_examples=3, split='train',
                                                      quick=False):
    if ds_name == 'food':
        ds = get_food_ds(split=split)

    indices = select_k_random_indices_with_label(ds, label, k, min_allowed_examples, quick=quick)
    return DatasetState(ds_name, indices, split)


def get_images_from_dataset_state(dataset_state: Union[DatasetState, dict]):
    if isinstance(dataset_state, DatasetState):
        dataset_state = dataset_state.__dict__
    img_index = 0
    if dataset_state['name'] == 'food':
        ds = get_food_ds(split=dataset_state['split'])
        return [ds[i][img_index] for i in dataset_state['indices']]


def show_img(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.cpu().squeeze()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
    plt.imshow(tensor)
    plt.show()
