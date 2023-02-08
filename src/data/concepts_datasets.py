import os
import pickle
from dataclasses import dataclass
from typing import Union

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy
import torchvision.transforms as transforms
from collections import defaultdict
import random
from collections import defaultdict


def get_project_dir():
    pwd = os.popen('pwd').read()
    # no / at the end
    return pwd[:pwd.index('/fast_textual_inversion') + len('/fast_textual_inversion')]


def get_cars_dir():
    return f'{get_project_dir()}/cars_data'


def get_cars_ds():
    ds = torchvision.datasets.StanfordCars(root=get_cars_dir(),
                                           download=True)  # ,transform=transforms.ToTensor())
    return ds


def get_food_dir():
    return f'{get_project_dir()}/food_data'


def get_celeb_dir():
    return f'{get_project_dir()}/celeba_hq_data'


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


def label_to_indices(dataset):
    res = defaultdict(list)
    for i, data in enumerate(dataset):
        res[data[1]].append(i)
    return res


# quick is for fast debugging
def select_k_random_indices_with_label(dataset, label, k, min_allowed_examples=1, quick=False):
    label_index = []
    for i, data in enumerate(dataset):
        if data[1] == label:
            label_index.append(i)
            if quick and len(label_index) == k:
                break
    return get_random_k_from_list_safe(label_index, k, min_allowed_examples)


def get_random_k_from_list_safe(label_index, k, min_allowed_examples=1):
    k = min(k, len(label_index))
    if k < min_allowed_examples:
        raise ValueError(
            f'needs at least {min_allowed_examples} images with label {label}, but have only {len(label_index)}')
    selected_index = random.sample(label_index, k)
    return selected_index


def get_datasetstate_with_k_random_indices_with_label(ds_name, label, k, min_allowed_examples=1, split='train',
                                                      quick=False):
    def cache_results():
        filename = ds_name + '_label_cache.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as fp:
                return pickle.load(fp)
        else:
            result = label_to_indices(ds)
            with open(filename, 'wb') as fp:
                pickle.dump(result, fp)
            return result

    ds = get_ds(ds_name, split)

    indices = select_k_random_indices_with_label(ds, label, k, min_allowed_examples, quick=quick)
    # indices = cache_results()
    # indices = indices[label]
    # get_random_k_from_list_safe(indices, k, min_allowed_examples)
    return DatasetState(ds_name, indices, split)


def get_ds(ds_name, split):
    if ds_name == 'food':
        ds = get_food_ds(split=split)
    elif ds_name == 'cars':
        ds = get_cars_ds()
    else:
        raise ValueError(f'unknown ds_name {ds_name}')
    return ds


def get_images_from_dataset_state(dataset_state: Union[DatasetState, dict]):
    if isinstance(dataset_state, DatasetState):
        dataset_state = dataset_state.__dict__
    img_index = 0

    ds = get_ds(dataset_state['name'], dataset_state['split'])
    return [ds[i][img_index] for i in dataset_state['indices']]


def show_img(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.cpu().squeeze()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
    plt.imshow(tensor)
    plt.show()
