import torch
import torchvision
import matplotlib.pyplot as plt
import numpy
import torchvision.transforms as transforms
from collections import defaultdict
import random


def get_cars_ds():
    ds = torchvision.datasets.StanfordCars(root='./cars_data', download=True)  # ,transform=transforms.ToTensor())
    return ds


def get_food_ds(split='test'):
    return torchvision.datasets.Food101(root='./food_data',
                                        download=True, split=split)  # ,transform=transforms.ToTensor())


def get_celeba_ds():
    return torchvision.datasets.CelebA(root='./celeba_data', download=True)


def select_random_items(dataset, label, k):
    data_items = []
    label_index = []
    for i, data in enumerate(dataset):
        if data[1] == label:
            data_items.append(data[0])
            label_index.append(i)
    selected_index = random.sample(label_index, k)
    selected_data = [dataset[i] for i in selected_index]
    return selected_data


def show_img(tensor):
    if type(tensor) == torch.Tensor and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    plt.imshow(tensor)
    plt.show()
