import argparse
from dataclasses import dataclass

import pyrallis

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImagesEmbeddingDataloader
from src.fast_inversion.fast_inversion import train_epoch, train

from src.fast_inversion.fast_inversion_model import SimpleModel


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 100
    exp_name: str = 'fast_inversion_train'


cfg = pyrallis.parse(config_class=TrainConfig)
ds = ImagesEmbeddingDataset()
loader = ImagesEmbeddingDataloader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

model = SimpleModel(len(ds.steps))

# todo change to #train, cause need optimizer etc
train(model, loader, cfg)
