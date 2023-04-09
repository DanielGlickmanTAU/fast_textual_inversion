from dataclasses import dataclass

import pyrallis

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImagesEmbeddingDataloader
from src.fast_inversion import fast_inversion
from src.fast_inversion.fast_inversion import train_epoch, train, TrainConfig

from src.fast_inversion.fast_inversion_model import SimpleModel

cfg = pyrallis.parse(config_class=TrainConfig)
ds = ImagesEmbeddingDataset(split='test')
loader = ImagesEmbeddingDataloader(ds, batch_size=cfg.batch_size, shuffle=True)

eval_ds = ImagesEmbeddingDataset(split='eval')
eval_loader = ImagesEmbeddingDataloader(eval_ds, batch_size=cfg.batch_size * 2)

model = SimpleModel(len(ds.steps))

fast_inversion.set_init_emb(ds.init_embd)

train(model, loader, eval_loader, cfg)
