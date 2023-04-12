from dataclasses import dataclass

import pyrallis

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImagesEmbeddingDataloader
from src.fast_inversion import fast_inversion
from src.fast_inversion.fast_inversion import train_epoch, train
from src.fast_inversion.config import TrainConfig, set_config

from src.fast_inversion.fast_inversion_model import SimpleModel, get_clip_image, SimpleCrossAttentionModel
from src.misc import compute

cfg = pyrallis.parse(config_class=TrainConfig)
set_config(cfg)

img_clip, img_processor = get_clip_image()
ds = ImagesEmbeddingDataset(split='train', download=False, image_processor=img_processor)
loader = ImagesEmbeddingDataloader(ds, batch_size=cfg.batch_size, shuffle=True)

eval_ds = ImagesEmbeddingDataset(split='eval', image_processor=img_processor)
eval_loader = ImagesEmbeddingDataloader(eval_ds, batch_size=cfg.batch_size * 2)

if cfg.model_type == 'simple':
    model = SimpleModel(len(ds.steps), img_clip)
if cfg.model_type == 'simplecross':
    model = SimpleCrossAttentionModel(len(ds.steps), img_clip)
model = model.to(compute.get_device())
fast_inversion.set_init_emb(ds.init_embd)

train(model, loader, eval_loader, cfg)
