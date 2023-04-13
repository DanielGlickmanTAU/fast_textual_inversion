from dataclasses import dataclass
import gc

import pyrallis
from tqdm import tqdm

from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImagesEmbeddingDataloader, CachedDataset
from src.fast_inversion import fast_inversion
from src.fast_inversion.fast_inversion import train_epoch, train
from src.fast_inversion.config import TrainConfig, set_config

from src.fast_inversion.fast_inversion_model import SimpleModel, get_clip_image, SimpleCrossAttentionModel
from src.misc import compute
import torch


def compute_embeddings(dataset, preprocess, clip_model):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            images = torch.stack([preprocess(img) for img in dataset[i]['images']])
            images = images.to(compute.get_device()).unsqueeze(0)
            embedding = clip_model(images)
            embedding = embedding.view(-1, 257, 1024)
            embeddings.append([x.squeeze(0).to('cpu').clone() for x in embedding.split(1)])
            # embeddings.append(embedding)
    return embeddings


cfg = pyrallis.parse(config_class=TrainConfig)
set_config(cfg)

img_clip, img_processor = get_clip_image()
ds = ImagesEmbeddingDataset(split=cfg.train_set, download=False, image_processor=img_processor)
eval_ds = ImagesEmbeddingDataset(split='eval', image_processor=img_processor)

if cfg.cache_clip:
    train_images = compute_embeddings(ds, img_processor, img_clip)
    ds_tmp = CachedDataset(ds, train_images)
    eval_images = compute_embeddings(eval_ds, img_processor, img_clip)
    eval_ds_tmp = CachedDataset(eval_ds, eval_images)
    image_encoder = lambda x: x
    del ds, eval_ds, img_clip, img_processor
    gc.collect()
    torch.cuda.empty_cache()

    ds = ds_tmp
    eval_ds = eval_ds_tmp

else:
    image_encoder = img_clip

loader = ImagesEmbeddingDataloader(ds, max_images_per_instance=cfg.max_images_per_instance, batch_size=cfg.batch_size,
                                   shuffle=True)
eval_loader = ImagesEmbeddingDataloader(eval_ds, max_images_per_instance=cfg.max_images_per_instance,
                                        batch_size=cfg.batch_size)

# need to move images to cpu and wrap dataset
if cfg.model_type == 'simple':
    model = SimpleModel(len(ds.steps), img_clip)
if cfg.model_type == 'simplecross':
    model = SimpleCrossAttentionModel(len(ds.steps), image_encoder, step_time_scale=cfg.step_time_scale,
                                      embedding_hidden_multiplier=cfg.embedding_hidden_multiplier,
                                      project_patches_dim=cfg.project_patches_dim)
model = model.to(compute.get_device())
fast_inversion.set_init_emb(ds.init_embd)

train(model, loader, eval_loader, cfg)
