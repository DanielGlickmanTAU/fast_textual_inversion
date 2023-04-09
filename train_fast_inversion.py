from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset, ImagesEmbeddingDataloader
from src.fast_inversion.fast_inversion import train_epoch

from src.fast_inversion.fast_inversion_model import SimpleModel

ds = ImagesEmbeddingDataset()
loader = ImagesEmbeddingDataloader(ds, batch_size=32, shuffle=True, pin_memory=True)

model = SimpleModel(len(ds.steps))

# todo change to #train, cause need optimizer etc
train_epoch(model, loader)
