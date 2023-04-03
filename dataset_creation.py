from src.data.images_to_embedding_dataset import ImagesEmbeddingDataset
from src.data.utils import celebhq_flow, create_splits, s3_upload, celebhq_dir

# celebhq_flow()
import json

# splits = create_splits()
# print(splits)
# create split file
# json.dump(splits,open('celebhq_dataset/split.json','w'))
# s3_upload('celebhq_dataset/', 'dataset_celebhq.zip')


ds = ImagesEmbeddingDataset()
ds[0]
