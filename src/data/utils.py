import torch


def get_embedding_checkpoint(args, num_checkpoint):
    path = f'{args.output_dir}/learned_embeds-steps-{num_checkpoint}.bin'
    ckpt = torch.load(path)
    emb = ckpt['my_new_token']
    return emb


def set_embedding(text_encoder, placeholder_token_id, embedding):
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[placeholder_token_id] = embedding.to(
            text_encoder.device)


def interpolate_embedding(text_encoder, args, placeholder_token_id, num_checkpoint, checkpoint_weight):
    assert 0. <= checkpoint_weight <= 1.
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[placeholder_token_id] = (
                checkpoint_weight * get_embedding_checkpoint(args, num_checkpoint) + (1 - checkpoint_weight)
                * get_embedding_checkpoint(args, 0)).to(text_encoder.device)


import boto3
import zipfile
import os
import zipfile


def s3_zip_and_upload(output_dir, zipname):
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), arcname=file)

    # always end with .zip
    zipname = zipname.replace('.zip', '') + '.zip'

    zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    zipdir(output_dir, zipf)
    zipf.close()

    s3_upload(zipname)


def s3_upload(filename, print_progress=False):
    c = [0]

    def progress(bytes_transferred):
        if print_progress:
            c[0] = c[0] + 1
            print(f'{bytes_transferred * c[0]} bytes transferred')

    s3 = boto3.client('s3')
    # Upload the zip file to Amazon S3
    with open(filename, 'rb') as data:
        s3.upload_fileobj(data, 'fast-inversion', filename, Callback=progress)


def get_all_keys_with_prefix(bucket_name='fast-inversion', objects_name_prefix='celeb', s3=boto3.client('s3')):
    paginator = s3.get_paginator('list_objects')
    result_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=objects_name_prefix,
        PaginationConfig={
            'PageSize': 1000
        }
    )
    l = []
    for page in result_iterator:
        l.extend([x['Key'] for x in page['Contents']])
    return l
    # response = s3.list_objects_v2(Bucket=bucket_name, Prefix=objects_name_prefix)
    # return [x['Key'] for x in response['Contents']]


def download_file_and_extract_zip(s3_client, filekey, bucket_name='fast-inversion'):
    # download
    download_path = f's3_data/{filekey}'
    extract_path = download_path.replace('.zip', '')

    if os.path.exists(extract_path):
        print(f'{extract_path} exists, skipping download')
        return False
    s3_client.download_file(bucket_name, filekey, download_path)
    print(f'downloaded {filekey} into {download_path}')
    # extract
    return extract_zip_to_path(download_path, extract_path)


def extract_zip_to_path(download_path, extract_path):
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path


import json


def extract_image_id(extracted_path):
    with open(f'{extracted_path}/args.json', 'r') as f:
        d = json.load(f)
        return d['mark_done']


def celebhq_flow():
    s3 = boto3.client('s3')
    keys = get_all_keys_with_prefix(objects_name_prefix='celeb', s3=s3)
    for key in keys:
        extracted_path = download_file_and_extract_zip(s3, key)
        if not extracted_path:
            continue
        celeb_img_id = extract_image_id(extracted_path)
        celeb_dataset_dir = f'celebhq_dataset/data/{celeb_img_id}'
        embeddings_dir = f'{celeb_dataset_dir}/embeddings'
        print(f'moving embeddings to {embeddings_dir}')
        os.mkdir(embeddings_dir)
        os.system(f'cp -r {extracted_path}/*.bin {embeddings_dir}')


import random

celebhq_dir = 'celebhq_dataset/data/'


def create_splits():
    images = os.listdir(celebhq_dir)
    random.shuffle(images)
    train_part = int(len(images) * 0.9)
    eval_part = int(len(images) * 0.95)
    return {'train': images[:train_part], 'eval': images[train_part:eval_part], 'test': images[eval_part:]}
