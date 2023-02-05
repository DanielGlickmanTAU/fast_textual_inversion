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


def s3_upload(output_dir, zipname):
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), arcname=file)

    s3 = boto3.client('s3')

    zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    zipdir(output_dir, zipf)
    zipf.close()

    # Upload the zip file to Amazon S3
    with open(zipname, 'rb') as data:
        s3.upload_fileobj(data, 'fast-inversion', zipname)
