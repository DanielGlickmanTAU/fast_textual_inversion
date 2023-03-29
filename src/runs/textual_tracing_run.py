import random
import time

import os
import json
from src.data.concepts_datasets import get_datasetstate_with_k_random_indices_with_label, get_project_dir, get_food_dir, \
    get_celeb_dir
from src.misc.gridsearch import gridsearch
from src.misc.slurm import run_on_slurm

food_num_classes = 101
food_split = 'test'

celeba_dir = get_celeb_dir()
# dataset = 'celeba'
dataset = 'food'
# quick = True
# validation_epochs = 10
quick = False
# validation_epochs = 50000000000000
# s3_upload = True
# max_train_steps = 3000
validation_epochs = 50
overall_runs = 1
s3_upload = False
max_train_steps = 1000
# min_num_class = 4
# max_num_class = 8
min_num_class = 1
max_num_class = 1
params = {
    '--enable_xformers_memory_efficient_attention': '',
    '--pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
    '--placeholder_token': 'my_new_token',
    '--resolution': 512,

    '--train_batch_size': 4,
    '--distance_loss_alpha': 0.1,
    '--gradient_accumulation_steps': 4,
    '--max_train_steps': max_train_steps,
    '--learning_rate': 5.0e-04,
    '--scale_lr': '',
    '--lr_scheduler': "constant",
    '--lr_warmup_steps': 0,
    '--report_to': 'wandb',
    '--save_steps': 10,
    # '--validation_prompt': '"An image of "',
    '--num_validation_images': 1,
    '--validation_epochs': validation_epochs,
    '--dataset': dataset,
    '--mode': 'causal',
    # '--validation_prompt': '"An image of "',

    # '--mode': 'cross',
    # '--mode': 'no_eos'
    # '--mode': 'eos'
    # '--left_side': '"a man in a blue shirt"',
    # '--left_side': '"a woman in a red shirt"',
    # '--right_side': '"a woman in a red shirt"',
    # '--left_side': '"a red banana"',
    # '--left_side': '"a red ball"',
    '--left_side': '"a red dog"',
    # '--right_side': '"and a yellow tomato"',
    # '--right_side': '"on a green box"',
    # '--right_side': '"a red ball"',
    # '--left_side': '"on a green box"',
    '--right_side': '"on a green bed"',
    # '--right_side': '"and a woman in a red shirt"'
    # '--right_side': '"and a man in a blue shirt"'

    # '--left_side': '"and a woman in a red shirt"'
    # '--right_side': '"a man in a blue shirt"',
    # '--left_side': '"and a man in a blue shirt"',
    # '--right_side': '"a woman in a red shirt"',

}

params_for_exp = {

}

os.chdir('..')
job_name = '''textual_tracing.py'''
ids = []
for _ in range(overall_runs):

    for p in gridsearch(params, params_for_exp):
        id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=1, wandb=False)
        ids.append(id)
    print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
