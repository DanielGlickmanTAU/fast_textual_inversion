import random
import time

import os
import json
from src.data.concepts_datasets import get_datasetstate_with_k_random_indices_with_label, get_project_dir, get_food_dir
from src.misc.gridsearch import gridsearch
from src.misc.slurm import run_on_slurm

food_num_classes = 101
food_split = 'test'

params = {
    '--pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',

    '--placeholder_token': '<my_new_token>',
    '--initializer_token': 'food',
    '--resolution': 512,
    '--train_batch_size': 1,
    '--gradient_accumulation_steps': 4,
    '--max_train_steps': 3000,
    '--learning_rate': 5.0e-04,
    '--scale_lr': True,
    '--lr_scheduler': "constant",
    '--lr_warmup_steps': 0,
    '--report_to': 'wandb',
    '--save_steps': 10,
    '--validation_prompt': "An image of ",
    '--num_validation_images': 1,
    '--validation_epochs': 500

}

params_for_exp = {

}

os.chdir('..')
job_name = '''textual_inversion.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    cls = random.randint(0, food_num_classes)
    k = random.randint(4, 8)
    state = get_datasetstate_with_k_random_indices_with_label('food', label=cls, k=k, split=food_split)
    p['--as_json'] = True
    p['--train_data_dir'] = json.dumps(state.__dict__)
    p['--output_dir'] = get_food_dir() + '/training_output/' + str(random.randint(0, 1_000_000_000 ** 2))
    # '--output_dir': be_random,

    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=5)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
