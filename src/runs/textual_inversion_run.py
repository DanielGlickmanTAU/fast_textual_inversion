import random
import time

import os
import json
from src.data.concepts_datasets import get_datasetstate_with_k_random_indices_with_label, get_project_dir, get_food_dir
from src.misc.gridsearch import gridsearch
from src.misc.slurm import run_on_slurm

food_num_classes = 101
food_split = 'test'

# quick = True
# validation_epochs = 10
quick = False
validation_epochs = 100
overall_runs = 1
params = {
    '--enable_xformers_memory_efficient_attention': '',
    '--pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
    '--placeholder_token': 'my_new_token',
    '--initializer_token': 'food',
    '--resolution': 512,
    # '--train_batch_size': 2???,
    '--train_batch_size': 4,
    '--distance_loss_alpha': 0.1,
    '--gradient_accumulation_steps': 4,
    '--max_train_steps': 3000,
    '--learning_rate': 5.0e-04,
    '--scale_lr': '',
    '--lr_scheduler': "constant",
    '--lr_warmup_steps': 0,
    '--report_to': 'wandb',
    '--save_steps': 10,
    '--validation_prompt': '"An image of "',
    '--num_validation_images': 1,
    '--validation_epochs': validation_epochs,

}

params_for_exp = {

}

os.chdir('..')
job_name = '''textual_inversion.py'''
ids = []
for _ in range(overall_runs):
    for p in gridsearch(params, params_for_exp):
        cls = random.randint(0, food_num_classes)
        k = random.randint(4, 8)
        p['--as_json'] = ''
        p['--num_images'] = k
        lbl = str(cls)
        p['--label_used'] = lbl
        state = get_datasetstate_with_k_random_indices_with_label('food', label=cls, k=k, split=food_split, quick=quick)
        train_output_dir = get_food_dir() + '/training_output/' + str(time.time()) + '_' + lbl
        p['--output_dir'] = train_output_dir
        os.makedirs(p['--output_dir'])
        dataset_state_json_ = p['--output_dir'] + '/dataset_state.json'
        json.dump(state.__dict__, open(dataset_state_json_, 'w'))
        p['--train_data_dir'] = dataset_state_json_

        id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=1, wandb=False)
        ids.append(id)
    print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
