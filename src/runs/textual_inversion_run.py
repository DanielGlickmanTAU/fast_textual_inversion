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
overall_runs = 4
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
    '--validation_prompt': '"An image of "',
    '--num_validation_images': 1,
    '--validation_epochs': validation_epochs,
    '--dataset': dataset

}

params_for_exp = {

}

os.chdir('..')
job_name = '''textual_inversion.py'''
ids = []
for _ in range(overall_runs):

    for p in gridsearch(params, params_for_exp):
        if s3_upload:
            p['--s3_upload'] = ''
        time_signature = str(time.time())
        if dataset == 'food' or dataset == 'cars':
            cls = random.randint(0, food_num_classes)

            k = random.randint(min_num_class, max_num_class)
            p['--as_json'] = ''
            p['--num_images'] = k
            lbl = str(cls)
            p['--label_used'] = lbl
            state = get_datasetstate_with_k_random_indices_with_label('food', label=cls, k=k, split=food_split,
                                                                      quick=quick)
            train_output_dir = get_food_dir() + '/training_output/' + dataset + '_' + lbl + '_' + time_signature
            p['--output_dir'] = train_output_dir

            dataset_state_json_ = p['--output_dir'] + '/dataset_state.json'
            os.makedirs(p['--output_dir'])
            json.dump(state.__dict__, open(dataset_state_json_, 'w'))
            p['--train_data_dir'] = dataset_state_json_
        else:
            # TODO: change this! should iterate all options rather than randomly
            train_dir = get_celeb_dir() + '/' + str(random.choice([int(x) for x in os.listdir(get_celeb_dir())]))
            p['--train_data_dir'] = train_dir
            train_output_dir = train_dir + '/training_output/' + dataset + '_' + time_signature
            p['--output_dir'] = train_output_dir
            os.makedirs(p['--output_dir'])

        if dataset == 'food':
            p['--initializer_token'] = 'food'
        if dataset == 'cars':
            p['--initializer_token'] = 'car'
        if dataset == 'celeba':
            p['--initializer_token'] = 'person'

        id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=1, wandb=False)
        ids.append(id)
    print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
