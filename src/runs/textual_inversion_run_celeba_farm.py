import random
import time
import sys
import os
import json

sys.path.append(os.path.abspath('../..'))
# print(os.popen('pwd').read())

from src import run_utils
from src.data.concepts_datasets import get_datasetstate_with_k_random_indices_with_label, get_project_dir, get_food_dir, \
    get_celeb_dir, get_cars_dir, get_celeb_short_dir
from src.misc.gridsearch import gridsearch
from src.misc.slurm import run_on_slurm

# dataset, num_classes, split = 'food', 101, 'test'
# dataset, num_classes, split = 'cars', 196, 'train'
dataset = 'celeba'
validation_epochs = 50000000000000
# validation_epochs = 50
s3_upload = True
# s3_upload = False
start_runner = True
# start_runner = False
max_train_steps = 5000
save_steps = 10
batch_size = 1
params = {
    '--enable_xformers_memory_efficient_attention': '',
    '--pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
    '--placeholder_token': 'my_new_token',
    '--resolution': 512,
    '--learnable_property': 'person',

    '--train_batch_size': batch_size,
    '--distance_loss_alpha': 0.1,
    '--gradient_accumulation_steps': max(1, int(4 / batch_size)),
    '--max_train_steps': max_train_steps,
    '--learning_rate': 5.0e-04,
    # '--learning_rate': 8.0e-04,
    '--scale_lr': '',
    '--lr_scheduler': "constant",
    '--lr_warmup_steps': 0,
    '--report_to': 'wandb',
    '--save_steps': save_steps,
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


def get_partition():
    """
    1) if no glick job is queues for sbatch, queue to sbatch
    2) if no job is queued for studentkill, queue to studentkill
    :return:
    """
    if run_utils.jobs_in_queue_studentbatch() == 0:
        return 'studentbatch'
    if run_utils.jobs_in_queue_studentkill() == 0:
        return 'studentkillable'
    return None


def get_partition_non_none_blocking():
    global partition
    partition = None
    while partition is None:
        partition = get_partition()
        if partition is None:
            time.sleep(60)
    return partition


celeb_ids = [int(x) for x in os.listdir(get_celeb_dir())]
random.shuffle(celeb_ids)

for celeb_id in celeb_ids:
    if run_utils.is_id_done(celeb_id):
        continue
    train_dir = get_celeb_dir() + '/' + str(celeb_id)
    num_images = len([x for x in os.listdir(train_dir) if 'jpg' in x])
    if num_images == 0:
        continue
    p = gridsearch(params, params_for_exp)[0]
    if s3_upload:
        p['--s3_upload'] = ''
    if start_runner:
        p['--start_runner'] = ''
    partition = get_partition_non_none_blocking()

    time_signature = str(time.time())
    p['--mark_done'] = str(celeb_id)
    p['--train_data_dir'] = train_dir
    train_output_dir = train_dir + '/training_output/' + dataset + '_' + time_signature
    p['--output_dir'] = train_output_dir
    os.makedirs(p['--output_dir'])

    p['--initializer_token'] = 'person'

    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=60, wandb=True, slurm_time_limit='2:00:00',
                      slurm_partition=partition)
    ids.append(id)
    print(f'submited {len(gridsearch(params, params_for_exp))} jobs')

celeb_ids = [int(x) for x in os.listdir(get_celeb_short_dir())]
random.shuffle(celeb_ids)

for celeb_id in celeb_ids:
    if run_utils.is_id_done(celeb_id):
        continue

    p = gridsearch(params, params_for_exp)[0]
    if s3_upload:
        p['--s3_upload'] = ''
    if start_runner:
        p['--start_runner'] = ''
    partition = get_partition_non_none_blocking()

    time_signature = str(time.time())
    p['--mark_done'] = str(celeb_id)
    train_dir = get_celeb_dir() + '/' + str(celeb_id)
    p['--train_data_dir'] = train_dir
    train_output_dir = train_dir + '/training_output/' + dataset + '_' + time_signature
    p['--output_dir'] = train_output_dir
    os.makedirs(p['--output_dir'])

    p['--initializer_token'] = 'person'

    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=60, wandb=True, slurm_time_limit='2:00:00',
                      slurm_partition=partition)
    ids.append(id)
    print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
