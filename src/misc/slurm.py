import os
import sys

import time
import random

print(os.path)
python = os.sys.executable

slurm_file = 'my_slurm.slurm'

num_jobs_that_can_run_on_studentbatch_at_one_time = 10


# num_jobs_that_can_run_on_studentbatch_at_one_time = 40


# num_jobs_that_can_run_on_studentbatch_at_one_time = 0


def get_partition_and_time_limit(partition=None):
    if partition is not None:
        if 'batch' in partition:
            return 'studentbatch', 'infinite'
        if 'kill' in partition:
            return 'studentkillable', 'infinite'
        raise RuntimeError('partition needs to be either kill or batch')

    num_jobs_in_student_batch = os.popen('squeue | grep glick | grep studentba | wc -l').read()
    num_jobs_in_student_batch = int(num_jobs_in_student_batch) if num_jobs_in_student_batch else 0
    # if 'studentb' in os.popen('squeue | grep glickman').read():

    if num_jobs_in_student_batch >= num_jobs_that_can_run_on_studentbatch_at_one_time:
        # return 'studentkillable', 'infinite'
        return 'studentkillable', '23:35:00'

    # return 'studentbatch', 'infinite'
    return 'studentbatch', '2-23:35:00'


def run_on_slurm(job_name, params, no_flag_param='', slurm=None, gpu=True, sleep=True, wandb=True, slurm_partition=None,
                 slurm_time_limit=None):
    if slurm is None:
        slurm = len(os.popen('which squeue').read()) > 1
    if 'sulim' in os.popen('whoami').read():
        job_name = job_name.replace('main', 'ssr')
    partition, time_limit = get_partition_and_time_limit()
    if slurm_partition:
        partition = slurm_partition
    if slurm_time_limit:
        time_limit = slurm_time_limit
    python_file = job_name
    python_file = python_file.replace('.py', '')
    job_name = job_name + str(time.time())
    # need to for gps main stuff
    if isinstance(no_flag_param, dict):
        if wandb and 'wandb_project' not in no_flag_param:
            no_flag_param['--wandb_project'] = os.path.basename(sys.argv[0]).replace('.py', '') \
                .replace('_slurm', '').replace('slurm_', '').replace('slurm', '')
        no_flag_param = ' '.join([f'{key} {value}' for key, value in no_flag_param.items()])
    command = f'{python} {python_file}.py ' + ' '.join(
        [f'--{key} {value}' for key, value in params.items()]) + ' ' + no_flag_param
    print(f'running {command}')
    if slurm:
        slurm_script = f'''#! /bin/sh
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={'1' if gpu else '0'}
{command}'''
        with open(slurm_file, 'w') as f:
            f.write(slurm_script)

        job_id = os.popen(f'sbatch {slurm_file}').read()[-7:].strip()
        print(f'executing {job_name} with job id {job_id}')
        open(f'./slurm_id_{job_id}_outfile_{job_name}', 'w').write(slurm_script)

        if sleep:
            if isinstance(sleep, bool):
                time.sleep(random.randint(0, 15))
            else:
                time.sleep(random.randint(0, sleep))
        else:
            time.sleep(1)
        return job_id

    else:
        time_time = time.time()
        res_file = f'res_{time_time}.txt'
        print(f'saving to {res_file}')
        os.system(f'echo {command} > command_{time_time}.txt')
        os.system(f"nohup sh -c ' {command} > {res_file} 2>&1 &'&")

    # os.system('chmod 700 slurm.py')
