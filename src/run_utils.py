import os
import datetime

from src.data.concepts_datasets import get_project_dir

celeba_py = "textual_inversion_run_celeba_farm.py"

runs_s_celeba_py = "/specific/netapp5_wolf/wolf/turing/glickman/anaconda3/envs/fast_textual_inversion/bin/python /specific/netapp5_wolf/wolf/turing/glickman/fast_textual_inversion/src/runs/%s" % celeba_py


def run_celeba():
    os.system(f"nohup sh -c ' {runs_s_celeba_py} > ./nohuop.out 2>&1 &'&")


def num_running_process():
    l = os.popen(f'ps -aux | grep {celeba_py}').readlines()
    return sum([runs_s_celeba_py in command for command in l])


def run_if_not_running():
    if num_running_process() == 0:
        run_celeba()


def hour_in_day():
    return datetime.datetime.now().hour


def jobs_in_queue_studentbatch():
    studentba = 'studentba'
    num_jobs_in_student_batch = os.popen('squeue | grep glick | grep %s | grep 0:00 | wc -l' % studentba).read()
    return int(num_jobs_in_student_batch) if num_jobs_in_student_batch else 0


def jobs_in_queue_studentkill():
    studentba = 'studentki'
    num_jobs_in_student_batch = os.popen('squeue| grep glick | grep %s | grep 0:00 | wc -l' % studentba).read()
    return int(num_jobs_in_student_batch) if num_jobs_in_student_batch else 0


def mark_id_done(celeb_id):
    os.system(f'touch {get_project_dir()}/{celeb_id}_done')


def is_id_done(celeb_id):
    return os.path.exists(f'{get_project_dir()}/{celeb_id}_done')
