import gc
from pathlib import Path

try:
    import comet_ml
except:
    print('failed importing comet ml')

import os
import time

minimum_free_giga = 4
max_num_gpus = 1

last_write = 0


def is_university_server():
    try:
        whoami = os.popen('whoami').read()
        return 'glickman1' in whoami or 'chaimc' in whoami or 'gamir' in os.environ['HOST'] or 'rack' in os.environ[
            'HOST']
    except:
        return False


def get_cache_dir():
    if is_university_server():
        return '/specific/netapp5_wolf/wolf/turing/glickman/cache/cache'
    return None


def get_index_of_free_gpus(minimum_free_giga=minimum_free_giga):
    def get_free_gpu():
        try:
            lines = os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()
        except Exception as e:
            print('error getting free memory', e)
            return {0: 10000, 1: 10000, 2: 0, 3: 10000, 4: 0, 5: 0, 6: 0, 7: 0}

        memory_available = [int(x.split()[2]) for x in lines]
        gpus = {index: mb for index, mb in enumerate(memory_available)}
        return gpus

    gpus = get_free_gpu()
    if len(gpus) == 0 and not is_university_server():
        return {0: 9999}

    gpus = {index: mega for index, mega in gpus.items() if mega >= minimum_free_giga * 1000}
    gpus = {k: v for k, v in sorted(gpus.items(), key=lambda x: x[1], reverse=True)}

    return gpus


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if is_university_server():
    os.environ['TRANSFORMERS_CACHE'] = get_cache_dir()

gpus = get_index_of_free_gpus()
gpus = list(map(str, gpus))[:max_num_gpus]
join = ','.join(gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = join
print('setting CUDA_VISIBLE_DEVICES=' + join)
if max_num_gpus == 1:
    print('working with 1 gpu:(')

import torch

old_rpr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f'{self.shape} {old_rpr(self)}'


def get_torch():
    return torch


def print_size_of_model(model):
    get_torch().save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def get_device():
    torch = get_torch()

    gpus = get_index_of_free_gpus()
    # print(gpus)
    return torch.device(compute_gpu_indent(gpus) if torch.cuda.is_available() else 'cpu')


def compute_gpu_indent(gpus):
    try:
        # return 'cuda'
        best_gpu = max(gpus, key=lambda gpu_num: gpus[int(gpu_num)])
        indented_gpu_index = list(gpus.keys()).index(best_gpu)
        return 'cuda:' + str(indented_gpu_index)
    except:
        return 'cuda'


def get_device_and_set_as_global():
    d = get_device()
    get_torch().cuda.set_device(d)
    return d


def clean_memory():
    t = time.time()
    gc.collect()
    get_torch().cuda.empty_cache()
    print(f'gc took:{time.time() - t}')


def get_project_root():
    project_name = 'code2seq_torch'
    current_dir = Path(__file__)
    return str([p for p in current_dir.parents if p.parts[-1] == project_name][0])
