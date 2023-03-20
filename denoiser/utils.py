import torch
import numpy as np
import random
import os

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_checkpoint(path):
    file_names = os.listdir(path)
    max_num = -1
    latest_path = None
    for file_name in file_names:
        if file_name.startswith('checkpoint_'):
            num = int(file_name.split('_')[1].split('.')[0])
            if num > max_num:
                max_num = num
                latest_path = file_name

    latest = None
    if latest_path is not None:
        latest_path = os.path.join(path, latest_path)
        latest = torch.load(latest_path)
    return latest, latest_path
