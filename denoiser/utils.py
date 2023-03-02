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

def load_checkpoint(task):
    if task == "train":
        torch.load(ckpt_path)
    elif task == "test":
        pass
    elif task == "compact":
        pass