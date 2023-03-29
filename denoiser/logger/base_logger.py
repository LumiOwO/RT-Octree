from tqdm import tqdm
import json

import os
from pathlib import Path
import torchvision

class BaseLogger():
    def __init__(self, args):
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.work_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)

    def print(self, s, **kwargs):
        tqdm.write(f"===== {s}", **kwargs)

    def log(self, logs_dict):
        self.print(json.dumps(logs_dict))

    def log_image(self, image, path, name, idx, logs_dict):
        # image: [1, H, W, C]
        torchvision.utils.save_image(
            image.permute(0, 3, 1, 2), os.path.join(path, f"{name}_{idx}.png"))
