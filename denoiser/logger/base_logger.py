from tqdm import tqdm
import json

import os
import torchvision

class BaseLogger():
    def __init__(self):
        pass

    def print(self, s, **kwargs):
        tqdm.write(f"===== {s}", **kwargs)

    def log(self, logs_dict):
        self.print(json.dumps(logs_dict))

    def log_image(self, image, path, name):
        # image: [H, W, C]
        torchvision.utils.save_image(image.permute(2, 0, 1), os.path.join(path, name))
