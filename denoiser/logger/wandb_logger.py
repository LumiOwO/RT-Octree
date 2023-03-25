import os
import wandb
from .base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, args):
        super().__init__()
        wandb.init(project="my-awesome-project")

    def log(self, logs_dict):
        super().log(logs_dict)
        wandb.log(logs_dict)

    def log_image(self, image, path, name, idx):
        # image: [1, H, W, C]
        super().log_image(image, path, name, idx)
        test_name = os.path.basename(path)
        np_image = (image.clamp(0, 1) * 255).byte().squeeze(0).cpu().numpy()
        wandb.log({
            f"{test_name}": wandb.Image(np_image),
            "idx" : idx
        })
