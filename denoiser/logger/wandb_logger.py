import os
import wandb
from .base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, args):
        wandb.init(project=args.exp_name)
        args.wandb_name = wandb.run.name
        args.work_dir = os.path.join(args.work_dir, args.wandb_name)

        super().__init__(args)
        wandb.log(vars(args))

    def log(self, logs_dict):
        super().log(logs_dict)
        wandb.log(logs_dict)

    def log_image(self, image, path, name, idx, logs_dict, upload=False):
        # image: [1, H, W, C]
        super().log_image(image, path, name, idx, logs_dict)

        if upload:
            test_name = os.path.basename(path)
            np_image = (image.clamp(0, 1) * 255).byte().squeeze(0).cpu().numpy()
            wandb.log({
                f"image/{name}": wandb.Image(np_image, caption=test_name),
                **logs_dict
            })
