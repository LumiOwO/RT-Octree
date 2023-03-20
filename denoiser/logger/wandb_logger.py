import wandb
from .base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, args):
        super().__init__()
        # wandb.init()

    def log(self, **kwargs):
        super().log()
        # self.print(json.dumps(kwargs))
