from tqdm import tqdm
import json

class BaseLogger():
    def __init__(self):
        pass

    def print(self, s, **kwargs):
        tqdm.write(f"===== {s}", **kwargs)

    def log(self, **kwargs):
        self.print(json.dumps(kwargs))
