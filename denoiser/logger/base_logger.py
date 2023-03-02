from tqdm import tqdm

class BaseLogger():
    def __init__(self):
        pass

    def print(self, s, **kwargs):
        tqdm.write(s, **kwargs)

    def log(self, **kwargs):
        pass