import os
import json
import imageio
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class DenoiserDatasetSplit(Dataset):
    def __init__(self, imgs_in, imgs_out):
        self.imgs_in = imgs_in
        self.imgs_out = imgs_out

    def __len__(self):
        return len(self.imgs_in)

    def __getitem__(self, idx):
        return self.imgs_in[idx], self.imgs_out[idx]

class DenoiserDataset():
    def __init__(self, args, device=None):
        self.args = args
        self.device = device

        # load images
        self.imgs_in = {}
        self.imgs_out = {}

        for s in ["train", "val", "test"]:
            if args.task == "test" and s != "test":
                continue

            with open(os.path.join(args.data_dir, "transforms_{}.json".format(s)), "r") as fp:
                meta = json.load(fp)

            imgs_in = []
            imgs_out = []
            for frame in tqdm(meta["frames"]):
                name = os.path.join(frame["file_path"])
                img_in = imageio.imread(
                    os.path.join(args.data_dir, "spp_1", s, name + ".png")).astype(np.float)
                imgs_in.append(img_in)
                img_out = imageio.imread(
                    os.path.join(args.data_dir, s, name + ".png")).astype(np.float)
                imgs_out.append(img_out)

            self.imgs_in[s] = imgs_in
            self.imgs_out[s] = imgs_out

        self.imgs_in = torch.from_numpy(np.stack(self.imgs_in, axis=0))
        self.imgs_out = torch.from_numpy(np.stack(self.imgs_out, axis=0))

    def dataloader(self, task):
        dataset = DenoiserDatasetSplit(self.imgs_in[task], self.imgs_out[task])
        loader = DataLoader(dataset,
            shuffle=(task == "train"),
            batch_size=self.args.batch_size if task == "train" else 1, 
            num_workers=0)
        return loader