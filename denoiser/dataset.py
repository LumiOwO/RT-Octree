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
            # FIXME: skip val set?
            if s == "val":
                continue

            with open(os.path.join(args.data_dir, "transforms_{}.json".format(s)), "r") as fp:
                meta = json.load(fp)

            imgs_in = []
            imgs_out = []
            tqdm.write(f"Loading {s} set...")
            for frame in tqdm(meta["frames"]):
                name = os.path.basename(frame["file_path"])
                img_in = imageio.imread(
                    os.path.join(args.data_dir, "spp_1", s, name + ".png")).astype(np.float32)
                img_out = imageio.imread(
                    os.path.join(args.data_dir, s, name + ".png")).astype(np.float32)

                # preprocess channels
                img_in = img_in[..., :3]
                if img_out.shape[-1] == 4:
                    alpha = img_out[..., :-1]
                    img_out = img_out[..., :3] * alpha + 1 * (1 - alpha)

                imgs_in.append(img_in / 255)
                imgs_out.append(img_out / 255)

            self.imgs_in[s] = torch.stack([torch.from_numpy(x) for x in imgs_in]).to(device)
            self.imgs_out[s] = torch.stack([torch.from_numpy(x) for x in imgs_out]).to(device)

    def dataloader(self, task):
        dataset = DenoiserDatasetSplit(self.imgs_in[task], self.imgs_out[task])
        loader = DataLoader(dataset,
            shuffle=(task == "train"),
            batch_size=1, 
            num_workers=0)
        return loader