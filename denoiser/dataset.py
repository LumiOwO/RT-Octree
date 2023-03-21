import os
import json
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torchvision
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

                # preprocess images
                if s == "train":
                    img_in_chunks, img_out_chunks = self.preprocess_train(img_in, img_out)
                    imgs_in.extend(img_in_chunks)
                    imgs_out.extend(img_out_chunks)
                else:
                    img_in, img_out = self.preprocess_test(img_in, img_out)
                    imgs_in.append(img_in)
                    imgs_out.append(img_out)

            if False:
                # print(img_in_chunks.shape)
                temp1 = torch.stack([torch.from_numpy(x) for x in imgs_in])
                print(temp1.shape)
                torchvision.utils.save_image(temp1.permute(0, 3, 1, 2), os.path.join(self.args.work_dir, "temp1.png"), nrow=10)
                temp2 = torch.stack([torch.from_numpy(x) for x in imgs_out])
                print(temp2.shape)
                torchvision.utils.save_image(temp2.permute(0, 3, 1, 2), os.path.join(self.args.work_dir, "temp2.png"), nrow=10)
                exit(1)

            self.imgs_in[s] = torch.stack([torch.from_numpy(x) for x in imgs_in]).to(device)
            self.imgs_out[s] = torch.stack([torch.from_numpy(x) for x in imgs_out]).to(device)

    def preprocess_train(self, img_in, img_out):
        img_in /= 255
        img_out /= 255
        img_in = img_in[..., :3]
        if img_out.shape[-1] == 4:
            alpha = img_out[..., -1:]
            img_out = img_out[..., :3] * alpha + 1 * (1 - alpha)

        # slice into chunks
        n = 80
        tolerance = 0.8

        def valid_chunk(img_chunk):
            percentage = np.sum(img_chunk == [1, 1, 1]) / img_chunk.size
            return percentage < tolerance
        img_in_chunks = []
        img_out_chunks = []
        for i in range(0, img_in.shape[0], n):
            for j in range(0, img_in.shape[1], n):
                img_out_chunk = img_out[i:i+n, j:j+n]
                if not valid_chunk(img_out_chunk):
                    continue
                img_in_chunk = img_in[i:i+n, j:j+n]

                img_in_chunks.append(img_in_chunk)
                img_out_chunks.append(img_out_chunk)

        return img_in_chunks, img_out_chunks

    def preprocess_test(self, img_in, img_out):
        img_in /= 255
        img_out /= 255
        img_in = img_in[..., :3]
        if img_out.shape[-1] == 4:
            alpha = img_out[..., -1:]
            img_out = img_out[..., :3] * alpha + 1 * (1 - alpha)
        return img_in, img_out

    def dataloader(self, task):
        dataset = DenoiserDatasetSplit(self.imgs_in[task], self.imgs_out[task])
        loader = DataLoader(dataset,
            shuffle=(task == "train"),
            batch_size=1, 
            num_workers=0)
        return loader