import os
import json
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class DenoiserDatasetSplit(Dataset):
    def __init__(self, buffers_in, imgs_gt):
        self.buffers_in = buffers_in
        self.imgs_gt = imgs_gt

    def __len__(self):
        return len(self.buffers_in)

    def __getitem__(self, idx):
        return self.buffers_in[idx], self.imgs_gt[idx]

class DenoiserDataset():
    def __init__(self, args, device=None):
        self.args = args
        self.device = device
        self.buffers_in = {}
        self.imgs_gt = {}

        for s in ["train", "val", "test"]:
            if args.task == "test" and s != "test":
                continue
            # FIXME: skip val set?
            if s == "val":
                continue

            with open(os.path.join(args.data_dir, "transforms_{}.json".format(s)), "r") as fp:
                meta = json.load(fp)

            buffers_in_list = []
            imgs_gt_list = []
            tqdm.write(f"Loading {s} set...")
            for frame in tqdm(meta["frames"]):
                name = os.path.basename(frame["file_path"])
                rgba = imageio.imread(
                    os.path.join(args.data_dir, "spp_1", s, name + ".png"))
                img_gt = imageio.imread(
                    os.path.join(args.data_dir, s, name + ".png"))

                buffers_in, img_gt = self.preprocess(rgba, img_gt)

                # if s == "train":
                if False:
                    buffers_in, img_gt = self.slice_imgs(buffers_in, img_gt)
                else:
                    buffers_in = [buffers_in]
                    img_gt = [img_gt]

                buffers_in_list.extend(buffers_in)
                imgs_gt_list.extend(img_gt)

            self.buffers_in[s] = torch.stack([torch.from_numpy(x) for x in buffers_in_list]).to(device)
            self.imgs_gt[s] = torch.stack([torch.from_numpy(x) for x in imgs_gt_list]).to(device)

    def slice_imgs(self, buffers_in, img_gt):
        # slice into chunks
        n = 80
        tolerance = 0.8

        def valid_chunk(img_gt_chunk):
            if img_gt.shape[-1] == 4:
                alpha = img_gt_chunk[..., -1]
                percentage = np.sum(alpha == 0) / alpha.size
            else:
                rgb = img_gt_chunk[..., :3]
                percentage = np.sum(rgb == [1, 1, 1]) / rgb.size
            # print(percentage)
            return percentage < tolerance
        buffers_in_chunks = []
        img_gt_chunks = []
        for i in range(0, buffers_in.shape[0], n):
            for j in range(0, buffers_in.shape[1], n):
                img_gt_chunk = img_gt[i:i+n, j:j+n]
                if not valid_chunk(img_gt_chunk):
                    continue
                buffers_in_chunk = buffers_in[i:i+n, j:j+n]

                img_gt_chunks.append(img_gt_chunk)
                buffers_in_chunks.append(buffers_in_chunk)

        if False:
            # print(img_in_chunks.shape)
            temp1 = torch.stack([torch.from_numpy(x[..., :3]) for x in buffers_in_chunks])
            print(temp1.shape)
            torchvision.utils.save_image(temp1.permute(0, 3, 1, 2), os.path.join(self.args.work_dir, "temp1.png"), nrow=10)
            temp2 = torch.stack([torch.from_numpy(x) for x in img_gt_chunks])
            print(temp2.shape)
            torchvision.utils.save_image(temp2.permute(0, 3, 1, 2), os.path.join(self.args.work_dir, "temp2.png"), nrow=10)
            exit(1)

        return buffers_in_chunks, img_gt_chunks

    def preprocess(self, rgba, img_gt):
        # convert to float
        rgba = rgba.astype(np.float32) / 255.0
        img_gt = img_gt.astype(np.float32) / 255.0

        # print(rgba.shape)
        if rgba.shape[-1] == 3:
            print("!!FIXME: rbga is always 4 channels!!")
            exit(1)
        # print(rgba.shape)

        # white background
        if img_gt.shape[-1] == 4:
            alpha = img_gt[..., -1:]
            img_gt[..., :3] = img_gt[..., :3] * alpha + 1 * (1 - alpha)

        return rgba, img_gt

    def dataloader(self, task):
        dataset = DenoiserDatasetSplit(self.buffers_in[task], self.imgs_gt[task])
        loader = DataLoader(dataset,
            shuffle=(task == "train"),
            batch_size=(self.args.batch_size if task == "train" else 1), 
            num_workers=0)
        return loader