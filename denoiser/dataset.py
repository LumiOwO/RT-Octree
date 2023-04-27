import os
import json
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class DenoiserDatasetSplit(Dataset):
    def __init__(self, aux_buffer, imgs_in, imgs_gt, device):
        self.aux_buffer = aux_buffer
        self.imgs_in = imgs_in
        self.imgs_gt = imgs_gt
        self.device = device

    def __len__(self):
        return len(self.aux_buffer)

    def __getitem__(self, idx):
        return self.aux_buffer[idx].to(self.device), \
            self.imgs_in[idx].to(self.device), \
            self.imgs_gt[idx].to(self.device)

class DenoiserDataset():
    def __init__(self, args, device=None):
        self.args = args
        self.device = device
        self.aux_buffer = {}
        self.imgs_in = {}
        self.imgs_gt = {}

        aux_buffer, imgs_in, imgs_gt = self.load_images(args)

        tqdm.write("From numpy to torch...")
        for s in aux_buffer.keys():
            self.aux_buffer[s] = torch.stack([
                torch.from_numpy(x[:args.in_channels, ...]) # [C, H, W]
                for x in aux_buffer[s]])
            self.imgs_in[s] = torch.stack([
                torch.from_numpy(x) # [H, W, 4]
                for x in imgs_in[s]])
            self.imgs_gt[s] = torch.stack([
                torch.from_numpy(x) # [H, W, 4]
                for x in imgs_gt[s]])
            
            if args.preload:
                tqdm.write("Moving to cuda...")
                self.aux_buffer[s] = self.aux_buffer[s].to(device)
                self.imgs_in[s] = self.imgs_in[s].to(device)
                self.imgs_gt[s] = self.imgs_gt[s].to(device)

    def load_images(self, args):
        raise NotImplementedError()
    
    def preprocess(self, aux_buffer, img_gt):
        # aux_buffer: [C, H, W]
        # img_gt: [H, W, 3/4]

        # convert png to float
        img_gt = img_gt.astype(np.float32) / 255.0

        # print(aux_buffer.shape)
        img_in = aux_buffer[:4, ...].transpose((1, 2, 0)) # [H, W, 4]

        # white background
        if img_gt.shape[-1] == 4:
            alpha = img_gt[..., -1:]
            img_gt[..., :3] = img_gt[..., :3] * alpha + 1 * (1 - alpha)

        return aux_buffer, img_in, img_gt

    def slice_imgs(self, nx, ny, aux_buffer, img_in, img_gt):
        # aux_buffer [C, H, W]
        # img_in [H, W, 4]
        # img_gt [H, W, 3/4]
        aux_buffer_chunks = []
        img_in_chunks = []
        img_gt_chunks = []

        def valid_chunk(img_gt_chunk):
            tolerance = 0.8
            if img_gt.shape[-1] == 4:
                alpha = img_gt_chunk[..., -1]
                percentage = np.sum(alpha == 0) / alpha.size
            else:
                rgb = img_gt_chunk[..., :3]
                percentage = np.sum(rgb == [1, 1, 1]) / rgb.size
            # print(percentage)
            return percentage < tolerance

        # slice into chunks
        H = aux_buffer.shape[1]
        W = aux_buffer.shape[2]
        dh = H // ny
        dw = W // nx
        for h in range(0, H, dh):
            for w in range(0, W, dw):
                img_gt_chunk = img_gt[h:h+dh, w:w+dw]
                if not valid_chunk(img_gt_chunk):
                    continue
                aux_buffer_chunk = aux_buffer[..., h:h+dh, w:w+dw]
                img_in_chunk = img_in[h:h+dh, w:w+dw]

                img_gt_chunks.append(img_gt_chunk)
                img_in_chunks.append(img_in_chunk)
                aux_buffer_chunks.append(aux_buffer_chunk)

        if False:
            # temp1 = torch.stack([torch.from_numpy(x) for x in [img_in]])
            # print(temp1.shape)
            # torchvision.utils.save_image(temp1.permute(0, 3, 1, 2), 
            #     os.path.join(self.args.work_dir, "temp1.png"), nrow=10)
            temp2 = torch.stack([torch.from_numpy(x) for x in img_gt_chunks])
            print(temp2.shape)
            torchvision.utils.save_image(temp2.permute(0, 3, 1, 2), 
                os.path.join(self.args.work_dir, "temp2.png"), nrow=10)
            temp3 = torch.stack([torch.from_numpy(x[:4, ...]) for x in aux_buffer_chunks])
            print(temp3.shape)
            torchvision.utils.save_image(temp3, 
                os.path.join(self.args.work_dir, "temp3.png"), nrow=10)
            temp4 = torch.stack([torch.from_numpy(x) for x in img_in_chunks])
            print(temp4.shape)
            torchvision.utils.save_image(temp4.permute(0, 3, 1, 2), 
                os.path.join(self.args.work_dir, "temp4.png"), nrow=10)
            exit(1)

        return aux_buffer_chunks, img_in_chunks, img_gt_chunks

    def dataloader(self, task):
        dataset = DenoiserDatasetSplit(
            self.aux_buffer[task], self.imgs_in[task], self.imgs_gt[task], self.device)
        loader = DataLoader(dataset,
            shuffle=(task == "train"),
            batch_size=(self.args.batch_size if task == "train" else 1), 
            num_workers=0)
        return loader


class BlenderDataset(DenoiserDataset):
    def load_images(self, args):
        aux_buffers = {}
        imgs_in = {}
        imgs_gt = {}

        for s in ["train", "val", "test"]:
            if args.task == "test" and s != "test":
                continue
            # FIXME: skip val set?
            if s == "val":
                continue

            with open(os.path.join(args.data_dir, "transforms_{}.json".format(s)), "r") as fp:
                meta = json.load(fp)

            aux_buffers_list = []
            imgs_in_list = []
            imgs_gt_list = []
            tqdm.write(f"Loading {s} set...")
            for frame in tqdm(meta["frames"]):
                name = os.path.basename(frame["file_path"])

                aux_buffer = np.fromfile(
                    os.path.join(args.data_dir, f"spp_{args.spp}", s, "buf_" + name + ".bin"), 
                    dtype=np.float32).reshape((8, 800, 800))
                img_gt = imageio.imread(
                    os.path.join(args.data_dir, s, name + ".png"))

                aux_buffer, img_in, img_gt = self.preprocess(aux_buffer, img_gt)

                if s == "train":
                    aux_buffer, img_in, img_gt = self.slice_imgs(
                            args.nx, args.ny, aux_buffer, img_in, img_gt)
                else:
                    aux_buffer = [aux_buffer]
                    img_in = [img_in]
                    img_gt = [img_gt]

                aux_buffers_list.extend(aux_buffer)
                imgs_in_list.extend(img_in)
                imgs_gt_list.extend(img_gt)
            
            aux_buffers[s] = aux_buffers_list
            imgs_in[s] = imgs_in_list
            imgs_gt[s] = imgs_gt_list
        
        return aux_buffers, imgs_in, imgs_gt

class TanksAndTemplesDataset(DenoiserDataset):
    def load_images(self, args):
        aux_buffers = {}
        imgs_in = {}
        imgs_gt = {}
        
        img_files = sorted(os.listdir(os.path.join(args.data_dir, "rgb")))

        for s in ["train", "val", "test"]:
            if args.task == "test" and s != "test":
                continue
            # FIXME: skip val set?
            if s == "val":
                continue

            if s == 'train':
                img_files_s = [x for x in img_files if x.startswith('0_')]
            elif s == 'test':
                img_files_s = [x for x in img_files if x.startswith('1_')]

            aux_buffers_list = []
            imgs_in_list = []
            imgs_gt_list = []
            tqdm.write(f"Loading {s} set...")
            for img_fname in tqdm(img_files_s):
                name = img_fname.split(".")[0]

                aux_buffer = np.fromfile(
                    os.path.join(args.data_dir, f"spp_{args.spp}", "buf_" + name + ".bin"), 
                    dtype=np.float32).reshape((8, 1920, 1080))
                img_gt = imageio.imread(
                    os.path.join(args.data_dir, "rgb", name + ".png"))

                aux_buffer, img_in, img_gt = self.preprocess(aux_buffer, img_gt)

                if s == "train":
                    aux_buffer, img_in, img_gt = self.slice_imgs(
                        args.nx, args.ny, aux_buffer, img_in, img_gt)
                else:
                    aux_buffer = [aux_buffer]
                    img_in = [img_in]
                    img_gt = [img_gt]

                aux_buffers_list.extend(aux_buffer)
                imgs_in_list.extend(img_in)
                imgs_gt_list.extend(img_gt)

            aux_buffers[s] = aux_buffers_list
            imgs_in[s] = imgs_in_list
            imgs_gt[s] = imgs_gt_list
           
        return aux_buffers, imgs_in, imgs_gt