import torch
import os

from tqdm import trange
from denoiser.utils import load_checkpoint
from denoiser.network import compact_and_compile

class Runner(object):
    def __init__(self, args, dataset, logger, device=None):
        self.args = args
        self.dataset = dataset
        self.logger = logger
        self.device = device

        # model
        self.optimizer_fn = lambda model: torch.optim.Adam(model.parameters(), 
                lr=self.args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: 0.1 ** min(epoch / self.args.epochs, 1))
        if args.loss_fn == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError("Invalid task type.")


    def train(self, model):
        # init
        dataloader = self.dataset.dataloader("train")
        optimizer = self.optimizer_fn(model)
        lr_scheduler = self.scheduler_fn(optimizer)
        start = 1

        # load checkpoint
        # ckpt = load_checkpoint(path, "train")
        # if ckpt is not None:
        #     model.load_state_dict(ckpt['model'])
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        #     start = ckpt["epoch"]

        # train
        for epoch in trange(start, self.args.epochs + 1):
            self.train_one_epoch(model, dataloader, optimizer, lr_scheduler, epoch)

        # test after training
        self.test(model, False)

    def train_one_epoch(self, model, dataloader, optimizer, lr_scheduler, epoch):
        for batch_idx, (imgs_in, imgs_gt) in enumerate(dataloader):
            # batch_size == 1
            optimizer.zero_grad()
            imgs_out = model.forward(imgs_in, requires_grad=True)
            loss = self.loss_fn(imgs_out, imgs_gt.squeeze(0))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # log
        if epoch % self.args.i_weights == 0:
            path = os.path.join(self.args.work_dir, f"{epoch:06d}.tar")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }, path)
            self.logger.print(f"Saved checkpoints at {path}")

    def test(self, model, load_ckpt=True):
        # init
        dataloader = self.dataset.dataloader("test")
        if load_ckpt:
            ckpt = load_checkpoint(path, "train")
            if ckpt is None:
                logger.print("No checkpoint found")
                return
            model.load_state_dict(ckpt['model'])

        with torch.no_grad():
            self.test_one_epoch(model, dataloader)

    def test_one_epoch(self, model, dataloader):
        # batch_size == 1
        for batch_idx, (imgs_in, imgs_gt) in enumerate(dataloader):
            imgs_out = model.forward(imgs_in)
            loss = self.loss_fn(imgs_out, imgs_gt)

    def compact(self, model):
        trt_ts_module = compact_and_compile(model)
        torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript
