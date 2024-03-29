import torch
import os
import gc

from tqdm import tqdm, trange
from denoiser.utils import load_checkpoint
from denoiser.network import compact_and_compile, filtering
from denoiser.metrics import get_loss_fn
from denoiser.metrics import PSNRMetric, SSIMMetric, LPIPSMetric

class Runner(object):
    def __init__(self, args, dataset=None, logger=None, device=None):
        self.args = args
        self.dataset = dataset
        self.logger = logger
        self.device = device

        # model
        self.optimizer_fn = lambda model: torch.optim.Adam(model.parameters(), 
                lr=self.args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: 0.1 ** min(epoch / (self.args.epochs + 1), 1))
        self.loss_fn = get_loss_fn(args.loss_fn, device)

        # Metrics
        if args.task == "train" or args.task == "test":
            self.metrics = [
                PSNRMetric(), 
                SSIMMetric(), 
                LPIPSMetric(device=device), 
            ]

        self.epoch = 0


    def train(self, model):
        # init
        dataloader = self.dataset.dataloader("train")
        optimizer = self.optimizer_fn(model)
        lr_scheduler = self.scheduler_fn(optimizer)
        scaler = torch.cuda.amp.GradScaler()
        start = 1

        # load checkpoint
        ckpt, ckpt_path = load_checkpoint(self.args.work_dir)
        if ckpt is not None:
            self.logger.print(f"Load checkpoint from {ckpt_path}")
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            start = ckpt["epoch"]
        else:
            self.logger.print("No checkpoint found")

        # train
        for epoch in trange(start, self.args.epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler)
            gc.collect()
            torch.cuda.empty_cache()

            # run test set
            if epoch > start and epoch < self.args.epochs and epoch % self.args.i_test == 0:
                self.logger.print(f"Testing at epoch {epoch}...")
                self.test(model, load_ckpt=False, save_dirname=f"test_{epoch:06d}")

        # test after training
        self.logger.print("Test after training")
        self.test(model, load_ckpt=False)

    def train_one_epoch(self, model, dataloader, optimizer, lr_scheduler, scaler):
        avg_loss = 0
        for batch_idx, (aux_buffer, img_in, img_gt) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            img_out = model.filtering(aux_buffer, img_in, requires_grad=True)

            loss = self.loss_fn(img_out[..., :3], img_gt[..., :3])
            # loss.backward()
            scaler.scale(loss).backward()

            optimizer.step()
            avg_loss += loss.item()

        # update learning rate per epoch
        lr_scheduler.step()

        # log train loss
        if self.epoch % self.args.i_print == 0:
            self.logger.log({
                "epoch": self.epoch,
                "train/loss": avg_loss / len(dataloader),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        # save
        if self.epoch % self.args.i_save == 0:
            # save torchscript model
            self.compact(model, load_ckpt=False, filename=f"ts_{self.epoch:06d}.ts")
            
            # save checkpoint
            path = os.path.join(self.args.work_dir, f"checkpoint_{self.epoch:06d}.tar")
            torch.save({
                "epoch": self.epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }, path)
            self.logger.print(f"Save checkpoint at {path}")


    def test(self, model, load_ckpt=True, save_dirname="test"):
        # init
        dataloader = self.dataset.dataloader("test")
        if load_ckpt:
            ckpt, ckpt_path = load_checkpoint(self.args.work_dir)
            if ckpt is None:
                self.logger.print("No checkpoint found.")
                return
            self.logger.print(f"Load checkpoint from {ckpt_path}")
            model.load_state_dict(ckpt['model'])

        with torch.no_grad():
            self.test_one_epoch(model, dataloader, save_dirname)
        
    def test_one_epoch(self, model, dataloader, save_dirname):
        # batch_size == 1 when testing

        if self.args.save_image:
            save_dir = os.path.join(self.args.work_dir, save_dirname)
            os.makedirs(save_dir, exist_ok=True)

        # Compact model & test
        ts_module = self.compact(model, load_ckpt=False, filename="")
        for m in self.metrics:
            m.reset()
        avg_loss = 0
        for batch_idx, (aux_buffer, img_in, img_gt) in enumerate(tqdm(dataloader)):
            # B == 1 in test
            img_out = filtering(ts_module, aux_buffer, img_in)

            loss = self.loss_fn(img_out[..., :3], img_gt[..., :3])
            avg_loss += loss.item()
            for m in self.metrics:
                m.measure(img_out[..., :3], img_gt[..., :3])
            
            if self.args.save_image:
                img_out[..., -1:] = 1
                self.logger.log_image(img_out, save_dir, "r", batch_idx, {"epoch": self.epoch})

        avg_loss = avg_loss / len(dataloader)
        compact_model_log = {
            "test/loss": avg_loss,
            **{f"test/{m.name()}": m.result() for m in self.metrics}
        }

        self.logger.log({
            "epoch": self.epoch,
            **compact_model_log,
        })

    def compact(self, model, load_ckpt=True, filename="ts_latest.ts"):
        if load_ckpt:
            ckpt, ckpt_path = load_checkpoint(self.args.work_dir)
            if ckpt is None:
                self.logger.print("No checkpoint found.")
                return
            self.logger.print(f"Load checkpoint from {ckpt_path}")
            model.load_state_dict(ckpt['model'])

        ts_module = compact_and_compile(model, self.device)
        if filename:
            torch.jit.save(ts_module, 
                os.path.join(self.args.work_dir, filename))
        return ts_module
