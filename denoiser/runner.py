import torch

from tqdm import tqdm, trange
from denoiser.utils import load_checkpoint
from denoiser.network import compact_and_compile

class Runner(object):
    def __init__(self, args, dataloader, device=None):
        self.args = args
        self.dataloader = dataloader
        self.device = device

        # loggers
        if args.use_wandb:
            self.logger = WandbLogger()
        else:
            self.logger = BaseLogger()

        # model
        self.model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            sigma_act=opt.sigma_act,
            color_act=opt.color_act
        )
        self.loss_fn = None
        self.optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        self.scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))


    def train(self, model):
        # init
        dataloader = self.dataset.dataloader("train")
        optimizer = torch.optim.Adam(model.parameters(), 
                lr=self.args.lr, betas=(0.9, 0.999), weight_decay=5e-4) 
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: 0.1 ** min(epoch / self.args.epochs, 1))
        start = 1

        # load checkpoint
        ckpt = load_checkpoint(path, "train")
        if ckpt is not None:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            start = ckpt["epoch"]

        # train
        for epoch in trange(start, self.args.epochs + 1):
            self.train_one_epoch(model, dataloader, optimizer, lr_scheduler, epoch)

        # test after training
        self.test(model, False)

    def train_one_epoch(self, model, dataloader, optimizer, lr_scheduler, epoch):
        for batch_idx, (imgs_in, imgs_gt) in enumerate(dataloader):
            optimizer.zero_grad()
            imgs_out = model.forward(imgs_in)
            loss = self.loss_fn(imgs_out, imgs_gt)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # log
        if epoch % self.args.i_weights == 0:
            path = os.path.join(basedir, expname, f"{epoch:06d}.tar")
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
        # batch_size == 1 in test phase
        for batch_idx, (imgs_in, imgs_gt) in enumerate(dataloader):
            imgs_out = model.forward(imgs_in)
            loss = self.loss_fn(imgs_out, imgs_gt)

    def compact(self, model):
        trt_ts_module = compact_and_compile(model)
        torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript
