class Runner(object):
    def __init__(self, args, dataloader, device=None):
        self.args = args
        self.dataloader = dataloader
        self.device = device

        # loggers
        if args.use_wandb:
            self.logger = WandbLogger()
        else
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
        
    
    def train(self):
        pass
    
    def eval(self):
        pass
    