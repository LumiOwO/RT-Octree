import torch
from pytorch_msssim import ssim
import lpips

def SMAPELoss(preds, truths):
    loss = (preds - truths).abs() / (preds.abs() + truths.abs() + 1e-5)
    return loss.mean()

class LPIPSLoss:
    def __init__(self, net, device):
        self.loss_fn = lpips.LPIPS(net=net).to(device)

    def __call__(self, preds, truths, *args, **kwargs):
        # permute from [B, H, W, C] to [B, C, H, W]
        preds = preds.permute(0, 3, 1, 2)
        truths = truths.permute(0, 3, 1, 2)
        return self.loss_fn(preds, truths, *args, **kwargs)

def get_loss_fn(name, device):
    if name == "mse":
        return torch.nn.MSELoss()
    elif name == "huber":
        return torch.nn.HuberLoss()
    elif name == "smape":
        return SMAPELoss
    elif name == "lpips_alex":
        return LPIPSLoss("alex", device)
    elif name == "lpips_vgg":
        return LPIPSLoss("vgg", device)
    else:
        raise NotImplementedError("Invalid loss funtion.")

class Metric:
    def __init__(self):
        self.sum = 0
        self.cnt = 0

    def name(self):
        raise NotImplementedError()

    def fn(self, preds, truths):
        raise NotImplementedError()

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def measure(self, preds, truths):
        # permute from [B, H, W, C] to [B, C, H, W]
        preds = preds.permute(0, 3, 1, 2)
        truths = truths.permute(0, 3, 1, 2)
        self.sum += self.fn(preds, truths)
        self.cnt += 1

    def result(self):
        return self.sum / self.cnt


def psnr(preds, truths):
    return -10 * torch.log10(((preds - truths) ** 2).mean()).item()

class PSNRMetric(Metric):
    def name(self):
        return "psnr"

    def fn(self, preds, truths):
        return -10 * torch.log10(((preds - truths) ** 2).mean()).item()

class SSIMMetric(Metric):
    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def name(self):
        return "ssim"

    def fn(self, preds, truths):
        return ssim(preds, truths, data_range=self.data_range).item()

class LPIPSMetric(Metric):
    def __init__(self, net, device):
        self.net = net
        self.loss_fn = lpips.LPIPS(net=net).eval().to(device)

    def name(self):
        return f"lpips_{self.net}"

    def fn(self, preds, truths):
        return self.loss_fn(preds, truths).item()