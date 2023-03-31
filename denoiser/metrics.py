import torch
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


def psnr(preds, truths):
    return -10 * torch.log10(((preds - truths) ** 2).mean()).item()