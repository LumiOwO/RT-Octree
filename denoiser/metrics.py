import torch

def smape(preds, truths):
    loss = (preds - truths).abs() / (preds.abs() + truths.abs() + 1e-5)
    return loss.mean()

def psnr(preds, truths):
    return -10 * torch.log10(((preds - truths) ** 2).mean()).item()