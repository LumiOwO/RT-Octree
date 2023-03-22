import torch

from pathlib import Path

from denoiser.utils import seed_everything
from denoiser.runner import Runner
from denoiser.dataset import DenoiserDataset
from denoiser.logger.base_logger import BaseLogger
from denoiser.logger.wandb_logger import WandbLogger

import configargparse

def main():
    # Init
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CPU-only not supported."

    # test
    from denoiser.network import _denoiser
    print("compile succeed")
    H = 25
    W = 25
    L = 6
    x = torch.rand((L * 2, H, W), dtype=torch.float64).to(device)
    imgs_in = torch.rand((H, W, 3), dtype=torch.float64).to(device)
    for j in range(H):
        imgs_in[j] = j
    imgs_out = torch.zeros((H, W, 3), dtype=torch.float64).to(device)
    x.requires_grad = True

    # forward
    # print("before filtering")
    # print(f"{x.requires_grad=}")
    # imgs_out = _denoiser.filtering(x[:L], x[L:], imgs_in, imgs_out, requires_grad=True)
    # print("after filtering")
    # print(imgs_in.permute(2, 0, 1))
    # print(imgs_out.permute(2, 0, 1))
    # loss = imgs_out.sum()
    # loss.backward()

    # time
    # print('warm up ...\n')
    # with torch.no_grad():
    #     for _ in range(1):
    #         imgs_out = _denoiser.filtering(x, imgs_in, imgs_out)
    # torch.cuda.synchronize()

    # # import numpy as np
    # # print('test time ...\n')
    # # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # iters = 100
    # # timings = np.zeros((iters, 1))
    # # with torch.no_grad():
    # #     for rep in range(iters):
    # #         starter.record()
    # #         imgs_out = _denoiser.filtering(x, imgs_in, imgs_out)
    # #         ender.record()
    # #         torch.cuda.synchronize()
    # #         curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
    # #         timings[rep] = curr_time
    # # avg = timings.sum() / iters
    # # print(f"{avg=}")

    # from denoiser.network import DenoiserNetwork
    # model = DenoiserNetwork(3, 14, 3, 6).to(device)
    # imgs_in = torch.rand((1, H, W, 3), dtype=torch.float32).to(device)
    
    # print(imgs_in.permute(0, 3, 1, 2))
    # print(imgs_out.permute(0, 3, 1, 2))
    # loss = imgs_out.sum()
    # print("before backward")
    # loss.backward()
    # print("after backward")
    # print(x.grad.permute(3, 0, 1, 2) / 3)
    # print(x)

    # check grad
    from torch.autograd import gradcheck
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    H = 800
    W = 800
    L = 6
    B = 1
    weight_map = torch.rand((L, B, H, W), dtype=torch.float32).to(device)
    kernel_map = torch.rand((L, B, H, W), dtype=torch.float32).to(device)
    imgs_in = torch.rand((B, H, W, 3), dtype=torch.float32).to(device)
    # for b in range(B):
    #     for h in range(H):
    #         for w in range(W):
    #             for c in range(3):
    #                 imgs_in[b][h][w][c] = c
    # print(imgs_in)
    imgs_out = torch.zeros((B, H, W, 3), dtype=torch.float32).to(device)
    weight_map.requires_grad = True
    kernel_map.requires_grad = True

    def f(inputs):
        weight_map, kernel_map, imgs_in = inputs
        return _denoiser.filtering(weight_map, kernel_map, imgs_in, requires_grad=True)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        imgs_out = f((weight_map, kernel_map, imgs_in))
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))

    # numerical_jacobian = get_numerical_jacobian(f, (weight_map, kernel_map, imgs_in), eps=1e-4)
    # analytical_jacobian = get_analytical_jacobian((weight_map, kernel_map, imgs_in), imgs_out)
    # print(numerical_jacobian[0])
    # print(analytical_jacobian[0][0])
    # print((numerical_jacobian[0] - analytical_jacobian[0][0]).abs().max())
    # print((numerical_jacobian[1] - analytical_jacobian[0][1]).abs().max())


if __name__ == "__main__":
    main()