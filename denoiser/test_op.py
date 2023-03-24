import torch

from pathlib import Path

from denoiser.utils import seed_everything
from denoiser.network import _denoiser, DenoiserNetwork

from torch.autograd import gradcheck
from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

def main():
    # Init
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CPU-only not supported."

    # # check filtering
    # H = 35
    # W = 35
    # L = 6
    # # B = 1
    # weight_map = torch.rand((L, H, W), dtype=torch.float32).to(device)
    # kernel_map = torch.rand((L, H, W), dtype=torch.float32).to(device)
    # imgs_in = torch.rand((H, W, 4), dtype=torch.float32).to(device)
    # # for h in range(H):
    # #     imgs_in[h] = h + 1
    # imgs_out = torch.zeros((H, W, 4), dtype=torch.float32).to(device)
    # requires_grad = True
    # weight_map.requires_grad = requires_grad
    # kernel_map.requires_grad = requires_grad

    # def f(inputs):
    #     weight_map, kernel_map, imgs_in = inputs
    #     return _denoiser.filtering(weight_map, kernel_map, imgs_in, requires_grad=requires_grad)

    # # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    #     for i in range(1):
    #         imgs_out = f((weight_map, kernel_map, imgs_in))
    # # print(imgs_in.permute(2, 0, 1))
    # # print(imgs_out.permute(2, 0, 1))
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=10))

    # numerical_jacobian = get_numerical_jacobian(f, (weight_map, kernel_map, imgs_in), eps=1e-2)
    # analytical_jacobian = get_analytical_jacobian((weight_map, kernel_map, imgs_in), imgs_out)
    # torch.set_printoptions(edgeitems=7, linewidth=200)
    # # print(numerical_jacobian[0])
    # # print()
    # # print(analytical_jacobian[0][0])
    # # print()
    # # print(numerical_jacobian[1])
    # # print()
    # # print(analytical_jacobian[0][1])
    # # print()
    # print((numerical_jacobian[0] - analytical_jacobian[0][0]).abs().max())
    # print((numerical_jacobian[1] - analytical_jacobian[0][1]).abs().max())

    # check network
    H = 800
    W = 800
    in_channels = 4
    mid_channels = 14
    num_layers = 3
    L = 6
    model = DenoiserNetwork(in_channels, mid_channels, num_layers, L).to(device)
    imgs_in = torch.rand((1, H, W, in_channels), dtype=torch.float32).to(device)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(1):
            model.forward(imgs_in)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()