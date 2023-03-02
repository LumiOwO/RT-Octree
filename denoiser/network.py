import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# load cpp extension
from torch.utils.cpp_extension import load
pwd = os.path.dirname(os.path.abspath(__file__))
_denoiser = load(name="_denoiser", 
                 verbose=True,
                 sources=[os.path.join(pwd, "extension", f) for f in [
                     "bindings.cpp",
                     "filtering.cu",
                 ]])
                

# wrap into autograd function
class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

class RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x5 = self.conv5(x)
        x3 = self.conv3(x)
        x1 = self.conv1(x)
        h = x5 + x3 + x1
        if self.in_channels == self.out_channels:
            h = h + x
        return F.relu(h)

class DenoiserNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, num_layers, kernel_levels):
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers
        self.kernel_levels = kernel_levels

        guidance_net = []
        guidance_net.append(RepVGG(in_channels, mid_channels))
        for _ in range(num_layers - 2):
            guidance_net.append(RepVGG(mid_channels, mid_channels))
        guidance_net.append(RepVGG(mid_channels, kernel_levels * 2)) # [weights, kernels]
        self.guidance_net = nn.ModuleList(guidance_net)

    def forward(self, imgs_in):
        # kernel prediction
        x = imgs_in
        for _, layer in enumerate(self.guidance_net):
            x = layer(x)
            x = F.relu(x, inplace=True)
        # kernel reconstruction and apply
        imgs_out = torch.tensor() # [B, H, W, 3]
        _denoiser.filtering(x, imgs_in, imgs_out)
        return imgs_out


class RepVGGCompact(nn.Module):
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x, d):
        raise NotImplementedError()

class DenoiserNetworkCompact(nn.Module):
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x, d):
        raise NotImplementedError()

def compact_and_compile(model: DenoiserNetwork):
    raise NotImplementedError()