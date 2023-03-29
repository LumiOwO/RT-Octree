import torch
import torch.nn as nn
import torch.nn.functional as F

import os

try:
    import _denoiser
except ImportError:
    print("===== Loading CUDA extension...")

    # load cpp extension
    from torch.utils.cpp_extension import load
    pwd = os.path.dirname(os.path.abspath(__file__))

    nvcc_flags = [
        '-O3', '-std=c++14',
    ]
    if os.name == "posix":
        c_flags = ['-O3', '-std=c++14']
    elif os.name == "nt":
        c_flags = ['/O2', '/std:c++17']

        # find cl.exe
        def find_cl_path():
            import glob
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

    _denoiser = load(name="_denoiser", 
                    verbose=True,
                    extra_cflags=c_flags,
                    extra_cuda_cflags=nvcc_flags,
                    sources=[os.path.join(pwd, "extension", f) for f in [
                        "bindings.cpp",
                        "filtering.cu",
                    ]])
    print("===== CUDA extension loaded.")

class RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepVGG, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding="same")

    def forward(self, x):
        x5 = self.conv5(x)
        x3 = self.conv3(x)
        x1 = self.conv1(x)
        h = x5 + x3 + x1
        if self.in_channels == self.out_channels:
            h = h + x
        return F.relu(h, inplace=True)

class DenoiserNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, num_layers, kernel_levels):
        super(DenoiserNetwork, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers
        self.kernel_levels = kernel_levels

        layers = []
        layers.append(RepVGG(in_channels, mid_channels))
        for _ in range(num_layers - 2):
            layers.append(RepVGG(mid_channels, mid_channels))
        self.layers = nn.ModuleList(layers)

        self.weight_layer = RepVGG(mid_channels, kernel_levels)
        self.kernel_layer = RepVGG(mid_channels, kernel_levels)

    def forward(self, buffers_in, img_in, requires_grad=False):
        # buffers_in [B, C, H, W]
        # imgs_in = buffers_in[..., :4]
        B = img_in.shape[0]
        # print(imgs_in.shape)

        # kernel prediction
        # x = buffers_in.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x = buffers_in
        for layer in self.layers:
            x = layer(x)

        weight_map = self.weight_layer(x) # [B, L, H, W]
        weight_map = F.softmax(weight_map, dim=1)

        kernel_map = self.kernel_layer(x) # [B, L, H, W]

        # kernel reconstruction and apply
        if B == 1:
            # filtering only support B == 1
            weight_map = weight_map.squeeze(0) # [L, H, W]
            kernel_map = kernel_map.squeeze(0) # [L, H, W]
            img_in = img_in.squeeze(0) # [H, W, 4]
            img_out = _denoiser.filtering(
                weight_map, kernel_map, img_in, requires_grad=requires_grad)
            return img_out.unsqueeze(0) # [B, H, W, 4]

        # streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        # torch.cuda.synchronize()
        img_out = torch.zeros_like(img_in)
        for i in range(B):
            # with torch.cuda.stream(streams[i % 2]):
            img_out[i] = _denoiser.filtering(
                weight_map[i], kernel_map[i], img_in[i], requires_grad=requires_grad)

        # torch.cuda.synchronize()
        return img_out


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