import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import torch_tensorrt

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

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding="same")

    def forward(self, x):
        # x5 = self.conv5(x)
        x3 = self.conv3(x)
        x3_2 = self.conv3_2(x)
        x1 = self.conv1(x)
        h = x3 + x1 + x3_2
        if self.in_channels == self.out_channels:
            h = h + x
        return F.relu6(h, inplace=True)

class GuidanceNet(nn.Module):
    def __init__(self, in_channels, mid_channels, num_layers, kernel_levels):
        super(GuidanceNet, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers
        self.kernel_levels = kernel_levels

        layers = []
        layers.append(RepVGGBlock(in_channels, mid_channels))
        for _ in range(num_layers - 2):
            layers.append(RepVGGBlock(mid_channels, mid_channels))

        layers.append(RepVGGBlock(mid_channels, kernel_levels * 2))
        self.layers = nn.ModuleList(layers)

    def forward(self, buffers_in):
        # buffers_in [B, C, H, W]
        # imgs_in = buffers_in[..., :4]
        # print(imgs_in.shape)

        # kernel prediction
        # x = buffers_in.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x = buffers_in
        with torch.cuda.amp.autocast():
            for layer in self.layers:
                x = layer(x)
        x = x.float()

        weight_map = x[:, :self.kernel_levels, ...].contiguous() # [B, L, H, W]
        weight_map = F.softmax(weight_map, dim=1)

        kernel_map = x[:, self.kernel_levels:, ...].contiguous() # [B, L, H, W]
        return weight_map, kernel_map

    def filtering(self, buffers_in, img_in, requires_grad=False):
        # # buffers_in [B, C, H, W]
        # # imgs_in = buffers_in[..., :4]
        # B = img_in.shape[0]
        # # print(imgs_in.shape)

        # # kernel prediction
        # # x = buffers_in.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        # x = buffers_in
        # for layer in self.layers:
        #     x = layer(x)

        # weight_map = self.weight_layer(x) # [B, L, H, W]
        # weight_map = F.softmax(weight_map, dim=1)

        # kernel_map = self.kernel_layer(x) # [B, L, H, W]
        weight_map, kernel_map = self.forward(buffers_in)

        # kernel reconstruction and apply
        B = img_in.shape[0]

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

class RepVGGBlockCompact(nn.Module):
    def __init__(self, full_model):
        super(RepVGGBlockCompact, self).__init__()
        self.in_channels = full_model.in_channels
        self.out_channels = full_model.out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding="same")
        weight = torch.zeros_like(self.conv.weight)
        bias = torch.zeros_like(self.conv.bias)

        weight += full_model.conv3.weight
        bias += full_model.conv3.bias

        weight += F.pad(full_model.conv1.weight, (1, 1, 1, 1))
        bias += full_model.conv1.bias

        if self.in_channels == self.out_channels:
            weight_identity = torch.zeros_like(self.conv.weight)
            for i in range(self.out_channels):
                weight_identity[i, i % self.in_channels, 1, 1] = 1
            weight += weight_identity

        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = nn.Parameter(bias)
        self.conv.requires_grad_(False)

    def forward(self, x):
        h = self.conv(x)
        return F.relu6(h, inplace=True)

class GuidanceNetCompact(GuidanceNet):
    def __init__(self, full_model):
        super(GuidanceNet, self).__init__()
        self.in_channels = full_model.in_channels
        self.mid_channels = full_model.mid_channels
        self.num_layers = full_model.num_layers
        self.kernel_levels = full_model.kernel_levels

        layers = []
        for layer in full_model.layers:
            layers.append(RepVGGBlockCompact(layer))
        self.layers = nn.ModuleList(layers)

def compact_and_compile(model: GuidanceNet, device=None):
    # Compact
    model = GuidanceNet(
        model.in_channels, model.mid_channels, model.num_layers, model.kernel_levels)
    compact = GuidanceNetCompact(model)
    model = model.half()
    compact = compact.half()
    compact = compact.eval()

    # Check compact correctness
    B, C, H, W = 1, 4, 800, 800
    model = model.to(device)
    compact = compact.to(device)
    buffers_in = torch.rand((B, C, H, W)).to(device)
    img_in = torch.rand((B, H, W, 4)).to(device)
    # with torch.no_grad():
    #     out1 = model(buffers_in, img_in)
    #     out2 = compact(buffers_in, img_in)
    #     print((out2 - out1).max())
    #     assert ((out1 - out2).abs() < 1e-4).all(), "Compact check failed."

    # buffers_in = buffers_in.half()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            # buffers_in = buffers_in.half()
            out1, out2 = model.forward(buffers_in)
            # buffers_in = buffers_in.float()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            # buffers_in = buffers_in.half()
            out1, out2 = compact.forward(buffers_in)
            # buffers_in = buffers_in.float()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    buffers_in = buffers_in.half()
    # compact_script = torch.jit.script(compact)
    with torch.no_grad():
        guidance_net_ts = torch.jit.trace(compact.forward, (buffers_in))
    buffers_in = buffers_in.float()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            buffers_in = buffers_in.half()
            out1, out2 = guidance_net_ts(buffers_in)
    print(out1.dtype)
    buffers_in = buffers_in.float()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    trt_guidance_net_ts = torch_tensorrt.compile(
        guidance_net_ts, 
        inputs=[torch_tensorrt.Input(buffers_in.shape, dtype=torch.float16)],
        enabled_precisions={torch.float16},
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            buffers_in = buffers_in.half()
            out1_trt, out2_trt = trt_guidance_net_ts(buffers_in)
            buffers_in = buffers_in.float()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    print((out1_trt - out1).max())
    print((out2_trt - out2).max())

    return trt_guidance_net_ts