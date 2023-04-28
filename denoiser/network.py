import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# import tensorrt 
# import torch_tensorrt

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
    def __init__(self, in_channels, out_channels, num_branches):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = num_branches

        conv3 = []
        for _ in range(num_branches):
            conv3.append(nn.Conv2d(in_channels, out_channels, 3, padding="same"))
        self.conv3 = nn.ModuleList(conv3)

        conv1 = []
        for _ in range(num_branches):
            conv1.append(nn.Conv2d(in_channels, out_channels, 1, padding="same"))
        self.conv1 = nn.ModuleList(conv1)

    def forward(self, x):
        h = self.conv3[self.num_branches - 1](x)
        for i in range(self.num_branches - 1):
            h += self.conv3[i](x)
        for i in range(self.num_branches):
            h += self.conv1[i](x)

        if self.in_channels == self.out_channels:
            h += x
        return F.relu6(h, inplace=True)

def filtering(model, aux_buffer, img_in, requires_grad=False):
    # aux_buffer [B, C, H, W]
    weight_map, kernel_map = model(aux_buffer)

    # kernel reconstruction and apply
    img_out = _denoiser.filtering_autograd(
        weight_map, kernel_map, img_in, requires_grad=requires_grad)
    return img_out

class GuidanceNet(nn.Module):
    def __init__(self, in_channels, mid_channels, num_branches, num_layers, kernel_levels):
        super(GuidanceNet, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_branches = num_branches
        self.num_layers = num_layers
        self.kernel_levels = kernel_levels

        layers = []
        layers.append(RepVGGBlock(in_channels, mid_channels, num_branches))
        for _ in range(num_layers - 2):
            layers.append(RepVGGBlock(mid_channels, mid_channels, num_branches))

        layers.append(RepVGGBlock(mid_channels, kernel_levels * 2, num_branches))
        self.layers = nn.ModuleList(layers)

    def forward(self, aux_buffer):
        # aux_buffer [B, C, H, W]

        # kernel prediction
        with torch.cuda.amp.autocast():
            x = aux_buffer
            for layer in self.layers:
                x = layer(x)
        x = x.float()

        weight_map = x[:, :self.kernel_levels, ...].contiguous() # [B, L, H, W]
        weight_map = F.softmax(weight_map, dim=1)

        kernel_map = x[:, self.kernel_levels:, ...].contiguous() # [B, L, H, W]
        return weight_map, kernel_map

    def filtering(self, aux_buffer, img_in, requires_grad=False):
        return filtering(self, aux_buffer, img_in, requires_grad)

class RepVGGBlockCompact(nn.Module):
    def __init__(self, full_model):
        super(RepVGGBlockCompact, self).__init__()
        in_channels = full_model.in_channels
        out_channels = full_model.out_channels
        num_branches = full_model.num_branches

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        weight = torch.zeros_like(self.conv.weight)
        bias = torch.zeros_like(self.conv.bias)

        for i in range(num_branches):
            weight += full_model.conv3[i].weight
            bias += full_model.conv3[i].bias

        for i in range(num_branches):
            weight += F.pad(full_model.conv1[i].weight, (1, 1, 1, 1))
            bias += full_model.conv1[i].bias

        if in_channels == out_channels:
            weight_identity = torch.zeros_like(self.conv.weight)
            for i in range(out_channels):
                weight_identity[i, i % in_channels, 1, 1] = 1
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
        self.num_branches = full_model.num_branches
        self.num_layers = full_model.num_layers
        self.kernel_levels = full_model.kernel_levels

        layers = []
        for layer in full_model.layers:
            layers.append(RepVGGBlockCompact(layer))
        self.layers = nn.ModuleList(layers)

def compact_and_compile(model: GuidanceNet, device=None):
    # Compact
    model = model.eval().cpu()
    compact = GuidanceNetCompact(model).eval()

    model = model.to(device)
    compact = compact.to(device)
    compact = compact.half()

    B, C, H, W = 1, 8, 800, 800
    aux_buffer = torch.rand((B, C, H, W)).to(device)

    profile = True
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                out1, out2 = model.forward(aux_buffer)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                out1, out2 = compact.forward(aux_buffer)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # Create torchscript model
    def cast_and_forward(aux_buffer):
        aux_buffer = aux_buffer.half()
        return compact.forward(aux_buffer)

    torch._C._jit_set_profiling_mode(False)
    # guidance_net_ts = torch.jit.script(compact)
    with torch.no_grad():
        guidance_net_ts = torch.jit.trace(cast_and_forward, (aux_buffer))
    # guidance_net_ts = torch.jit.optimize_for_inference(guidance_net_ts)
    print(guidance_net_ts.code)
    
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                out1, out2 = guidance_net_ts(aux_buffer)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # trt_guidance_net_ts = torch_tensorrt.compile(
    #     guidance_net_ts, 
    #     inputs=[torch_tensorrt.Input(aux_buffer.shape, dtype=aux_buffer.dtype)],
    #     enabled_precisions={torch.float16},
    # )
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    #     with torch.no_grad():
    #         out1_trt, out2_trt = trt_guidance_net_ts(aux_buffer)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # print((out1_trt - out1).max())
    # print((out2_trt - out2).max())

    return guidance_net_ts