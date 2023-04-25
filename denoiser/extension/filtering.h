#pragma once
#include <cuda_runtime.h>
#include <torch/all.h>

namespace denoiser {

void filtering(
    cudaStream_t        stream,
    torch::Tensor       weight_map,  // [L, H, W]
    torch::Tensor       kernel_map,  // [L, H, W]
    cudaTextureObject_t img_in,      // [H, W, 4]
    cudaSurfaceObject_t img_out      // [H, W, 4]
);

torch::Tensor filtering_autograd(
    torch::Tensor weight_map,  // [L, H, W]
    torch::Tensor kernel_map,  // [L, H, W]
    torch::Tensor img_in,      // [H, W, 4]
    bool          requires_grad = false);

}  // namespace denoiser