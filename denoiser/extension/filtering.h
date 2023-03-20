#pragma once
#include <torch/extension.h>

namespace denoiser {

torch::Tensor filtering(
    torch::Tensor weight_map,  // [L, H, W]
    torch::Tensor kernel_map,  // [L, H, W]
    torch::Tensor imgs_in,     // [H, W, 3]
    torch::Tensor imgs_out,    // [H, W, 3], output tensor must be given
    bool          requires_grad = false);

}  // namespace denoiser