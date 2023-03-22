#pragma once
#include <torch/extension.h>

namespace denoiser {

torch::Tensor filtering(
    torch::Tensor weight_map,  // [L, B, H, W]
    torch::Tensor kernel_map,  // [L, B, H, W]
    torch::Tensor imgs_in,     // [B, H, W, 3]
    bool          requires_grad = false);

}  // namespace denoiser