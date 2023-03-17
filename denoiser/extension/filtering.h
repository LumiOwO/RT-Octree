#pragma once
#include <torch/extension.h>

namespace denoiser {

torch::Tensor filtering(
    torch::Tensor input,    // [B, H, W, L * 2], will be in-place modified
    torch::Tensor imgs_in,  // [B, H, W, 3]
    torch::Tensor imgs_out  // [B, H, W, 3], output tensor must be given
);

}  // namespace denoiser