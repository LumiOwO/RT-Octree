#pragma once
#include <torch/extension.h>

namespace denoiser {

void filtering(
    torch::Tensor input,
    torch::Tensor imgs_in,
    torch::Tensor imgs_out);

}  // namespace denoiser