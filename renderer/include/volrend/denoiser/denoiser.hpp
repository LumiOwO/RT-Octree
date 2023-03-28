#pragma once
#include <torch/torch.h>

namespace volrend {

class Denoiser {
public:
    static torch::Tensor test() {
        auto tensor = torch::zeros({2,3});
        return tensor;
    }
};

}  // namespace volrend