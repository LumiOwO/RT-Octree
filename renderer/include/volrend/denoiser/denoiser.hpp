#pragma once

#include <memory>

#include "volrend/camera.hpp"
#include "volrend/render_context.hpp"

namespace volrend {

class Denoiser final {
public:
    // static torch::Tensor test() {
    //     torch::TensorOptions tensor_options =
    //         torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    //     auto weight_map = torch::rand({3, 25, 25}, tensor_options);
    //     auto kernel_map = torch::rand({3, 25, 25}, tensor_options);
    //     auto imgs_in = torch::rand({25, 25, 4}, tensor_options);

    //     return denoiser::filtering_autograd(weight_map, kernel_map, imgs_in);
    // }

    Denoiser();
    virtual ~Denoiser();

    void denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend