#include "filtering.h"
#include "volrend/denoiser/denoiser.hpp"

namespace volrend {

struct Denoiser::Impl {
    void denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream) {
        const int H = cam.height;
        const int W = cam.width;
        const int L = 6;

        // Wrap with tensor
        torch::TensorOptions tensor_options =
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto aux_buffer = torch::from_blob(
            ctx.aux_buffer, {1, RenderContext::CHANNELS, H, W}, tensor_options);

        // TODO: kernel prediction
        // auto maps = model->forward(aux_buffer).toTensorVector();
        // torch::Tensor weight_map = maps[0];
        // torch::Tensor kernel_map = maps[1];
        auto weight_map = torch::ones({L, H, W}, tensor_options) / L;
        auto kernel_map = torch::rand({L, H, W}, tensor_options);

        // kernel applying
        denoiser::filtering(
            stream, weight_map, kernel_map, ctx.noisy_tex_obj, ctx.surf_obj);
    }
};

Denoiser::Denoiser()
    : impl_(std::make_unique<Denoiser::Impl>()) {}

Denoiser::~Denoiser() {}

void Denoiser::denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream) {
    impl_->denoise(cam, ctx, stream);
}

}  // namespace volrend
