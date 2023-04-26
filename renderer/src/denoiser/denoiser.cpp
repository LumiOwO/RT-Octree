#include <torch/script.h>

#include "filtering.h"
#include "volrend/denoiser/denoiser.hpp"

namespace volrend {

struct Denoiser::Impl {
    torch::jit::script::Module ts_module;

public:
    Impl(const std::string& ts_module_path) {
        if (ts_module_path.empty()) {
            throw std::runtime_error(
                "No torchscript module is given to denoiser.");
        }

        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            ts_module = torch::jit::load(ts_module_path);

        } catch (const c10::Error& e) {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error(
                "Error when loading torchscript model from " + ts_module_path);
        }

        
    }

    void denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream) {
        const int H = cam.height;
        const int W = cam.width;
        const int L = 6;

        // Wrap with tensor
        torch::TensorOptions tensor_options =
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto aux_buffer = torch::from_blob(
            ctx.aux_buffer, {1, RenderContext::CHANNELS, H, W}, tensor_options);

        // kernel prediction
        //auto weight_map = torch::ones({L, H, W}, tensor_options) / L;
        //auto kernel_map = torch::rand({L, H, W}, tensor_options);
        auto maps = ts_module.forward({aux_buffer}).toTuple()->elements();
        torch::Tensor weight_map = maps[0].toTensor().squeeze(0); // [L, H, W]
        torch::Tensor kernel_map = maps[1].toTensor().squeeze(0); // [L, H, W]

        // kernel applying
        denoiser::filtering(
            stream, weight_map, kernel_map, ctx.noisy_tex_obj, ctx.surf_obj);
    }
};

Denoiser::Denoiser(const std::string& ts_module_path)
    : impl_(std::make_unique<Denoiser::Impl>(ts_module_path)) {}

Denoiser::~Denoiser() {}

void Denoiser::denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream) {
    impl_->denoise(cam, ctx, stream);
}

}  // namespace volrend
