#pragma once

#include <memory>
#include <string>

#include "volrend/camera.hpp"
#include "volrend/render_context.hpp"

namespace volrend {

class Denoiser final {
public:
    Denoiser(const std::string& ts_module_path);
    virtual ~Denoiser();

    void denoise(const Camera& cam, RenderContext& ctx, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend