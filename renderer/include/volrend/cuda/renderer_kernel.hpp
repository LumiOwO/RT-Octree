#pragma once

#include <cuda_runtime.h>
#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"
#include "volrend/render_options.hpp"
#include "volrend/render_context.hpp"


namespace volrend {
__host__ void launch_renderer(const N3Tree& tree, 
                              const Camera& cam,
                              const RenderOptions& options,
                              RenderContext& ctx,
                              cudaStream_t stream, 
                              bool offscreen = false);
}  // namespace volrend
