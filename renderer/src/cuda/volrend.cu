#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/rt_core.cuh"
#include "volrend/internal/data_spec.hpp"
#include "volrend/render_context.hpp"
#include "volrend/render_options.hpp"

namespace volrend {

#define MAX3(a, b, c) max(max(a, b), c)
#define MIN3(a, b, c) min(min(a, b), c)

using internal::TreeSpec;
using internal::CameraSpec;

namespace {
template<typename scalar_t>
__host__ __device__ __inline__ static void screen2worlddir(
        int ix, int iy,
        const CameraSpec& cam,
        scalar_t* out,
        scalar_t* cen) {
    scalar_t xyz[3] ={ (ix - 0.5f * cam.width) / cam.fx,
                    -(iy - 0.5f * cam.height) / cam.fy, -1.0f};
    _mv3(cam.transform, xyz, out);
    _normalize(out);
    _copy3(cam.transform + 9, cen);
}
template<typename scalar_t>
__host__ __device__ __inline__ void maybe_world2ndc(
        const TreeSpec& tree,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen) {
    if (tree.ndc_width <= 0)
        return;
    scalar_t t = -(1.f + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 / cen[2];

    cen[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 / cen[2];

    _normalize(dir);
}

template<typename scalar_t>
__host__ __device__ __inline__ void rodrigues(
        const scalar_t* __restrict__ aa,
        scalar_t* __restrict__ dir) {
    scalar_t angle = _norm(aa);
    if (angle < 1e-6) return;
    scalar_t k[3];
    for (int i = 0; i < 3; ++i) k[i] = aa[i] / angle;
    scalar_t cos_angle = cos(angle), sin_angle = sin(angle);
    scalar_t cross[3];
    _cross3(k, dir, cross);
    scalar_t dot = _dot3(k, dir);
    for (int i = 0; i < 3; ++i) {
        dir[i] = dir[i] * cos_angle + cross[i] * sin_angle + k[i] * dot * (1.0 - cos_angle);
    }
}

__device__ __inline__ float clamp(
        float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

}  // namespace

namespace device {

template <int SPP>
__global__ static void render_kernel(
    const CameraSpec    cam,
    const TreeSpec      tree,
    const RenderOptions opt,
    const float*        probe_coeffs,
    RenderContext ctx  // use value, not reference
) {
    const int SIZE = cam.width * cam.height;
    CUDA_GET_THREAD_ID(idx, SIZE);

    const int x = idx % cam.width, y = idx / cam.width;
    float     dir[3], cen[3], out[4];

    bool      enable_draw = tree.N > 0;
    out[0] = out[1] = out[2] = out[3] = 0.f;
    if (opt.enable_probe && y < opt.probe_disp_size + 5 &&
        x >= cam.width - opt.probe_disp_size - 5) {
        // Draw probe circle
        float basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        int   xx = x - (cam.width - opt.probe_disp_size) + 5;
        int   yy = y - 5;
        cen[0]   = -(xx / (0.5f * opt.probe_disp_size) - 1.f);
        cen[1]   = (yy / (0.5f * opt.probe_disp_size) - 1.f);

        float c  = cen[0] * cen[0] + cen[1] * cen[1];
        if (c <= 1.f) {
            enable_draw = false;
            if (tree.data_format.basis_dim >= 0) {
                cen[2] = -sqrtf(1 - c);
                _mv3(cam.transform, cen, dir);

                internal::maybe_precalc_basis(tree, dir, basis_fn);
                for (int t = 0; t < 3; ++t) {
                    int   off = t * tree.data_format.basis_dim;
                    float tmp = 0.f;
                    for (int i = opt.basis_minmax[0]; i <= opt.basis_minmax[1];
                         ++i) {
                        tmp += basis_fn[i] * probe_coeffs[off + i];
                    }
                    out[t] = 1.f / (1.f + expf(-tmp));
                }
                out[3] = 1.f;
            } else {
                for (int i = 0; i < 3; ++i) out[i] = probe_coeffs[i];
                out[3] = 1.f;
            }
        } else {
            out[0] = out[1] = out[2] = 0.f;
        }
    }

    float t_max = 1e9f;
    if (enable_draw) {
        screen2worlddir(x, y, cam, dir, cen);
        // out[3]=1.f;
        float vdir[3] = {dir[0], dir[1], dir[2]};
        maybe_world2ndc(tree, dir, cen);
        for (int i = 0; i < 3; ++i) {
            cen[i] = tree.offset[i] + tree.scale[i] * cen[i];
        }

        if (!ctx.offscreen) {
            surf2Dread(
                &t_max,
                ctx.surf_obj_depth,
                x * sizeof(float),
                y,
                cudaBoundaryModeZero);
        }

        rodrigues(opt.rot_dirs, vdir);

        ctx.rng.advance(idx * SPP);  // init random number generator
        trace_ray<float, SPP>(
            tree, dir, vdir, cen, opt, t_max, out, ctx.rng);
    }

    float rgbx_init[4];
    if (!ctx.offscreen) {
        // Read existing values for compositing (with meshes)
        surf2Dread(
            reinterpret_cast<float4*>(rgbx_init),
            ctx.surf_obj,
            x * (int)sizeof(float4),
            y,
            cudaBoundaryModeZero);
    }

    // Compositing with existing color
    const float nalpha = 1.f - out[3];
    if (ctx.offscreen) {
        const float remain = opt.background_brightness * nalpha;
        out[0] += remain;
        out[1] += remain;
        out[2] += remain;
    } else {
        out[0] += rgbx_init[0] * nalpha;
        out[1] += rgbx_init[1] * nalpha;
        out[2] += rgbx_init[2] * nalpha;
    }

    // Write auxiliary buffer
    int aux_idx             = idx;
    ctx.aux_buffer[aux_idx] = out[0];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[1];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[2];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[3];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[0] * out[0];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[1] * out[1];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[2] * out[2];
    aux_idx += SIZE;
    ctx.aux_buffer[aux_idx] = out[3] * out[3];

    // Write image
    out[3] = 1.0f;
    cudaSurfaceObject_t dst = opt.denoise ? ctx.noisy_surf_obj : ctx.surf_obj;
    surf2Dwrite(
        *reinterpret_cast<float4*>(out),
        dst,
        x * (int)sizeof(float4),
        y,
        cudaBoundaryModeZero);
}

__global__ static void retrieve_cursor_lumisphere_kernel(
        TreeSpec tree,
        RenderOptions opt,
        float* out) {
    float cen[3];
    for (int i = 0; i < 3; ++i) {
        cen[i] = tree.offset[i] + tree.scale[i] * opt.probe[i];
    }

    float _cube_sz;
    const half* tree_val;
    internal::query_single_from_root(tree, cen, &tree_val, &_cube_sz);

    for (int i = 0; i < tree.data_dim - 1; ++i) {
        out[i] = __half2float(tree_val[i]);
    }
}


}  // namespace device

__host__ void launch_renderer(
    const N3Tree&        tree,
    const Camera&        cam,
    const RenderOptions& options,
    RenderContext&       ctx,
    cudaStream_t         stream,
    bool                 offscreen) {

    float* probe_coeffs = nullptr;
    if (options.enable_probe) {
        cuda(Malloc(&probe_coeffs, (tree.data_dim - 1) * sizeof(float)));
        device::retrieve_cursor_lumisphere_kernel<<<1, 1, 0, stream>>>(
                tree,
                options,
                probe_coeffs);
    }

    // less threads is weirdly faster for me than 1024
    // Not sure if this scales to a good GPU
    constexpr int N_CUDA_THREADS = 512;
    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, N_CUDA_THREADS);

#define __CASE(SPP)                                                        \
    {                                                                      \
        device::render_kernel<SPP><<<blocks, N_CUDA_THREADS, 0, stream>>>( \
            cam, tree, options, probe_coeffs, ctx);                        \
    }

    // render
    // clang-format off
    switch (options.spp) {
        case 1: __CASE(1); break;
        case 2: __CASE(2); break;
        case 3: __CASE(3); break;
        case 4: __CASE(4); break;
        case 6: __CASE(6); break;
        case 8: __CASE(8); break;
        case 16: __CASE(16); break;
        case 32: __CASE(32); break;
        default:
            throw std::runtime_error(
                "spp == " + std::to_string(options.spp) + " not supported.");
    }
    // clang-format on
#undef __CASE

    if (options.enable_probe) {
        cudaFree(probe_coeffs);
    }
}
}  // namespace volrend
