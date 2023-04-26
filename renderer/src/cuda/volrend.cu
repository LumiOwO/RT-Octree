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

template <typename scalar_t>
__host__ __device__ __inline__ static void world2screen(
    const CameraSpec& cam,
    const scalar_t* pos,
    scalar_t& x, scalar_t& y) {
    // matrix 4 x 3
    //  0   1   2
    //  3   4   5  
    //  6   7   8
    //  9  10  11 
    //transform[0] = v_right;
    //transform[1] = v_up;
    //transform[2] = v_back;
    //transform[3] = center;

    // world -> camera
    scalar_t w_xyz[3];
    w_xyz[0] = pos[0] - cam.transform[9];
    w_xyz[1] = pos[1] - cam.transform[10];
    w_xyz[2] = pos[2] - cam.transform[11];
    scalar_t c_x = _dot3(w_xyz, cam.transform);
    scalar_t c_y = _dot3(w_xyz, cam.transform + 3);
    scalar_t c_z = _dot3(w_xyz, cam.transform + 6);

    // camera -> image
    scalar_t c_z_inv = 1.0f / c_z;
    x = cam.width - (cam.fx * c_x * c_z_inv + (cam.width * 0.5f));
    y = (cam.fy * c_y * c_z_inv + (cam.height * 0.5f));
}

template <typename scalar_t>
__host__ __device__ __inline__ scalar_t luminance(
        const scalar_t* rgb) {
    constexpr static scalar_t coe[3] = {0.2126, 0.7152, 0.0722};
    return _dot3(rgb, coe);
}

__device__ __inline__ float clamp(
        float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

}  // namespace

namespace device {

// Primary rendering kernel
__global__ static void render_kernel(
        cudaSurfaceObject_t surf_obj,
        cudaSurfaceObject_t surf_obj_depth,
        CameraSpec cam,
        TreeSpec tree,
        RenderOptions opt,
        float* probe_coeffs,
        RenderContext ctx,
        bool offscreen) {
    CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
    
    const int x = idx % cam.width, y = idx / cam.width;
    float dir[3], cen[3], out[4];
    
    bool enable_draw = tree.N > 0;
    out[0] = out[1] = out[2] = out[3] = 0.f;
    if (opt.enable_probe && y < opt.probe_disp_size + 5 &&
                            x >= cam.width - opt.probe_disp_size - 5) {
        // Draw probe circle
        float basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        int xx = x - (cam.width - opt.probe_disp_size) + 5;
        int yy = y - 5;
        cen[0] = -(xx / (0.5f * opt.probe_disp_size) - 1.f);
        cen[1] = (yy / (0.5f * opt.probe_disp_size) - 1.f);

        float c = cen[0] * cen[0] + cen[1] * cen[1];
        if (c <= 1.f) {
            enable_draw = false;
            if (tree.data_format.basis_dim >= 0) {
                cen[2] = -sqrtf(1 - c);
                _mv3(cam.transform, cen, dir);

                internal::maybe_precalc_basis(tree, dir, basis_fn);
                for (int t = 0; t < 3; ++t) {
                    int off = t * tree.data_format.basis_dim;
                    float tmp = 0.f;
                    for (int i = opt.basis_minmax[0]; i <= opt.basis_minmax[1]; ++i) {
                        tmp += basis_fn[i] * probe_coeffs[off + i];
                    }
                    out[t] = 1.f / (1.f + expf(-tmp));
                }
                out[3] = 1.f;
            } else {
                for (int i = 0; i < 3; ++i)
                    out[i] = probe_coeffs[i];
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

        if (!offscreen) {
            surf2Dread(&t_max, surf_obj_depth, x * sizeof(float), y, cudaBoundaryModeZero);
        }

        rodrigues(opt.rot_dirs, vdir);

        trace_ray(tree, dir, vdir, cen, opt, t_max, out);
    }

    float rgbx_init[4];
    if (!offscreen) {
        // Read existing values for compositing (with meshes)
        surf2Dread(
            reinterpret_cast<float4*>(rgbx_init), 
            surf_obj, 
            x * (int)sizeof(float4),
            y, 
            cudaBoundaryModeZero);
    }

    // Compositing with existing color
    const float nalpha = 1.f - out[3];
    if (offscreen) {
        const float remain = opt.background_brightness * nalpha;
        out[0] += remain;
        out[1] += remain;
        out[2] += remain;
    } else {
        out[0] += rgbx_init[0] * nalpha;
        out[1] += rgbx_init[1] * nalpha;
        out[2] += rgbx_init[2] * nalpha;
    }

    // Output pixel color
    float rgbx[4] = {
        out[0],
        out[1],
        out[2],
        1.0f
    };
    surf2Dwrite(
        *reinterpret_cast<float4*>(rgbx),
        surf_obj,
        x * (int)sizeof(float4),
        y,
        cudaBoundaryModeZero); // squelches out-of-bound writes
    
}

template <int SPP>
__global__ static void render_kernel_delta_trace(
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
        delta_trace_ray<float, SPP>(
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

    // render
    switch (options.spp) {
        case 1:
            device::render_kernel_delta_trace<1>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 2:
            device::render_kernel_delta_trace<2>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 3:
            device::render_kernel_delta_trace<3>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 4:
            device::render_kernel_delta_trace<4>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 8:
            device::render_kernel_delta_trace<8>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 16:
            device::render_kernel_delta_trace<16>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        case 32:
            device::render_kernel_delta_trace<32>
                <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                    cam,
                    tree,
                    options,
                    probe_coeffs,
                    ctx);
            break;
        default:
            throw std::runtime_error(
                "spp == " + std::to_string(options.spp) + " not supported.");
    }

    if (options.enable_probe) {
        cudaFree(probe_coeffs);
    }
}
}  // namespace volrend
