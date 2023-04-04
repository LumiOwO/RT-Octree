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

__device__ __inline__ void read_rgba_float4(
        const float* img_arr, float* rgba, int i) {
    i <<= 2; // i *= 4;
    rgba[0] = img_arr[i];
    rgba[1] = img_arr[i + 1];
    rgba[2] = img_arr[i + 2];
    rgba[3] = img_arr[i + 3];
}

__device__ __inline__ void write_rgba_float4(
        float* img_arr, const float* rgba, int i) {
    i <<= 2; // i *= 4;
    img_arr[i] = rgba[0];
    img_arr[i + 1] = rgba[1];
    img_arr[i + 2] = rgba[2];
    img_arr[i + 3] = rgba[3];
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
        RenderContext ctx, // use value, not reference
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
    float depth = 0;
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

        if (opt.delta_tracking) {
            constexpr int spp = 1;
            constexpr float inv_spp = 1.0f / spp;
            ctx.rng.advance(idx * spp);  // init random number generator
            // const float dst = -__logf(1.0f - ctx.rng.next_float());
            delta_trace_ray<float, 1>(
                tree, dir, vdir, cen, opt, t_max, out, ctx.rng, depth);

            out[0] *= inv_spp;
            out[1] *= inv_spp;
            out[2] *= inv_spp;
            out[3] *= inv_spp;
        } else {
            trace_ray(tree, dir, vdir, cen, opt, t_max, out);
        }
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
    // float alpha;
    // if (opt.delta_tracking) {
    //     alpha = (out[3] > 0);
    // } else {
    //     alpha = out[3];
    // }
    // const float nalpha = 1.f - alpha;
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
    
    if (opt.delta_tracking) {
        // write float colors into delta tracking context
        // float&& alpha = 1.0f / ctx.spp;
        // float&& n_alpha = 1.0f - alpha;
        // float* dst = &ctx.data[CUR_RGBA][idx << 2];
        // dst[0] = out[0] * alpha + dst[0] * n_alpha;
        // dst[1] = out[1] * alpha + dst[1] * n_alpha;
        // dst[2] = out[2] * alpha + dst[2] * n_alpha;
        // dst[3] = out[3] * alpha + dst[3] * n_alpha;

        // float& dst_d = ctx.data[CUR_D][idx];
        // dst_d = depth * alpha + dst_d * n_alpha;

        float4& rgba_noisy = ctx.rgba_noisy[idx];
        rgba_noisy.x = out[0];
        rgba_noisy.y = out[1];
        rgba_noisy.z = out[2];
        rgba_noisy.w = out[3];
        // rgba_noisy.w = 1.0f - __expf(-out[3]); // normalization
        // rgba_noisy.w = fminf(out[3] * 0.001f, 1.0f); // normalization

        ctx.depth_noisy[idx] = fminf(depth * 0.3f, 1.0f);
    }
}

template <int SPP>
__global__ static void render_kernel_delta_trace(
    cudaSurfaceObject_t surf_obj,
    cudaSurfaceObject_t surf_obj_depth,
    CameraSpec          cam,
    TreeSpec            tree,
    RenderOptions       opt,
    float*              probe_coeffs,
    RenderContext       ctx,  // use value, not reference
    bool                offscreen) {
    CUDA_GET_THREAD_ID(idx, cam.width * cam.height);

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
    float depth = 0;
    if (enable_draw) {
        screen2worlddir(x, y, cam, dir, cen);
        // out[3]=1.f;
        float vdir[3] = {dir[0], dir[1], dir[2]};
        maybe_world2ndc(tree, dir, cen);
        for (int i = 0; i < 3; ++i) {
            cen[i] = tree.offset[i] + tree.scale[i] * cen[i];
        }

        if (!offscreen) {
            surf2Dread(
                &t_max,
                surf_obj_depth,
                x * sizeof(float),
                y,
                cudaBoundaryModeZero);
        }

        rodrigues(opt.rot_dirs, vdir);

        constexpr float INV_SPP = 1.0f / SPP;
        ctx.rng.advance(idx * SPP);  // init random number generator
        // const float dst = -__logf(1.0f - ctx.rng.next_float());
        delta_trace_ray<float, SPP>(
            tree, dir, vdir, cen, opt, t_max, out, ctx.rng, depth);

        out[0] *= INV_SPP;
        out[1] *= INV_SPP;
        out[2] *= INV_SPP;
        out[3] *= INV_SPP;
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
    float rgbx[4] = {out[0], out[1], out[2], 1.0f};
    surf2Dwrite(
        *reinterpret_cast<float4*>(rgbx),
        surf_obj,
        x * (int)sizeof(float4),
        y,
        cudaBoundaryModeZero);  // squelches out-of-bound writes

    // write float colors into delta tracking context
    // float&& alpha = 1.0f / ctx.spp;
    // float&& n_alpha = 1.0f - alpha;
    // float* dst = &ctx.data[CUR_RGBA][idx << 2];
    // dst[0] = out[0] * alpha + dst[0] * n_alpha;
    // dst[1] = out[1] * alpha + dst[1] * n_alpha;
    // dst[2] = out[2] * alpha + dst[2] * n_alpha;
    // dst[3] = out[3] * alpha + dst[3] * n_alpha;

    // float& dst_d = ctx.data[CUR_D][idx];
    // dst_d = depth * alpha + dst_d * n_alpha;

    float4& rgba_noisy = ctx.rgba_noisy[idx];
    rgba_noisy.x       = out[0];
    rgba_noisy.y       = out[1];
    rgba_noisy.z       = out[2];
    rgba_noisy.w       = out[3];
    // rgba_noisy.w = 1.0f - __expf(-out[3]); // normalization
    // rgba_noisy.w = fminf(out[3] * 0.001f, 1.0f); // normalization

    ctx.depth_noisy[idx] = fminf(depth * 0.3f, 1.0f);
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

__global__ void temporal_accumulate(
        cudaSurfaceObject_t surf_obj,
        cudaSurfaceObject_t surf_obj_depth,
        CameraSpec cam,
        RenderContext ctx,
        RenderOptions opt,
        bool offscreen) {
    CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
    const int x = idx % cam.width, y = idx / cam.width;

    // get pixel information
    float rgba[4];
    read_rgba_float4(ctx.data[CUR_RGBA], rgba, idx);
    if (!opt.denoise || !ctx.has_history) {
        ; // do nothing
    } else {
        // reprojection
        float dir[3];
        float cen[3];
        screen2worlddir(x, y, cam, dir, cen);
        float cur_d = ctx.data[CUR_D][idx];
        float world_pos[3];
        for (int i = 0; i < 3; i++) {
            world_pos[i] = (cen[i] + cur_d * dir[i]);
        }
        float prev_x, prev_y;
        CameraSpec prev_cam = cam;
        prev_cam.transform = ctx.prev_transform_device;
        world2screen(prev_cam, world_pos, prev_x, prev_y);

        // check
        float alpha = ctx.spp / (ctx.spp + opt.prev_weight);
        int prev_ix = x;
        int prev_iy = y;
        float dx = x - prev_x;
        float dy = y - prev_y;
        if ((dx * dx + dy * dy) > 1.0f) {
            prev_ix = roundf(prev_x);
            prev_iy = roundf(prev_y);
        }
        if (prev_ix < 0 || prev_ix >= cam.width || prev_iy < 0 || prev_iy >= cam.height) {
            prev_ix = x;
            prev_iy = y;
            alpha = 1.0f;
        }

        // clamp
        float prev_rgba[4];
        read_rgba_float4(ctx.data[PREV_RGBA], prev_rgba, prev_iy * cam.width + prev_ix);
        if (opt.clamp && alpha < 1.0f) {
            float mean[4] = {};
            float sigma[4] = {};
            float buf[4] = {};

            int support = opt.clamp_support;
            int cnt = 0;
            for (int xx = x - support; xx <= x + support; xx++) {
                if (xx < 0 || xx >= cam.width) continue;
                int i = idx + xx - x - support * cam.width;
                for (int yy = y - support; yy <= y + support; yy++, i += cam.width) {
                    if (yy < 0 || yy >= cam.height) continue;
                    cnt++;
                    read_rgba_float4(ctx.data[CUR_RGBA], buf, i);
#pragma unroll 4
                    for (int i = 0; i < 4; i++) {
                        mean[i] += buf[i];
                        sigma[i] += buf[i] * buf[i];
                    }
                }
            }
            float w = 1.0f / cnt;
#pragma unroll 4
            for (int i = 0; i < 4; i++) {
                mean[i] *= w;
                sigma[i] *= w;
                sigma[i] = sqrtf(sigma[i] - mean[i] * mean[i] + 1e-5);

                prev_rgba[i] = clamp(
                    prev_rgba[i], 
                    mean[i] - opt.clamp_k * sigma[i],
                    mean[i] + opt.clamp_k * sigma[i]
                );
            }

        }
        // remove leaked camera ray
        float prev_d = ctx.data[PREV_D][idx];
        if (fabsf(prev_d - cur_d) > opt.depth_diff_thresh) {
            alpha = 0.f;
        }
        

        // alpha blend
        const float nalpha = 1.0f - alpha;
    #pragma unroll
        for (int i = 0; i < 4; i++) {
            rgba[i] = rgba[i] * alpha + prev_rgba[i] * nalpha;
        }
    }

    // output
    float out[4];
    if (opt.show_ctx == CUR_RGBA) {
        read_rgba_float4(ctx.data[CUR_RGBA], out, idx);
    } else if (opt.show_ctx == PREV_RGBA) {
        read_rgba_float4(ctx.data[PREV_RGBA], out, idx);
    } else if (opt.show_ctx == CUR_D) {
        out[0] = out[1] = out[2] = fminf(ctx.data[CUR_D][idx] * 0.3f, 1.0f);
        out[3] = 1.0f;
    } else if (opt.show_ctx == PREV_D) {
        out[0] = out[1] = out[2] = fminf(ctx.data[PREV_D][idx] * 0.3f, 1.0f);
        out[3] = 1.0f;
    } else {
        out[0] = rgba[0];
        out[1] = rgba[1];
        out[2] = rgba[2];
        out[3] = rgba[3];
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
    const float bg_alpha = 1.f - out[3];
    if (offscreen) {
        const float remain = opt.background_brightness * bg_alpha;
        out[0] += remain;
        out[1] += remain;
        out[2] += remain;
    } else {
        out[0] += rgbx_init[0] * bg_alpha;
        out[1] += rgbx_init[1] * bg_alpha;
        out[2] += rgbx_init[2] * bg_alpha;
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

__global__ void update_prev_frame(
        RenderContext ctx,
        RenderOptions opt,
        CameraSpec cam) {
    const int& width = cam.width;
    const int& height = cam.height;
    CUDA_GET_THREAD_ID(idx, width * height);
    // const int x = idx % width, y = idx / width;

    // save rgba to prev frame
    float rgba[4];
    read_rgba_float4(ctx.data[CUR_RGBA], rgba, idx);
    float&& alpha = ctx.spp / (ctx.spp + opt.prev_weight);
    float&& n_alpha = 1.0f - alpha;
    float* dst = &ctx.data[PREV_RGBA][idx << 2];
    dst[0] = rgba[0] * alpha + dst[0] * n_alpha;
    dst[1] = rgba[1] * alpha + dst[1] * n_alpha;
    dst[2] = rgba[2] * alpha + dst[2] * n_alpha;
    dst[3] = rgba[3] * alpha + dst[3] * n_alpha;

    float& dst_d = ctx.data[PREV_D][idx];
    dst_d = ctx.data[CUR_D][idx] * alpha + dst_d * n_alpha;
}

}  // namespace device

__host__ void launch_renderer(const N3Tree& tree,
        const Camera& cam, const RenderOptions& options, cudaArray_t& image_arr,
        cudaArray_t& depth_arr,
        cudaStream_t stream,
        RenderContext& ctx,
        bool offscreen) {
    cudaSurfaceObject_t surf_obj = 0, surf_obj_depth = 0;

    float* probe_coeffs = nullptr;
    if (options.enable_probe) {
        cuda(Malloc(&probe_coeffs, (tree.data_dim - 1) * sizeof(float)));
        device::retrieve_cursor_lumisphere_kernel<<<1, 1, 0, stream>>>(
                tree,
                options,
                probe_coeffs);
    }

    {
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = image_arr;
        cudaCreateSurfaceObject(&surf_obj, &res_desc);
    }
    if (!offscreen) {
        {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = depth_arr;
            cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
        }
    }

    // less threads is weirdly faster for me than 1024
    // Not sure if this scales to a good GPU
    const int N_CUDA_THREADS = 512;
    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, N_CUDA_THREADS);

    // camera compare
    if (options.delta_tracking) {
        bool same_pose = true;
        if (!ctx.cam_inited) {
            ctx.recordCamera(cam);
            ctx.cam_inited = true;
        } else {
            // check camera pose
            auto&& a = ctx.prev_transform_host;
            auto&& b = (float*)&cam.transform;
#pragma unroll 12
            for (int i = 0; i < 12; i++) {
                if (fabsf(a[i] - b[i]) > 1e-5) {
                    same_pose = false;
                    break;
                }
            }
        }
        
        // update frame buffer
        constexpr static int MAX_SPP = 1e6;
        if (same_pose) {
            if (ctx.spp < MAX_SPP) ctx.spp++;
        } else {
            // copy current frame to previous buffer
            device::update_prev_frame<<<blocks, N_CUDA_THREADS, 0, stream>>>(
                ctx,
                options,
                cam
            );
            // record camera
            ctx.recordCamera(cam);
            if (!ctx.has_history) ctx.has_history = true;
            ctx.spp = 1;
        }
    }

    // render
    if (options.delta_tracking) {
        switch (options.spp) {
            case 1:
                device::render_kernel_delta_trace<1>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            case 2:
                device::render_kernel_delta_trace<2>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            case 3:
                device::render_kernel_delta_trace<3>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            case 4:
                device::render_kernel_delta_trace<4>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            case 8:
                device::render_kernel_delta_trace<8>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            case 16:
                device::render_kernel_delta_trace<16>
                    <<<blocks, N_CUDA_THREADS, 0, stream>>>(
                        surf_obj,
                        surf_obj_depth,
                        cam,
                        tree,
                        options,
                        probe_coeffs,
                        ctx,
                        offscreen);
                break;
            default:
                throw std::runtime_error(
                    "spp == " + std::to_string(options.spp) +
                    " not supported.");
        }
        // // temporal denoise
        // device::temporal_accumulate<<<blocks, N_CUDA_THREADS, 0, stream>>>(
        //     surf_obj,
        //     surf_obj_depth,
        //     cam,
        //     ctx,
        //     options,
        //     image_arr
        // );

        // auto temp = torch.cat(
        //     {
        //         ctx.rgba_noisy.permute({2, 0, 1}),   // [4, H, W]
        //         ctx.depth_noisy.permute({2, 0, 1}),  // [1, H, W]
        //     },
        //     /*dim=*/0);
        // std::cout << temp.sizes() << std::endl;
        // std::cout << temp.is_contiguous() << std::endl;

        // update rng
        ctx.rng.advance();
    } else {

        device::render_kernel<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            surf_obj_depth,
            cam,
            tree,
            options,
            probe_coeffs,
            ctx,
            offscreen);
    }

    if (options.enable_probe) {
        cudaFree(probe_coeffs);
    }
}
}  // namespace volrend
