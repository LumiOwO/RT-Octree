#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cuda_fp16.h>

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/rt_core.cuh"
#include "volrend/render_options.hpp"
#include "volrend/internal/data_spec.hpp"
#include "volrend/render_context.hpp"

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
template<typename scalar_t>
__host__ __device__ __inline__ scalar_t luminance(const scalar_t* rgb) {
    constexpr static scalar_t coe[3] = {0.2126, 0.7152, 0.0722};
    return _dot3(rgb, coe);
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
    uint8_t rgbx_init[4];
    if (!offscreen) {
        // Read existing values for compositing (with meshes)
        surf2Dread(reinterpret_cast<uint32_t*>(rgbx_init), surf_obj, x * 4,
                y, cudaBoundaryModeZero);
    }

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
    float depth = t_max;
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
            ctx.rng.advance(idx); // init random number generator
            const float dst = -__logf(1.0f - ctx.rng.next_float());
            delta_trace_ray(tree, dir, vdir, cen, opt, t_max, out, dst, depth);
        } else {
            trace_ray(tree, dir, vdir, cen, opt, t_max, out);
        }
    }

    if (!opt.delta_tracking) {
        // Compositing with existing color
        const float nalpha = 1.f - out[3];
        if (offscreen) {
            const float remain = opt.background_brightness * nalpha;
            out[0] += remain;
            out[1] += remain;
            out[2] += remain;
        } else {
            out[0] += rgbx_init[0] / 255.f * nalpha;
            out[1] += rgbx_init[1] / 255.f * nalpha;
            out[2] += rgbx_init[2] / 255.f * nalpha;
        }

        // Output pixel color
        uint8_t rgbx[4] = {
            uint8_t(out[0] * 255),
            uint8_t(out[1] * 255),
            uint8_t(out[2] * 255),
            255
        };
        surf2Dwrite(
            *reinterpret_cast<uint32_t*>(rgbx),
            surf_obj,
            x * (int)sizeof(uint32_t),
            y,
            cudaBoundaryModeZero); // squelches out-of-bound writes
    } else {
        // write float colors into delta tracking context
        surf2Dwrite(
            *reinterpret_cast<float4*>(out),
            ctx.surface[CUR_RGBA],
            x * (int)sizeof(float4),
            y,
            cudaBoundaryModeZero); // squelches out-of-bound writes
        // write depth
        surf2Dwrite(
            depth,
            ctx.surface[PREV_MU_D],
            x * (int)sizeof(float4) + CTX_CUR_D * (int)sizeof(float),
            y,
            cudaBoundaryModeZero); // squelches out-of-bound writes
    }
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

#define READ_CTX(buf, sid, x, y)        \
    surf2Dread(                         \
        reinterpret_cast<float4*>(buf), \
        ctx.surface[sid],               \
        x * (int)sizeof(float4),        \
        y,                              \
        cudaBoundaryModeZero)

#define WRITE_CTX(buf, sid, x, y)       \
    surf2Dwrite(                        \
        *reinterpret_cast<float4*>(buf),\
        ctx.surface[sid],               \
        x * (int)sizeof(float4),        \
        y,                              \
        cudaBoundaryModeZero)

__global__ void temporal_accumulate(
        RenderContext ctx,
        RenderOptions opt,
        CameraSpec cam) {
    CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
    const int x = idx % cam.width, y = idx / cam.width;

    // get pixel information
    float rgba[4];
    READ_CTX(rgba, CUR_RGBA, x, y);
    float mu_d[4];
    READ_CTX(mu_d, PREV_MU_D, x, y);

    // special case: no history
    float mu1 = luminance(rgba);
    float mu2 = mu1 * mu1;
    if (!ctx.has_history) {
        mu_d[CTX_MU1] = mu1;
        mu_d[CTX_MU2] = mu2;
        WRITE_CTX(mu_d, PREV_MU_D, x, y);
        return;
    }

    // reprojection
    float dir[3];
    float cen[3];
    screen2worlddir(x, y, cam, dir, cen);
    float& cur_d = mu_d[CTX_CUR_D];
    float& prev_d = mu_d[CTX_PREV_D];
    float world_pos[3];
    for (int i = 0; i < 3; i++) {
        world_pos[i] = (cen[i] + cur_d * dir[i]);
    }
    float prev_x, prev_y;
    world2screen(ctx.prev_cam, world_pos, prev_x, prev_y);
    //if (cur_d < 1e9) {
    //    printf("%f: %f, %f ## %d, %d\n", cur_d, prev_x, prev_y, x, y);
    //}

    // check
    float alpha = opt.alpha;
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
    READ_CTX(prev_rgba, PREV_RGBA, prev_ix, prev_iy);
    if (opt.clamp && alpha < 1.0f) {
        float mean[4] = {};
        float sigma[4] = {};
        float buf[4] = {};

        int support = opt.clamp_support;
        int cnt = 0;
        for (int xx = x - support; xx <= x + support; xx++) {
            if (xx < 0 || xx >= cam.width) continue;
            for (int yy = y - support; yy <= y + support; yy++) {
                if (yy < 0 || yy >= cam.height) continue;
                cnt++;
                READ_CTX(buf, CUR_RGBA, xx, yy);
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
            sigma[i] = sqrtf(mean[i] * mean[i] - sigma[i]);

            prev_rgba[i] = fminf(
                fmaxf(prev_rgba[i], mean[i] - opt.clamp_k * sigma[i]),
                mean[i] + opt.clamp_k * sigma[i]
            );
        }
    }

    // alpha blend
    const float nalpha = 1.0f - alpha;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        rgba[i] = rgba[i] * alpha + prev_rgba[i] * nalpha;
    }
    cur_d = cur_d * alpha + prev_d * nalpha;
    
    // write rgba
    WRITE_CTX(rgba, CUR_RGBA, x, y);
    // write moments
    WRITE_CTX(mu_d, PREV_MU_D, x, y);
}


__global__ void wavelet_filter(
        RenderContext ctx,
        RenderOptions opt,
        int level) {
    const int& width = ctx.prev_cam.width;
    const int& height = ctx.prev_cam.height;
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x = idx % width, y = idx / width;

    constexpr static float epsilon = 1e-5f;
    constexpr static float h[25] = {
        1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
        1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
        3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
        1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
        1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };
    constexpr static float gaussianKernel[9] = {
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 };

    // locate
    // const int x = n % width;
    // const int y = n / width;
    //scalar_t* pImage_ptr = image + n * 3;
    //scalar_t* pVariance_ptr = variance + n;
    //const scalar_t* pPosition_ptr = position + n * 3;
    //const scalar_t* pDensity_ptr = density + n;

    //const float pLuminance = pImage_ptr[0] * 0.2126f + pImage_ptr[1] * 0.7152f + pImage_ptr[2] * 0.0722f;
    //const float pVariance = pVariance_ptr[0];
    //const float pPosition_x = pPosition_ptr[0];
    //const float pPosition_y = pPosition_ptr[1];
    //const float pPosition_z = pPosition_ptr[2];
    //const float pDensity = pDensity_ptr[0];

    //// filter variance
    //int delta_locs[9] = {
    //    -width - 1, -width, -width + 1,
    //    -1, 0, 1,
    //    width - 1, width, width + 1,
    //};
    //float gaussian_sum = 0.0f;
    //float gaussian_sumw = 0.0f;
    //for (int i = 0; i < 9; i++) {
    //    int loc = (int)n + delta_locs[i];
    //    if (loc < 0 || loc >= N) continue;
    //    gaussian_sum += gaussianKernel[i] * variance[loc];
    //    gaussian_sumw += gaussianKernel[i];
    //}
    //float gaussian_v = gaussian_sumw > epsilon ? gaussian_sum / gaussian_sumw
    //    : 0;
    //const int x_step = 1 << level;
    //const int y_step = x_step * width;
    //const int northwest = (int)n - support * (x_step + y_step);
    //float r = 0.0f;
    //float g = 0.0f;
    //float b = 0.0f;
    //float v = 0.0f;
    //float weights = 0.0f;
    //float weight_squares = 0.0f;

    //for (int offsety = -support; offsety <= support; offsety++)
    //{
    //    int loc = northwest + (offsety - support) * y_step;
    //    for (int offsetx = -support; offsetx <= support; offsetx++, loc += x_step) {
    //        if (loc < 0 || loc >= N)
    //            continue;

    //        // locate
    //        const scalar_t* qImage_ptr = image + loc * 3;
    //        const scalar_t* qVariance_ptr = variance + loc;
    //        const scalar_t* qPosition_ptr = position + loc * 3;
    //        const scalar_t* qDensity_ptr = density + loc;

    //        const float qLuminance = qImage_ptr[0] * 0.2126f + qImage_ptr[1] * 0.7152f + qImage_ptr[2] * 0.0722f;
    //        const float qVariance = qVariance_ptr[0];
    //        const float qPosition_x = qPosition_ptr[0];
    //        const float qPosition_y = qPosition_ptr[1];
    //        const float qPosition_z = qPosition_ptr[2];
    //        const float qDensity = qDensity_ptr[0];

    //        float t_x = pPosition_x - qPosition_x;
    //        float t_y = pPosition_y - qPosition_y;
    //        float t_z = pPosition_z - qPosition_z;
    //        float dist_p = t_x * t_x + t_y * t_y + t_z * t_z;
    //        float wp = fminf(__expf(-dist_p / (kp + epsilon)), 1.0f);

    //        float dist_d = fabsf(pDensity - qDensity);
    //        float wd = fminf(__expf(-dist_d / (kd + epsilon)), 1.0f);

    //        float dist_l = fabsf(pLuminance - qLuminance);
    //        float wl = fminf(__expf(-dist_l / (kl * sqrtf(gaussian_v) + epsilon)), 1.0f);

    //        float w = wp * wd * wl;
    //        float weight = h[5 * (offsety + support) + offsetx + support] * w;

    //        float weight_square = weight * weight;
    //        weights += weight;
    //        weight_squares += weight_square;
    //        r += weight * qImage_ptr[0];
    //        g += weight * qImage_ptr[1];
    //        b += weight * qImage_ptr[2];
    //        v += weight_square * qVariance;
    //    }
    //}

    //if (weights > epsilon)
    //{
    //    pImage_ptr[0] = clamp(r / weights, 0.0f, 10.0f);
    //    pImage_ptr[1] = clamp(g / weights, 0.0f, 10.0f);
    //    pImage_ptr[2] = clamp(b / weights, 0.0f, 10.0f);
    //    pVariance_ptr[0] = fminf(fmaxf(v / (weights * weights), 0.0f), 10.0f);
    //}

    if (level > 0) return;
    // save rgb to prev frame
    float rgba[4];
    surf2Dread(
        reinterpret_cast<float4*>(rgba),
        ctx.surface[CUR_RGBA],
        x * (int)sizeof(float4),
        y,
        cudaBoundaryModeZero);
    surf2Dwrite(
        *reinterpret_cast<float4*>(rgba),
        ctx.surface[PREV_RGBA],
        x * (int)sizeof(float4),
        y,
        cudaBoundaryModeZero);
}

__global__ void resultFromContext(
        cudaSurfaceObject_t surf_obj,
        cudaSurfaceObject_t surf_obj_depth,
        RenderContext ctx,
        RenderOptions opt,
        bool offscreen) {
    const int& width = ctx.prev_cam.width;
    const int& height = ctx.prev_cam.height;
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x = idx % width, y = idx / width;

    uint8_t rgbx_init[4];
    if (!offscreen) {
        // Read existing values for compositing (with meshes)
        surf2Dread(
            reinterpret_cast<uint32_t*>(rgbx_init), 
            surf_obj, 
            x * (int)sizeof(uint32_t),
            y, 
            cudaBoundaryModeZero);
    }

    // Compositing with existing color
    float out[4];
    surf2Dread(
        reinterpret_cast<float4*>(out),
        ctx.surface[CUR_RGBA], 
        x * (int)sizeof(float4),
        y, 
        cudaBoundaryModeZero);
    const float nalpha = 1.f - out[3];
    if (offscreen) {
        const float remain = opt.background_brightness * nalpha;
        out[0] += remain;
        out[1] += remain;
        out[2] += remain;
    }
    else {
        out[0] += rgbx_init[0] / 255.f * nalpha;
        out[1] += rgbx_init[1] / 255.f * nalpha;
        out[2] += rgbx_init[2] / 255.f * nalpha;
    }

    // Output pixel color
    uint8_t rgbx[4] = {
        uint8_t(out[0] * 255),
        uint8_t(out[1] * 255),
        uint8_t(out[2] * 255),
        255
    };
    surf2Dwrite(
        *reinterpret_cast<uint32_t*>(rgbx),
        surf_obj,
        x * (int)sizeof(uint32_t),
        y,
        cudaBoundaryModeZero); // squelches out-of-bound writes
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
    if (options.delta_tracking) {
        ctx.createSurfaceObjects();
    }

    // less threads is weirdly faster for me than 1024
    // Not sure if this scales to a good GPU
    const int N_CUDA_THREADS = 512;
    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, N_CUDA_THREADS);
    device::render_kernel<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            surf_obj_depth,
            cam,
            tree,
            options,
            probe_coeffs,
            ctx,
            offscreen
    );
    if (options.delta_tracking) {
        // ===== denoise =====
        // temporal 
        device::temporal_accumulate<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            ctx,
            options,
            cam
        );
        // record camera
        if (!ctx.has_history) {
            ctx.prev_cam.width = cam.width;
            ctx.prev_cam.height = cam.height;
            ctx.prev_cam.fx = cam.fx;
            ctx.prev_cam.fy = cam.fy;
        }
        cuda(Memcpy(ctx.prev_cam.transform, cam.device.transform,
            12 * sizeof(float), cudaMemcpyDeviceToDevice
        ));
        
        // spatial
        for (int level = 0; level < options.filter_iters; level++) {
            device::wavelet_filter<<<blocks, N_CUDA_THREADS, 0, stream>>>(
                ctx,
                options,
                level
            );
        }

        // convert float rgb image to uint32_t rgb image
        device::resultFromContext<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            surf_obj_depth,
            ctx,
            options,
            offscreen
        );

        // update context
        ctx.rng.advance();
        if (!ctx.has_history) {
            ctx.has_history = true;
        }
    }

    if (options.enable_probe) {
        cudaFree(probe_coeffs);
    }
}
}  // namespace volrend
