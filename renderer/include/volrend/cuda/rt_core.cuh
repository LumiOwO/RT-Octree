#pragma once
#include "volrend/internal/n3tree_query.hpp"
#include <cmath>
#include <cuda_fp16.h>
#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/render_options.hpp"
#include "volrend/cuda/common.cuh"
#include "volrend/internal/data_spec.hpp"
#include "volrend/internal/lumisphere.hpp"
#include "volrend/internal/morton.hpp"
#include <cstdio>
#include <cfloat>

namespace volrend {
namespace device {
namespace {

template<typename scalar_t>
__device__ __inline__ void _dda_world(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax,
        const float* __restrict__ render_bbox) {
    scalar_t t1, t2;
    *tmin = 0.0;
    *tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = (render_bbox[i] + 1e-6 - cen[i]) * _invdir[i];
        t2 = (render_bbox[i + 3] - 1e-6 - cen[i]) * _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

template<typename scalar_t>
__device__ __inline__ scalar_t _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir) {
    scalar_t t1, t2;
    scalar_t tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template <int SPP>
__device__ __forceinline__ void sample_dst_recursive(
    float* __restrict__ dst,
    pcg32& rng) {

    sample_dst_recursive<SPP - 1>(dst, rng);

    float t = -__logf(1.0f - rng.next_float());
    if (t <= dst[0]) {
        for (int i = SPP - 1; i > 0; i--) {
            dst[i] = dst[i - 1];
        }
        dst[0] = t;
    } else {
        int i = SPP - 1;
        while (dst[i - 1] > t) {
            dst[i] = dst[i - 1];
            i--;
        }
        dst[i] = t;
    }
}

template <>
__device__ __forceinline__ void sample_dst_recursive<1>(
    float* __restrict__ dst,
    pcg32& rng) {

    dst[0] = -__logf(1.0f - rng.next_float());
}

template <>
__device__ __forceinline__ void sample_dst_recursive<2>(
    float* __restrict__ dst,
    pcg32& rng) {

    dst[0] = -__logf(1.0f - rng.next_float());

    float t = -__logf(1.0f - rng.next_float());
    if (t >= dst[0]) {
        dst[1] = t;
    } else {
        dst[1] = dst[0];
        dst[0] = t;
    }
}

template <>
__device__ __forceinline__ void sample_dst_recursive<3>(
    float* __restrict__ dst,
    pcg32& rng) {

    dst[0]  = -__logf(1.0f - rng.next_float());

    float t = -__logf(1.0f - rng.next_float());
    if (t >= dst[0]) {
        dst[1] = t;
    } else {
        dst[1] = dst[0];
        dst[0] = t;
    }

    t = -__logf(1.0f - rng.next_float());
    if (t >= dst[1]) {
        dst[2] = t;
    } else if (t >= dst[0]) {
        dst[2] = dst[1];
        dst[1] = t;
    } else {
        dst[2] = dst[1];
        dst[1] = dst[0];
        dst[0] = t;
    }
}

template <>
__device__ __forceinline__ void sample_dst_recursive<4>(
    float* __restrict__ dst,
    pcg32& rng) {

    dst[0]  = -__logf(1.0f - rng.next_float());

    float t = -__logf(1.0f - rng.next_float());
    if (t >= dst[0]) {
        dst[1] = t;
    } else {
        dst[1] = dst[0];
        dst[0] = t;
    }

    t = -__logf(1.0f - rng.next_float());
    if (t >= dst[1]) {
        dst[2] = t;
    } else if (t >= dst[0]) {
        dst[2] = dst[1];
        dst[1] = t;
    } else {
        dst[2] = dst[1];
        dst[1] = dst[0];
        dst[0] = t;
    }

    t = -__logf(1.0f - rng.next_float());
    if (t >= dst[2]) {
        dst[3] = t;
    } else if (t >= dst[1]) {
        dst[3] = dst[2];
        dst[2] = t;
    } else if (t >= dst[0]) {
        dst[3] = dst[2];
        dst[2] = dst[1];
        dst[1] = t;
    } else {
        dst[3] = dst[2];
        dst[2] = dst[1];
        dst[1] = dst[0];
        dst[0] = t;
    }
}

template <int SPP>
__device__ __forceinline__ void sample_dst(
    float* __restrict__ dst,
    pcg32& rng) {
    sample_dst_recursive<SPP>(dst, rng);
    dst[SPP] = FLT_MAX;
}

template <typename scalar_t, int SPP>
__device__ __inline__ void trace_ray(
    const internal::TreeSpec& __restrict__ tree,
    scalar_t* __restrict__ dir,
    const scalar_t* __restrict__ vdir,
    const scalar_t* __restrict__ cen,
    const RenderOptions& __restrict__ opt,
    float tmax_bg,
    scalar_t* __restrict__ out,
    pcg32& __restrict__ rng) {

    const float delta_scale = _get_delta_scale(
            tree.scale, /*modifies*/ dir);
    tmax_bg /= delta_scale;

    scalar_t tmin, tmax;
    scalar_t invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    _dda_world(cen, invdir, &tmin, &tmax, opt.render_bbox);
    tmax = min(tmax, tmax_bg);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    }

    scalar_t    pos[3], tmp;
    const half* tree_val;
    scalar_t t = tmin;
    scalar_t cube_sz;

    // init random samples
    float src = 0;
    float dst[SPP + 1];
    sample_dst<SPP>(dst, rng);

    const half* tree_vals[SPP];
    float       cnts[SPP] = {};
    uint32_t    spp       = 0;
    uint32_t    sh_nums   = 0;

    // find tree_val for samples
    //int step = 0;
    while (t < tmax) {
        //step++;
        pos[0] = cen[0] + t * dir[0];
        pos[1] = cen[1] + t * dir[1];
        pos[2] = cen[2] + t * dir[2];

        internal::query_single_from_root(tree, pos, &tree_val, &cube_sz);

        const scalar_t t_subcube = _dda_unit(pos, invdir) / cube_sz;
        const scalar_t delta_t   = t_subcube + opt.step_size;
        const float    sigma     = __half2float(tree_val[tree.data_dim - 1]);
        if (sigma > opt.sigma_thresh) {
            const float delta = delta_t * delta_scale * sigma;
            if (src + delta >= dst[spp]) {
                float& cnt         = cnts[sh_nums];
                tree_vals[sh_nums] = tree_val;
                ++sh_nums;
                do {
                    ++cnt;
                    ++spp;
                } while (src + delta >= dst[spp]);

                if (spp == SPP) {
                    break;
                }
            }
            src += delta;
        }
        t += delta_t;
    }

    if (sh_nums == 0) {
        return;
    }

    // accumulate color
    scalar_t basis_fn[VOLREND_GLOBAL_BASIS_MAX];
    internal::maybe_precalc_basis(tree, vdir, basis_fn);
    for (int i = 0; i < opt.basis_minmax[0]; ++i) {
        basis_fn[i] = 0.f;
    }
    for (int i = opt.basis_minmax[1] + 1; i < VOLREND_GLOBAL_BASIS_MAX; ++i) {
        basis_fn[i] = 0.f;
    }

    for (uint32_t i = 0; i < sh_nums; i++) {
        if (tree.data_format.basis_dim >= 0) {
            int         off      = 0;
            const half* tree_val = tree_vals[i];
#define MUL_BASIS_I(t) basis_fn[t] * __half2float(tree_val[off + t])
#pragma unroll 3
            for (int t = 0; t < 3; ++t) {
                tmp = basis_fn[0] * __half2float(tree_val[off]);
                switch (tree.data_format.basis_dim) {
                    case 25:
                        tmp += MUL_BASIS_I(16) + MUL_BASIS_I(17) +
                               MUL_BASIS_I(18) + MUL_BASIS_I(19) +
                               MUL_BASIS_I(20) + MUL_BASIS_I(21) +
                               MUL_BASIS_I(22) + MUL_BASIS_I(23) +
                               MUL_BASIS_I(24);
                    case 16:
                        tmp += MUL_BASIS_I(9) + MUL_BASIS_I(10) +
                               MUL_BASIS_I(11) + MUL_BASIS_I(12) +
                               MUL_BASIS_I(13) + MUL_BASIS_I(14) +
                               MUL_BASIS_I(15);

                    case 9:
                        tmp += MUL_BASIS_I(4) + MUL_BASIS_I(5) +
                               MUL_BASIS_I(6) + MUL_BASIS_I(7) + MUL_BASIS_I(8);

                    case 4:
                        tmp += MUL_BASIS_I(1) + MUL_BASIS_I(2) + MUL_BASIS_I(3);
                }
                out[t] += cnts[i] / (1.f + __expf(-tmp));
                off += tree.data_format.basis_dim;
            }
#undef MUL_BASIS_I
        } else {
            for (int j = 0; j < 3; ++j) {
                out[j] += __half2float(tree_val[j]) * cnts[i];
            }
        }

        out[3] += cnts[i];
    }

    constexpr float INV_SPP = 1.0f / SPP;
    out[0] *= INV_SPP;
    out[1] *= INV_SPP;
    out[2] *= INV_SPP;
    out[3] *= INV_SPP;
}
}  // namespace
}  // namespace device
}  // namespace volrend
