#pragma once

#include <array>
#include <cuda_runtime.h>

#include "volrend/pcg32.h"
#include "volrend/cuda/common.cuh"
#include "volrend/internal/data_spec.hpp"

namespace volrend {

// Enum for RenderContext
enum {
    // Previous frame
    PREV_RGBA = 0, // A is for foreground / background
    PREV_MU_D, // mu1, mu2, prev_d, cur_d
    CUR_RGBA,

    CTX_ALL_COUNT,

    // Index
    CTX_R = 0,
    CTX_G,
    CTX_B,
    CTX_A,
    CTX_MU1 = 0,
    CTX_MU2,
    CTX_PREV_D,
    CTX_CUR_D,
};

// Rendering context for delta tracking 
struct RenderContext {
    bool has_history = false;
    
    // rng
    pcg32 rng = pcg32(20230226);

    // image buffers
    cudaArray_t data[CTX_ALL_COUNT] = {};
    cudaSurfaceObject_t surface[CTX_ALL_COUNT] = {};

    // prev camera
    internal::CameraSpec prev_cam{};

public:
    RenderContext() = default;
    virtual ~RenderContext() = default;

    void clearHistory() {
        has_history = false;
    }

    void freeResource() {
        clearHistory();
        if (prev_cam.transform != nullptr) {
            cuda(Free((void*)prev_cam.transform));
            prev_cam.transform = nullptr;
        }
        for (auto& arr : data) {
            cuda(FreeArray(arr));
        }
    }

    void resize(int width, int height) {
        freeResource();

        // alloc previous camera
        cuda(Malloc((void**)&prev_cam.transform, 12 * sizeof(prev_cam.transform[0])));

        // prev frame
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        cuda(MallocArray(&data[PREV_RGBA], &desc, width, height));
        cuda(MallocArray(&data[PREV_MU_D], &desc, width, height));
        cuda(MallocArray(&data[CUR_RGBA], &desc, width, height));
    }

    void createSurfaceObjects() {
        for (int i = 0; i < CTX_ALL_COUNT; i++) {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = data[i];
            cudaCreateSurfaceObject(&surface[i], &res_desc);
        }
    }
};

} // namespace volrend