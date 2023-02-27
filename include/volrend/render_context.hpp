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
    PREV_RGBA = 0,
    CUR_RGBA,
    PREV_D,
    CUR_D,

    CTX_ALL_COUNT,
};

// Rendering context for delta tracking 
struct RenderContext {
    bool has_history = false;
    int spp = 0; // samples for current pose
    
    // rng
    pcg32 rng = pcg32(20230226);

    // image buffers, device array
    float* data[CTX_ALL_COUNT] = {};

    // prev camera
    float* VOLREND_RESTRICT prev_transform_device = nullptr;
    float* prev_transform_host = nullptr;

public:
    RenderContext() = default;
    virtual ~RenderContext() = default;

    void clearHistory() {
        has_history = false;
        spp = 0;
    }

    void freeResource() {
        clearHistory();
        // free device
        if (prev_transform_device != nullptr) {
            cuda(Free((void*)prev_transform_device));
            prev_transform_device = nullptr;
        }
        for (auto& arr : data) {
            cuda(Free(arr));
        }
        // free host
        delete prev_transform_host;
        prev_transform_host = nullptr;
    }

    void resize(int width, int height) {
        freeResource();

        // host buffer
        prev_transform_host = new float[12];
        // previous camera
        cuda(Malloc((void**)&prev_transform_device, 12 * sizeof(float)));
        // frames
        size_t&& d_size = sizeof(float) * width * height;
        size_t&& rgba_size = d_size * 4;
        cuda(Malloc(&data[PREV_RGBA], rgba_size));
        cuda(Malloc(&data[CUR_RGBA], rgba_size));
        cuda(Malloc(&data[PREV_D], d_size));
        cuda(Malloc(&data[CUR_D], d_size));
    }
};

} // namespace volrend