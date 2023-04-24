#pragma once
#include <cuda_runtime.h>

#include <memory>

#include "pcg32.h"
#include "volrend/cuda/common.cuh"

namespace volrend {

struct RenderContext {
    // rng
    pcg32 rng = pcg32(20230418);

    // pointer to extern cuda array
    cudaArray_t image_arr = nullptr;
    cudaArray_t depth_arr = nullptr;

    // surface object
    cudaSurfaceObject_t surf_obj       = 0;
    cudaSurfaceObject_t surf_obj_depth = 0;

    // auxiliary buffer
    constexpr static const int CHANNELS   = 8;
    float*                     aux_buffer = nullptr;

    bool offscreen = false;

public:
    RenderContext()          = default;
    virtual ~RenderContext() = default;

    void freeResource() {
        if (aux_buffer) {
            cudaFree(aux_buffer);
            aux_buffer = nullptr;
        }

        if (surf_obj) {
            cudaDestroySurfaceObject(surf_obj);
        }
        if (surf_obj_depth) {
            cudaDestroySurfaceObject(surf_obj_depth);
        }
        image_arr = depth_arr = nullptr;
    }

    void update(
        cudaArray_t image_arr,
        cudaArray_t depth_arr,
        const int   width,
        const int   height) {

        freeResource();

        this->image_arr = image_arr;
        this->depth_arr = depth_arr;
        createSurface();

        cudaMalloc(&aux_buffer, sizeof(float) * CHANNELS * width * height);
    }

private:
    void createSurface() {
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = image_arr;
        cudaCreateSurfaceObject(&surf_obj, &res_desc);

        if (!offscreen) {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType         = cudaResourceTypeArray;
            res_desc.res.array.array = depth_arr;
            cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
        }
    }
};

}  // namespace volrend