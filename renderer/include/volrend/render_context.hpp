#pragma once
#include <cuda_runtime.h>

#include <memory>
#include <cstring>

#include "pcg32.h"
#include "volrend/cuda/common.cuh"

#define DEBUG_TIME_RECORD

namespace volrend {

struct RenderContext {
    // rng
    pcg32 rng = pcg32(20230418);

    // pointer to extern cuda array
    cudaArray_t image_arr = nullptr;
    cudaArray_t depth_arr = nullptr;

    // auxiliary buffer
    constexpr static const int CHANNELS   = 8;
    float*                     aux_buffer = nullptr;

    // noisy image
    cudaArray_t noisy_image_arr = nullptr;

    // surface & texture objects
    cudaSurfaceObject_t surf_obj       = 0;
    cudaSurfaceObject_t surf_obj_depth = 0;
    cudaSurfaceObject_t noisy_surf_obj = 0;
    cudaTextureObject_t noisy_tex_obj  = 0;

    bool offscreen = false;

public:
    RenderContext()          = default;
    virtual ~RenderContext() = default;

    void freeResource() {
        // Destroy surfaces & textures
        if (surf_obj) {
            cudaDestroySurfaceObject(surf_obj);
        }
        if (surf_obj_depth) {
            cudaDestroySurfaceObject(surf_obj_depth);
        }
        if (noisy_surf_obj) {
            cudaDestroySurfaceObject(noisy_surf_obj);
        }
        if (noisy_tex_obj) {
            cudaDestroySurfaceObject(noisy_tex_obj);
        }

        // Destroy noisy image
        if (noisy_image_arr) {
            cudaFreeArray(noisy_image_arr);
        }
        
        // Destroy aux buffer
        if (aux_buffer) {
            cudaFree(aux_buffer);
            aux_buffer = nullptr;
        }

        image_arr = depth_arr = nullptr;
    }

    void update(
        cudaArray_t image_arr,
        cudaArray_t depth_arr,
        const int   width,
        const int   height) {

        freeResource();

        // Set cuda array pointers
        this->image_arr = image_arr;
        this->depth_arr = depth_arr;

        // Create aux buffer
        cudaMalloc(&aux_buffer, sizeof(float) * CHANNELS * width * height);

        // Create noisy image
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&noisy_image_arr, &channelDesc, width, height);

        // Create surfaces & textures
        createSurface();
    }

    void createSurface() {
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = image_arr;
        cudaCreateSurfaceObject(&surf_obj, &res_desc);

        if (!offscreen) {
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType         = cudaResourceTypeArray;
            res_desc.res.array.array = depth_arr;
            cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
        }

        // noisy image
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = noisy_image_arr;
        cudaCreateSurfaceObject(&noisy_surf_obj, &res_desc);

        cudaTextureDesc tex_desc  = {};
        tex_desc.normalizedCoords = false;
        tex_desc.filterMode       = cudaFilterModePoint;
        tex_desc.addressMode[0]   = cudaAddressModeWrap;
        tex_desc.addressMode[1]   = cudaAddressModeWrap;
        tex_desc.readMode         = cudaReadModeElementType;
        cudaCreateTextureObject(&noisy_tex_obj, &res_desc, &tex_desc, nullptr);
    }

#ifdef DEBUG_TIME_RECORD
    struct Timer final {
        enum {
            T_RENDER,
            T_TORCH,
            T_FILTER,
            T_CNT,
        };

        cudaStream_t stream       = nullptr;
        cudaEvent_t  start[T_CNT] = {};
        cudaEvent_t  stop[T_CNT]  = {};
        float        sum[T_CNT]   = {};

        int cnt = 0;

    public:
        Timer() {
            for (int i = 0; i < T_CNT; i++) {
                cudaEventCreate(&start[i]);
                cudaEventCreate(&stop[i]);
            }
        }

        ~Timer() {
            for (int i = 0; i < T_CNT; i++) {
                cudaEventDestroy(start[i]);
                cudaEventDestroy(stop[i]);
            }
        }

        Timer(const Timer& rhs) = delete;

        void reset(cudaStream_t stream) {
            this->stream = stream;
            cnt = 0;
            for (int i = 0; i < T_CNT; i++) {
                sum[i] = 0;
            }
        }

        void start_record(int idx) {
            cudaEventRecord(start[idx], stream);
        }

        void stop_record(int idx) {
            cudaEventRecord(stop[idx], stream);
        }

        void render_start() { start_record(T_RENDER); }
        void torch_start()  { start_record(T_TORCH); }
        void filter_start() { start_record(T_FILTER); }

        void render_stop()  { stop_record(T_RENDER); }
        void torch_stop()   { stop_record(T_TORCH); }
        void filter_stop()  { stop_record(T_FILTER); }

        void record(bool denoise) {
            cudaEventSynchronize(stop[denoise ? T_FILTER : T_RENDER]);

            cnt++;
            for (int i = 0; i < T_CNT; i++) {
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start[i], stop[i]);
                sum[i] += milliseconds;
            }
        }

        void report() {
            float all = 0;
            float t   = sum[T_RENDER] / cnt;
            printf("render: %.10f ms per frame\n", t);
            all += t;

            t = sum[T_TORCH] / cnt;
            printf("torch:  %.10f ms per frame\n", t);
            all += t;

            t = sum[T_FILTER] / cnt;
            printf("filter: %.10f ms per frame\n", t);
            all += t;

            printf("all:    %.10f ms per frame\n", all);
        }
    };

    static Timer& timer() {
        static Timer instance;
        return instance;
    }
#endif
};

}  // namespace volrend