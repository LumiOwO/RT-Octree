#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <ATen/cuda/Atomic.cuh>

#include "filtering.h"

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) \
    TORCH_CHECK(        \
        x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x)                           \
    TORCH_CHECK(                                       \
        x.scalar_type() == at::ScalarType::Float ||    \
            x.scalar_type() == at::ScalarType::Half || \
            x.scalar_type() == at::ScalarType::Double, \
        #x " must be a floating tensor")

// Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))

#define CUDA_GET_THREAD_ID(tid, Q)                             \
    const int tid = IMAD(blockIdx.x, blockDim.x, threadIdx.x); \
    if (tid >= Q) return
#define N_BLOCKS_NEEDED(Q, N_CUDA_THREADS) ((Q - 1) / N_CUDA_THREADS + 1)

// macros for magic numbers
#define IMG_CHANNELS        4
#define SAVED_INPUTS_CNT    3
#define SAVED_PER_LEVEL_CNT 3

namespace denoiser {

namespace kernel {

template <typename OutputType, bool OVERRIDE>
__device__ __forceinline__ void write_to_img_out(
    OutputType    img_out,
    const float4& rgba,
    const int     idx,
    const int     ix,
    const int     iy);

template <>
__device__ __forceinline__ void write_to_img_out<float*, true>(
    float*        img_out,
    const float4& rgba,
    const int     idx,
    const int     ix,
    const int     iy) {

    img_out += (idx << 2);
    img_out[0] = rgba.x;
    img_out[1] = rgba.y;
    img_out[2] = rgba.z;
    img_out[3] = 1.0f;
}

template <>
__device__ __forceinline__ void write_to_img_out<float*, false>(
    float*        img_out,
    const float4& rgba,
    const int     idx,
    const int     ix,
    const int     iy) {

    img_out += (idx << 2);
    img_out[0] += rgba.x;
    img_out[1] += rgba.y;
    img_out[2] += rgba.z;
}

template <>
__device__ __forceinline__ void write_to_img_out<cudaSurfaceObject_t, true>(
    cudaSurfaceObject_t img_out,
    const float4&       rgba,
    const int           idx,
    const int           ix,
    const int           iy) {

    surf2Dwrite(
        rgba, img_out, ix * (int)sizeof(float4), iy, cudaBoundaryModeZero);
}

template <>
__device__ __forceinline__ void write_to_img_out<cudaSurfaceObject_t, false>(
    cudaSurfaceObject_t img_out,
    const float4&       rgba,
    const int           idx,
    const int           ix,
    const int           iy) {

    float4 out = {};
    surf2Dread(
        &out, img_out, ix * (int)sizeof(float4), iy, cudaBoundaryModeZero);

    out.x += rgba.x;
    out.y += rgba.y;
    out.z += rgba.z;

    surf2Dwrite(
        out, img_out, ix * (int)sizeof(float4), iy, cudaBoundaryModeZero);
}

template <typename OutputType, int BLOCK_H, int BLOCK_W, int SUPPORT>
__global__ void applying(
    const int           H,
    const int           W,
    cudaTextureObject_t img_in_tex,          // [H, W, 4]
    const float* __restrict__ weight_map,    // [H, W]
    const float* __restrict__ guidance_map,  // [H, W]
    OutputType img_out,                      // [H, W, 4]
    float* __restrict__ rgb_filtered,        // [H, W, 4]
    float* __restrict__ max_map,             // [H, W]
    float* __restrict__ inv_kernel_sum       // [H, W]
) {
    constexpr int SUPPORT2 = SUPPORT * 2;
    constexpr int TILE_H   = BLOCK_H + SUPPORT2;
    constexpr int TILE_W   = BLOCK_W + SUPPORT2;

    // locate
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);  // col
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);  // row
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // load tile elements
    __shared__ float4 rgba_tile[TILE_H][TILE_W];
    __shared__ float  kernel_tile[TILE_H][TILE_W];

#define __LOAD(tile_x, tile_y, image_x, image_y)                     \
    do {                                                             \
        int _tx = (tile_x);                                          \
        int _ty = (tile_y);                                          \
        int _ix = (image_x);                                         \
        int _iy = (image_y);                                         \
        if (_iy >= 0 && _iy < H && _ix >= 0 && _ix < W) {            \
            rgba_tile[_ty][_tx] =                                    \
                tex2D<float4>(img_in_tex, _ix + 0.5f, _iy + 0.5f);   \
            kernel_tile[_ty][_tx] = guidance_map[IMAD(_iy, W, _ix)]; \
        } else {                                                     \
            rgba_tile[_ty][_tx]   = float4{0, 0, 0, 0};              \
            kernel_tile[_ty][_tx] = -FLT_MAX;                        \
        }                                                            \
    } while (0)

    // left-up
    __LOAD(tx, ty, ix - SUPPORT, iy - SUPPORT);
    // right
    if (tx >= BLOCK_W - SUPPORT2) {
        __LOAD(tx + SUPPORT2, ty, ix + SUPPORT, iy - SUPPORT);
    }
    // down
    if (ty >= BLOCK_H - SUPPORT2) {
        __LOAD(tx, ty + SUPPORT2, ix - SUPPORT, iy + SUPPORT);
    }
    // right-down
    if (tx < SUPPORT2 && ty < SUPPORT2) {
        __LOAD(
            tx + BLOCK_W,
            ty + BLOCK_H,
            ix - SUPPORT + BLOCK_W,
            iy - SUPPORT + BLOCK_H);
    }

    __syncthreads();
#undef __LOAD

    // locate
    if (ix >= W || iy >= H) {
        return;
    }
    const int idx = IMAD(iy, W, ix);

    float max_val = -FLT_MAX;
    for (int dy = 0; dy <= SUPPORT2; dy++) {
        for (int dx = 0; dx <= SUPPORT2; dx++) {
            max_val = fmaxf(max_val, kernel_tile[ty + dy][tx + dx]);
        }
    }

    // apply
    float4 rgba       = {0.f, 0.f, 0.f, 1.f};
    float  kernel_sum = 0;
    for (int dy = 0; dy <= SUPPORT2; dy++) {
        int _ty = ty + dy;

        for (int dx = 0; dx <= SUPPORT2; dx++) {
            int _tx = tx + dx;

            // compute
            float k = __expf(kernel_tile[_ty][_tx] - max_val);
            kernel_sum += k;
            auto& t_rgb = rgba_tile[_ty][_tx];
            rgba.x += t_rgb.x * k;
            rgba.y += t_rgb.y * k;
            rgba.z += t_rgb.z * k;
        }
    }

    const float inv = 1.0f / kernel_sum;
    // save for backward
    if (rgb_filtered != nullptr) {
        rgb_filtered += idx * IMG_CHANNELS;
        max_map += idx;
        inv_kernel_sum += idx;

        max_map[0]        = max_val;
        inv_kernel_sum[0] = inv;
        rgb_filtered[0]   = rgba.x * inv;
        rgb_filtered[1]   = rgba.y * inv;
        rgb_filtered[2]   = rgba.z * inv;
    }

    const float w = weight_map[idx] * inv;
    rgba.x *= w;
    rgba.y *= w;
    rgba.z *= w;

    // accumulate
    // !!! IMPORTANT: only valid when all levels
    // !!!            are dispatched on the same stream!
    constexpr bool OVERRIDE = (SUPPORT == 1);
    write_to_img_out<OutputType, OVERRIDE>(img_out, rgba, idx, ix, iy);
}

template <typename scalar_t>
__global__ void grad_weight_accumulate(
    const int SIZE,
    const scalar_t* __restrict__ grad_output,   // [H, W, 4]
    const scalar_t* __restrict__ weight_map,    // [H, W]
    const scalar_t* __restrict__ rgb_filtered,  // [H, W, 4]
    scalar_t* grad_weight                       // [H, W]
) {
    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    grad_output  += idx * IMG_CHANNELS;
    weight_map   += idx;
    rgb_filtered += idx * IMG_CHANNELS;
    grad_weight  += idx;

    grad_weight[0] = grad_output[0] * rgb_filtered[0] +
                     grad_output[1] * rgb_filtered[1] +
                     grad_output[2] * rgb_filtered[2];
}

template <typename scalar_t>
__global__ void grad_guidance_accumulate(
    const int SIZE,
    const int H,
    const int W,
    const int support,
    const scalar_t* __restrict__ grad_output,     // [H, W, 4]
    const scalar_t* __restrict__ img_in,          // [H, W, 4]
    const scalar_t* __restrict__ weight_map,      // [H, W]
    const scalar_t* __restrict__ rgb_filtered,    // [H, W, 4]
    const scalar_t* __restrict__ guidance_map,    // [H, W]
    const scalar_t* __restrict__ max_map,         // [H, W]
    const scalar_t* __restrict__ inv_kernel_sum,  // [H, W]
    scalar_t* grad_guidance                       // [H, W]
) {
    // locate
    const int K      = 1 + (support << 1);
    const int K_SIZE = K * K;
    CUDA_GET_THREAD_ID(idx, SIZE * K_SIZE);
    const int pixel_idx   = idx / K_SIZE;
    const int kernel_idx  = idx % K_SIZE;

    const int lower_bound = pixel_idx - pixel_idx % (H * W);
    const int pixel_y     = (pixel_idx - lower_bound) / W;
    const int neighbor_y  = pixel_y - support + kernel_idx / K;
    if (neighbor_y < 0 || neighbor_y >= H) return;
    const int pixel_x    = pixel_idx % W;
    const int neighbor_x = pixel_x - support + kernel_idx % K;
    if (neighbor_x < 0 || neighbor_x >= W) return;

    const int neighbor_idx = lower_bound + neighbor_y * W + neighbor_x;
    // clang-format off
    grad_output    += pixel_idx * IMG_CHANNELS;
    img_in         += neighbor_idx * IMG_CHANNELS;
    weight_map     += pixel_idx;
    rgb_filtered   += pixel_idx * IMG_CHANNELS;
    guidance_map   += neighbor_idx;
    max_map        += pixel_idx;
    inv_kernel_sum += pixel_idx;
    grad_guidance  += neighbor_idx;
    // clang-format on

    // accumulate grad
    const scalar_t& k   = __expf(guidance_map[0] - max_map[0]) * inv_kernel_sum[0];
    scalar_t        res = 0;
    res += grad_output[0] * (img_in[0] - rgb_filtered[0]);
    res += grad_output[1] * (img_in[1] - rgb_filtered[1]);
    res += grad_output[2] * (img_in[2] - rgb_filtered[2]);
    res *= weight_map[0] * k;

    gpuAtomicAdd(&grad_guidance[0], res);
}

}  // namespace kernel

namespace host {

template <typename OutputType>
__host__ inline void kernel_apply(
    cudaStream_t        stream,
    const int           support,
    const int           H,
    const int           W,
    cudaTextureObject_t img_in_tex,     // [H, W, 4]
    const float*        weight_map,     // [H, W]
    const float*        guidance_map,   // [H, W]
    OutputType          img_out,        // [H, W, 4]
    float*              rgb_filtered,   // [H, W, 4]
    float*              max_map,        // [H, W]
    float*              inv_kernel_sum  // [H, W]
) {
    constexpr int BLOCK_H = 16;
    constexpr int BLOCK_W = 32;

    // Compute block & grid size
    dim3          block_size;
    block_size.x = BLOCK_W; /* col */
    block_size.y = BLOCK_H; /* row */
    block_size.z = 1;
    dim3 grid_size;
    grid_size.x = N_BLOCKS_NEEDED(W, BLOCK_W); /* col */
    grid_size.y = N_BLOCKS_NEEDED(H, BLOCK_H); /* row */
    grid_size.z = 1;

#define __ARGS_TEMP__                                                  \
    H, W, img_in_tex, weight_map, guidance_map, img_out, rgb_filtered, \
        max_map, inv_kernel_sum

    switch (support) {
        case 1:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 1>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        case 2:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 2>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        case 3:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 3>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        case 4:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 4>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        case 5:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 5>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        case 6:
            kernel::applying<OutputType, BLOCK_H, BLOCK_W, 6>
                <<<grid_size, block_size, 0, stream>>>(__ARGS_TEMP__);
            break;
        default:
            throw std::runtime_error(
                "Kernel size == " + std::to_string(support * 2 + 1) +
                " not supported.");
    }
#undef __ARGS_TEMP__
}

template <typename float_n>
__host__ cudaTextureObject_t create_texture_from_tensor(
    cudaArray_t&  cuArray,
    int           H,
    int           W) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float_n>();
    cudaMallocArray(&cuArray, &channelDesc, W, H);

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = cuArray;

    cudaTextureDesc texDesc  = {};
    texDesc.normalizedCoords = false;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.readMode         = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    return texObj;
}

template <typename OutputType>
__host__ inline void accumulate_one_level(
    cudaStream_t        stream,
    const int           level,
    const int           H,
    const int           W,
    const int           L,
    cudaTextureObject_t img_in_tex,    // [H, W, 4]
    torch::Tensor       weight_map,    // [H, W]
    torch::Tensor       guidance_map,  // [H, W]
    OutputType          img_out,       // [H, W, 4]

    // clang-format off
    torch::autograd::variable_list& save,
    const int batch_idx
    // clang-format on
) {
    // tensors
    bool need_save      = !save.empty();
    auto rgb_filtered   = torch::Tensor();
    auto max_map        = torch::Tensor();
    auto inv_kernel_sum = torch::Tensor();
    if (need_save) {
        int&& idx      = SAVED_INPUTS_CNT + level * SAVED_PER_LEVEL_CNT;
        rgb_filtered   = save[idx].index({batch_idx});
        max_map        = save[idx + 1].index({batch_idx});
        inv_kernel_sum = save[idx + 2].index({batch_idx});
    }

    // apply kernel
    kernel_apply<OutputType>(
        stream,
        level + 1,  // support
        H,
        W,
        img_in_tex,
        weight_map.data_ptr<float>(),
        guidance_map.data_ptr<float>(),
        img_out,
        (need_save ? rgb_filtered.data_ptr<float>() : nullptr),
        (need_save ? max_map.data_ptr<float>() : nullptr),
        (need_save ? inv_kernel_sum.data_ptr<float>() : nullptr));
}

template <typename OutputType>
__host__ inline void forward(
    cudaStream_t        stream,
    torch::Tensor       weight_map,    // [L, H, W]
    torch::Tensor       guidance_map,  // [L, H, W]
    cudaTextureObject_t img_in_tex,    // [H, W, 4]
    OutputType          img_out,       // [H, W, 4]
    // clang-format off
    torch::autograd::variable_list& save,
    const int batch_idx
    // clang-format on
) {
    const int L = guidance_map.size(0);
    const int H = guidance_map.size(1);
    const int W = guidance_map.size(2);

    for (int level = 0; level < L; level++) {
        accumulate_one_level<OutputType>(
            stream,
            level,
            H,
            W,
            L,
            img_in_tex,
            weight_map.index({level}),
            guidance_map.index({level}),
            img_out,
            save, 
            batch_idx);
    }
}

__host__ inline void grad_accumulate_one_level(
    cudaStream_t  stream,
    const int     level,
    const int     SIZE,
    const int     H,
    const int     W,
    const int     L,
    torch::Tensor grad_output,     // [H, W, 4]
    torch::Tensor img_in,          // [H, W, 4]
    torch::Tensor rgb_filtered,    // [H, W, 4]
    torch::Tensor max_map,         // [H, W]
    torch::Tensor inv_kernel_sum,  // [H, W]
    torch::Tensor weight_map,      // [H, W]
    torch::Tensor guidance_map,    // [H, W]
    torch::Tensor grad_weight,     // [H, W]
    torch::Tensor grad_guidance    // [H, W]
) {
    constexpr static const int N_THREADS = 512;

    // compute weight grads
    const int grad_weight_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
    // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_guidance.type(), "backward_grad_weight", ([&] {
                kernel::grad_weight_accumulate<scalar_t>
                    <<<grad_weight_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        grad_output.data_ptr<scalar_t>(),
                        weight_map.data_ptr<scalar_t>(),
                        rgb_filtered.data_ptr<scalar_t>(),
                        grad_weight.data_ptr<scalar_t>());
            }));
    // clang-format on

    // kernel size
    const int support = level + 1;
    const int K       = 1 + (support << 1);
    const int K_SIZE  = K * K;

    // compute kernel grads
    const int grad_guidance_blocks = N_BLOCKS_NEEDED(SIZE * K_SIZE, N_THREADS);
    // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_guidance.type(), "backward_grad_guidance", ([&] {
                kernel::grad_guidance_accumulate<scalar_t>
                    <<<grad_guidance_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        H,
                        W,
                        support,
                        grad_output.data_ptr<scalar_t>(),
                        img_in.data_ptr<scalar_t>(),
                        weight_map.data_ptr<scalar_t>(),
                        rgb_filtered.data_ptr<scalar_t>(),
                        guidance_map.data_ptr<scalar_t>(),
                        max_map.data_ptr<scalar_t>(),
                        inv_kernel_sum.data_ptr<scalar_t>(),
                        grad_guidance.data_ptr<scalar_t>());
            }));
    // clang-format on
}

__host__ inline void backward(
    cudaStream_t  stream,
    torch::Tensor grad_output,    // [H, W, 4]
    torch::Tensor img_in,         // [H, W, 4]
    torch::Tensor weight_map,     // [L, H, W]
    torch::Tensor guidance_map,   // [L, H, W]
    torch::Tensor grad_weight,    // [L, H, W]
    torch::Tensor grad_guidance,  // [L, H, W]
    // clang-format off
    torch::autograd::variable_list& saved,
    const int batch_idx
    // clang-format on
) {

    // get dimensions
    const int L    = guidance_map.size(0);
    const int H    = guidance_map.size(1);
    const int W    = guidance_map.size(2);
    const int SIZE = H * W;

    // accumulate grads for each level
    for (int level = 0; level < L; level++) {
        int&& saved_idx = SAVED_INPUTS_CNT + level * SAVED_PER_LEVEL_CNT;

        grad_accumulate_one_level(
            stream,
            level,
            SIZE,
            H,
            W,
            L,
            grad_output,
            img_in,
            saved[saved_idx].index({batch_idx}),      // rgb_filtered
            saved[saved_idx + 1].index({batch_idx}),  // max_map
            saved[saved_idx + 2].index({batch_idx}),  // inv_kernel_sum
            weight_map.index({level}),                // weight_map
            guidance_map.index({level}),              // guidance_map
            grad_weight.index({level}),               // grad_weight
            grad_guidance.index({level})              // grad_guidance
        );
    }
}

};  // namespace host

class Filtering : public torch::autograd::Function<Filtering> {
public:
    // forward
    static torch::Tensor forward(
        // clang-format off
        torch::autograd::AutogradContext* ctx,
        // clang-format on
        torch::Tensor weight_map,    // [B, L, H, W]
        torch::Tensor guidance_map,  // [B, L, H, W]
        torch::Tensor img_in,        // [B, H, W, 4]
        bool          requires_grad) {

        const int B = guidance_map.size(0);
        const int L = guidance_map.size(1);
        const int H = guidance_map.size(2);
        const int W = guidance_map.size(3);

        // variables to save
        auto save = torch::autograd::variable_list{};
        if (requires_grad) {
            // std::cout << "save context" << std::endl;
            save.resize(SAVED_INPUTS_CNT + L * SAVED_PER_LEVEL_CNT);
            save[0] = weight_map;
            save[1] = guidance_map;
            save[2] = img_in;

            auto options = torch::TensorOptions()
                               .device(weight_map.device())
                               .dtype(weight_map.dtype());
            for (int level = 0; level < L; level++) {
                int&& idx = SAVED_INPUTS_CNT + level * SAVED_PER_LEVEL_CNT;
                // clang-format off
                save[idx] = torch::zeros({B, H, W, 4}, options);   // rgb_filtered
                save[idx + 1] = torch::zeros({B, H, W}, options);  // max_map
                save[idx + 2] = torch::zeros({B, H, W}, options);  // inv_kernel_sum
                // clang-format on
            }
        }
        
        // create img_in texture
        cudaArray_t         img_in_cu;
        cudaTextureObject_t img_in_tex =
            host::create_texture_from_tensor<float4>(img_in_cu, H, W);
        auto stream  = at::cuda::getCurrentCUDAStream();

        // Write to output for each batch
        auto img_out = torch::zeros_like(img_in);  // [B, H, W, 4]
        for (int i = 0; i < B; i++) {
            // Read img_in to img_tex
            cudaMemcpy2DToArray(
                img_in_cu,
                0,
                0,
                img_in.index({i}).data_ptr<float>(),
                W * sizeof(float4),
                W * sizeof(float4),
                H,
                cudaMemcpyDeviceToDevice);

            // applying & fusing
            host::forward<float*>(
                stream,
                weight_map.index({i}),
                guidance_map.index({i}),
                img_in_tex,
                img_out.index({i}).data_ptr<float>(),
                save,
                i);
        }

        // free cuda array for texture object
        cudaDestroyTextureObject(img_in_tex);
        cudaFreeArray(img_in_cu);
       
        // save context
        if (requires_grad) {
            ctx->save_for_backward(save);
        }
        return img_out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list      grad_outputs) {
        auto saved        = ctx->get_saved_variables();
        auto weight_map   = saved[0];                      // [B, L, H, W]
        auto guidance_map = saved[1];                      // [B, L, H, W]
        auto img_in       = saved[2];                      // [B, H, W, 4]
        auto grad_output  = grad_outputs[0].contiguous();  // [B, H, W, 4]

        // get dimensions
        const int B = guidance_map.size(0);

        // create buffer
        auto grad_weight   = torch::zeros_like(weight_map);    // [B, L, H, W]
        auto grad_guidance = torch::zeros_like(guidance_map);  // [B, L, H, W]
        // std::cout << "grad_input" << grad_input << std::endl;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        for (int i = 0; i < B; i++) {
            host::backward(
                stream,
                grad_output.index({i}),
                img_in.index({i}),
                weight_map.index({i}),
                guidance_map.index({i}),
                grad_weight.index({i}),
                grad_guidance.index({i}),
                saved,
                i);
        }

        return {
            grad_weight,
            grad_guidance,
            torch::Tensor(),
            torch::Tensor(),
        };
    }
};

void filtering(
    cudaStream_t        stream,
    torch::Tensor       weight_map,    // [L, H, W]
    torch::Tensor       guidance_map,  // [L, H, W]
    cudaTextureObject_t img_in,        // [H, W, 4]
    cudaSurfaceObject_t img_out        // [H, W, 4]
) {
    auto empty = torch::autograd::variable_list{};
    host::forward<cudaSurfaceObject_t>(
        stream,
        weight_map,
        guidance_map,
        img_in,
        img_out,
        empty,
        -1);
}

torch::Tensor filtering_autograd(
    torch::Tensor weight_map,    // [B, L, H, W]
    torch::Tensor guidance_map,  // [B, L, H, W]
    torch::Tensor img_in,        // [B, H, W, 4]
    bool          requires_grad) {
    return Filtering::apply(weight_map, guidance_map, img_in, requires_grad);
}

}  // namespace denoiser