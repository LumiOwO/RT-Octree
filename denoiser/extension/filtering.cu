#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>

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
    cudaTextureObject_t img_in_tex,        // [H, W, 4]
    cudaTextureObject_t kernel_tex,        // [H, W, 4]
    const float* __restrict__ weight_map,  // [H, W]
    OutputType img_out,                    // [H, W, 4]
    float* __restrict__ rgb_filtered,      // [H, W, 4]
    float* __restrict__ max_map,           // [H, W]
    float* __restrict__ inv_kernel_sum     // [H, W]
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

#define __LOAD(tile_x, tile_y, image_x, image_y)                   \
    do {                                                           \
        int _tx = (tile_x);                                        \
        int _ty = (tile_y);                                        \
        int _ix = (image_x);                                       \
        int _iy = (image_y);                                       \
        if (_iy >= 0 && _iy < H && _ix >= 0 && _ix < W) {          \
            rgba_tile[_ty][_tx] =                                  \
                tex2D<float4>(img_in_tex, _ix + 0.5f, _iy + 0.5f); \
            kernel_tile[_ty][_tx] =                                \
                tex2D<float>(kernel_tex, _ix + 0.5f, _iy + 0.5f);  \
        } else {                                                   \
            rgba_tile[_ty][_tx]   = float4{0, 0, 0, 0};            \
            kernel_tile[_ty][_tx] = -FLT_MAX;                      \
        }                                                          \
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

    // if (ix == 0 && iy == 0) {
    //     printf("%d x %d\n", TILE_H, TILE_W);
    //     printf("rgba_tile\n");
    //     for (int i = 0; i < TILE_H; i++) {
    //         for (int j = 0; j < TILE_W; j++) {
    //             printf("%.0f ", rgba_tile[i][j].x);
    //         }
    //         printf("\n");
    //     }

    //     printf("kernel_tile\n");
    //     for (int i = 0; i < TILE_H; i++) {
    //         for (int j = 0; j < TILE_W; j++) {
    //             printf("%.0f ", kernel_tile[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }

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

    // grad *= weight_map[0] * (1 - weight_map[0]); // Already softmax!
    // printf(
    //     "%d: %f %f %f %f %f %f %f %f %f\n",
    //     idx,
    //     grad_output[0],
    //     grad_output[1],
    //     grad_output[2],
    //     rgb_sum_map[0],
    //     rgb_sum_map[1],
    //     rgb_sum_map[2],
    //     weight_map[0],
    //     (rgb_sum_map[0] + rgb_sum_map[1] + rgb_sum_map[2]) * weight_map[0] * (1 - weight_map[0]),
    //     grad);
}

template <typename scalar_t>
__global__ void grad_kernel_accumulate(
    const int SIZE,
    const int H,
    const int W,
    const int support,
    const scalar_t* __restrict__ grad_output,     // [H, W, 4]
    const scalar_t* __restrict__ img_in,          // [H, W, 4]
    const scalar_t* __restrict__ weight_map,      // [H, W]
    const scalar_t* __restrict__ rgb_filtered,    // [H, W, 4]
    const scalar_t* __restrict__ kernel_map,      // [H, W]
    const scalar_t* __restrict__ max_map,         // [H, W]
    const scalar_t* __restrict__ inv_kernel_sum,  // [H, W]
    scalar_t* grad_kernel                         // [H, W]
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
    kernel_map     += neighbor_idx;
    max_map        += pixel_idx;
    inv_kernel_sum += pixel_idx;
    grad_kernel    += neighbor_idx;
    // clang-format on

    // accumulate grad
    const scalar_t& k   = __expf(kernel_map[0] - max_map[0]) * inv_kernel_sum[0];
    scalar_t        res = 0;
    res += grad_output[0] * (img_in[0] - rgb_filtered[0]);
    res += grad_output[1] * (img_in[1] - rgb_filtered[1]);
    res += grad_output[2] * (img_in[2] - rgb_filtered[2]);
    res *= weight_map[0] * k;

    gpuAtomicAdd(&grad_kernel[0], res);
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
    cudaTextureObject_t kernel_tex,     // [H, W, 4]
    const float*        weight_map,     // [H, W]
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

#define __ARGS_TEMP__                                                         \
    H, W, img_in_tex, kernel_tex, weight_map, img_out, rgb_filtered, max_map, \
        inv_kernel_sum

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
    int           W,
    torch::Tensor tensor) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float_n>();
    cudaMallocArray(&cuArray, &channelDesc, W, H);
    cudaMemcpy2DToArray(
        cuArray,
        0,
        0,
        tensor.data_ptr<float>(),
        W * sizeof(float_n),
        W * sizeof(float_n),
        H,
        cudaMemcpyDeviceToDevice);

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
__host__ void accumulate_one_level(
    cudaStream_t        stream,
    const int           level,
    const int           H,
    const int           W,
    const int           L,
    cudaTextureObject_t img_in_tex,  // [H, W, 4]
    cudaTextureObject_t kernel_tex,  // [H, W]
    torch::Tensor       weight_map,  // [H, W]
    OutputType          img_out,     // [H, W, 4]

    // clang-format off
    torch::autograd::variable_list& save
    // clang-format on
) {
    // auto stream_guard =
    //     at::cuda::CUDAStreamGuard(stream);

    // tensors
    bool need_save      = !save.empty();
    auto rgb_filtered   = torch::Tensor();
    auto max_map        = torch::Tensor();
    auto inv_kernel_sum = torch::Tensor();
    if (need_save) {
        rgb_filtered = torch::zeros(
            {weight_map.sizes()[0], weight_map.sizes()[1], 4},
            torch::TensorOptions()
                .device(weight_map.device())
                .dtype(weight_map.dtype()));
        max_map        = torch::zeros_like(weight_map);
        inv_kernel_sum = torch::zeros_like(weight_map);
    }

    // apply kernel
    kernel_apply<OutputType>(
        stream,
        level + 1,  // support
        H,
        W,
        img_in_tex,
        kernel_tex,
        weight_map.data_ptr<float>(),
        img_out,
        (need_save ? rgb_filtered.data_ptr<float>() : nullptr),
        (need_save ? max_map.data_ptr<float>() : nullptr),
        (need_save ? inv_kernel_sum.data_ptr<float>() : nullptr));

    // save tensors
    if (need_save) {
        int&& idx     = SAVED_INPUTS_CNT + level * SAVED_PER_LEVEL_CNT;
        save[idx]     = rgb_filtered;
        save[idx + 1] = max_map;
        save[idx + 2] = inv_kernel_sum;
    }
}

template <typename OutputType>
__host__ void forward(
    cudaStream_t        stream,
    torch::Tensor       weight_map,  // [L, H, W]
    torch::Tensor       kernel_map,  // [L, H, W]
    cudaTextureObject_t img_in_tex,  // [H, W, 4]
    OutputType          img_out,     // [H, W, 4]
    // clang-format off
    torch::autograd::variable_list& save
    // ,int batch_idx
    // clang-format on
) {
    const int L       = kernel_map.size(0);
    const int H       = kernel_map.size(1);
    const int W       = kernel_map.size(2);

    // create kernel texture
    auto kernel_cus  = std::vector<cudaArray_t>{};
    auto kernel_texs = std::vector<cudaTextureObject_t>{};
    for (int level = 0; level < L; level++) {
        cudaArray_t         kernel_cu;
        cudaTextureObject_t kernel_tex = create_texture_from_tensor<float>(
            kernel_cu, H, W, kernel_map.index({level}));
        kernel_cus.emplace_back(kernel_cu);
        kernel_texs.emplace_back(kernel_tex);
    }

    for (int level = 0; level < L; level++) {
        accumulate_one_level<OutputType>(
            stream,
            level,
            H,
            W,
            L,
            img_in_tex,
            kernel_texs[level],
            weight_map.index({level}),
            img_out,
            save);
    }

    for (int level = 0; level < L; level++) {
        cudaDestroyTextureObject(kernel_texs[level]);
        cudaFreeArray(kernel_cus[level]);
    }
}
};  // namespace host

class Filtering : public torch::autograd::Function<Filtering> {
public:
    constexpr static const int N_THREADS = 512;

    // forward
    static torch::Tensor forward(
        // clang-format off
        torch::autograd::AutogradContext* ctx,
        // clang-format on
        torch::Tensor weight_map,  // [L, H, W]
        torch::Tensor kernel_map,  // [L, H, W]
        torch::Tensor img_in,      // [H, W, 4]
        bool          requires_grad) {

        const int L = kernel_map.size(0);
        const int H = kernel_map.size(1);
        const int W = kernel_map.size(2);

        // variables to save
        auto save = torch::autograd::variable_list{};
        if (requires_grad) {
            // std::cout << "save context" << std::endl;
            save.resize(SAVED_INPUTS_CNT + L * SAVED_PER_LEVEL_CNT);
            save[0] = weight_map;
            save[1] = kernel_map;
            save[2] = img_in;
        }

        // create img_in texture
        cudaArray_t         img_in_cu;
        cudaTextureObject_t img_in_tex =
            host::create_texture_from_tensor<float4>(img_in_cu, H, W, img_in);
        
        // applying & fusing
        auto stream  = at::cuda::getCurrentCUDAStream();
        auto img_out = torch::zeros_like(img_in);  // [H, W, 4]
        host::forward<float*>(
            stream,
            weight_map,
            kernel_map,
            img_in_tex,
            img_out.data_ptr<float>(),
            save);

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
        auto saved       = ctx->get_saved_variables();
        auto weight_map  = saved[0];                      // [L, H, W]
        auto kernel_map  = saved[1];                      // [L, H, W]
        auto img_in      = saved[2];                      // [H, W, 4]
        auto grad_output = grad_outputs[0].contiguous();  // [H, W, 4]

        // get dimensions
        const int L    = kernel_map.size(0);
        const int H    = kernel_map.size(1);
        const int W    = kernel_map.size(2);
        const int SIZE = H * W;

        // create buffer
        auto grad_weight = torch::zeros_like(weight_map);
        auto grad_kernel = torch::zeros_like(kernel_map);
        // std::cout << "grad_input" << grad_input << std::endl;

        // accumulate grads for each level
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
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
                saved[saved_idx],            // rgb_filtered
                saved[saved_idx + 1],        // max_map
                saved[saved_idx + 2],        // inv_kernel_sum
                weight_map.index({level}),   // weight_map
                kernel_map.index({level}),   // kernel_map
                grad_weight.index({level}),  // grad_weight
                grad_kernel.index({level})   // grad_kernel
            );
        }

        return {
            grad_weight,
            grad_kernel,
            torch::Tensor(),
            torch::Tensor(),
        };
    }

    static void grad_accumulate_one_level(
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
        torch::Tensor kernel_map,      // [H, W]
        torch::Tensor grad_weight,     // [H, W]
        torch::Tensor grad_kernel      // [H, W]
    ) {
        // compute weight grads
        const int grad_weight_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
        // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_kernel.type(), "backward_grad_weight", ([&] {
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
        const int grad_kernel_blocks =
            N_BLOCKS_NEEDED(SIZE * K_SIZE, N_THREADS);
        // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_kernel.type(), "backward_grad_kernel", ([&] {
                kernel::grad_kernel_accumulate<scalar_t>
                    <<<grad_kernel_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        H,
                        W,
                        support,
                        grad_output.data_ptr<scalar_t>(),
                        img_in.data_ptr<scalar_t>(),
                        weight_map.data_ptr<scalar_t>(),
                        rgb_filtered.data_ptr<scalar_t>(),
                        kernel_map.data_ptr<scalar_t>(),
                        max_map.data_ptr<scalar_t>(),
                        inv_kernel_sum.data_ptr<scalar_t>(),
                        grad_kernel.data_ptr<scalar_t>());
            }));
        // clang-format on
    }
};

void filtering(
    cudaStream_t        stream,
    torch::Tensor       weight_map,  // [L, H, W]
    torch::Tensor       kernel_map,  // [L, H, W]
    cudaTextureObject_t img_in,      // [H, W, 4]
    cudaSurfaceObject_t img_out      // [H, W, 4]
){
    auto empty = torch::autograd::variable_list{};
    host::forward<cudaSurfaceObject_t>(
        stream,
        weight_map,
        kernel_map,
        img_in,
        img_out,
        empty);
}

torch::Tensor filtering_autograd(
    torch::Tensor weight_map,  // [L, H, W]
    torch::Tensor kernel_map,  // [L, H, W]
    torch::Tensor img_in,      // [H, W, 4]
    bool          requires_grad) {
    return Filtering::apply(weight_map, kernel_map, img_in, requires_grad);
}

}  // namespace denoiser