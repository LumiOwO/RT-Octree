#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "filtering.h"

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x)                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
                #x " must be an int tensor")
#define CHECK_IS_FLOATING(x)                                   \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float ||    \
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

__device__ __forceinline__ float  _exp(const float x) { return __expf(x); }
__device__ __forceinline__ double _exp(const double x) { return exp(x); }
__device__ __forceinline__ c10::Half _exp(const c10::Half x) { return hexp(x); }

template <typename scalar_t>
__global__ void weights_softmax(const int SIZE, const int L, scalar_t* input) {
    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    input += idx * (L << 1);

    scalar_t max_val = input[0];
    for (int i = 1; i < L; i++) {
        max_val = fmaxf(max_val, input[i << 1]);
    }

    scalar_t sum = 0;
    for (int i = 0; i < L; i++) {
        auto& w = input[i << 1];
        w = _exp(w - max_val);
        sum += w;
    }

    const scalar_t inv_sum = 1 / sum;
    for (int i = 0; i < L; i++) {
        input[i << 1] *= inv_sum;
    }
}

template <int BLOCK_H, int BLOCK_W, int SUPPORT>
__global__ void applying(
    const int           H,
    const int           W,
    cudaTextureObject_t imgs_in_tex,       // [H, W, 4]
    cudaTextureObject_t kernel_tex,        // [H, W, 4]
    const float* __restrict__ weight_map,  // [H, W]
    const float* __restrict__ max_map,     // [H, W]
    float* imgs_out,                       // [H, W, 4]
    float* rgb_filtered,                   // [H, W, 4]
    float* inv_kernel_sum                  // [H, W]
) {
    constexpr int SUPPORT2 = SUPPORT * 2;
    constexpr int TILE_H = BLOCK_H + SUPPORT2;
    constexpr int TILE_W = BLOCK_W + SUPPORT2;

    // locate
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);  // col
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);  // row
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // load tile elements
    __shared__ float4 rgba_tile[TILE_H][TILE_W];
    __shared__ float  kernel_tile[TILE_H][TILE_W];
    // !!!!! values out of image is undefined !!!!!
#define __LOAD(tile_x, tile_y, image_x, image_y)                    \
    do {                                                            \
        int _tx = (tile_x);                                         \
        int _ty = (tile_y);                                         \
        int _ix = (image_x);                                        \
        int _iy = (image_y);                                        \
        if (_iy >= 0 && _iy < H && _ix >= 0 && _ix < W) {           \
            rgba_tile[_ty][_tx] =                                   \
                tex2D<float4>(imgs_in_tex, _ix + 0.5f, _iy + 0.5f); \
            kernel_tile[_ty][_tx] =                                 \
                tex2D<float>(kernel_tex, _ix + 0.5f, _iy + 0.5f);   \
        }                                                           \
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
    weight_map += idx;
    max_map    += idx;
    imgs_out   += (idx << 2);

    // apply
    float3 rgb = {0, 0, 0};
    float max_val = max_map[0];
    float kernel_sum = 0;
    for (int dy = 0; dy <= SUPPORT2; dy++) {
        int _iy = iy - SUPPORT + dy;
        if (_iy < 0 || _iy >= H) continue;
        int _ty = ty + dy;

        for (int dx = 0; dx <= SUPPORT2; dx++) {
            int _ix = ix - SUPPORT + dx;
            if (_ix < 0 || _ix >= W) continue;
            int _tx = tx + dx;

            // compute
            float k = __expf(kernel_tile[_ty][_tx] - max_val);
            kernel_sum += k;
            auto& t_rgb = rgba_tile[_ty][_tx];
            rgb.x += t_rgb.x * k;
            rgb.y += t_rgb.y * k;
            rgb.z += t_rgb.z * k;
        }
    }

    // accumulate
    float w = weight_map[0] / kernel_sum;
    at::native::fastAtomicAdd(imgs_out, 0, 1, rgb.x * w, true);
    at::native::fastAtomicAdd(imgs_out, 1, 1, rgb.y * w, true);
    at::native::fastAtomicAdd(imgs_out, 2, 1, rgb.z * w, true);

    // save for backward
    if (rgb_filtered != nullptr) {
        rgb_filtered   += idx * IMG_CHANNELS;
        inv_kernel_sum += idx;
        inv_kernel_sum[0] = 1.0f / kernel_sum;
        rgb_filtered[0]   = rgb.x * inv_kernel_sum[0];
        rgb_filtered[1]   = rgb.y * inv_kernel_sum[0];
        rgb_filtered[2]   = rgb.z * inv_kernel_sum[0];
    }
}

template <typename scalar_t>
__global__ void normalize(
    const int SIZE,
    const scalar_t* __restrict__ weight_map,  // [H, W]
    const scalar_t* __restrict__ rgb_sum,     // [H, W, 4]
    const scalar_t* __restrict__ kernel_sum,  // [H, W]
    scalar_t* imgs_out                        // [H, W, 4]
) {

    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    weight_map += idx;
    rgb_sum += idx * IMG_CHANNELS;
    kernel_sum += idx;
    imgs_out += idx * IMG_CHANNELS;

    // accumulate
    const scalar_t& k = weight_map[0] / kernel_sum[0];
    gpuAtomicAdd(&imgs_out[0], k * rgb_sum[0]);
    gpuAtomicAdd(&imgs_out[1], k * rgb_sum[1]);
    gpuAtomicAdd(&imgs_out[2], k * rgb_sum[2]);
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
    const scalar_t* __restrict__ imgs_in,         // [H, W, 4]
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
    imgs_in        += neighbor_idx * IMG_CHANNELS;
    weight_map     += pixel_idx;
    rgb_filtered   += pixel_idx * IMG_CHANNELS;
    kernel_map     += neighbor_idx;
    max_map        += pixel_idx;
    inv_kernel_sum += pixel_idx;
    grad_kernel    += neighbor_idx;
    // clang-format on

    // accumulate grad
    const scalar_t& k   = _exp(kernel_map[0] - max_map[0]) * inv_kernel_sum[0];
    scalar_t res = 0;
    res += grad_output[0] * (imgs_in[0] - rgb_filtered[0]);
    res += grad_output[1] * (imgs_in[1] - rgb_filtered[1]);
    res += grad_output[2] * (imgs_in[2] - rgb_filtered[2]);
    res *= weight_map[0] * k;
    
    gpuAtomicAdd(&grad_kernel[0], res);
}

}  // namespace kernel

#define __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, ...)   \
    {                                                                  \
        dim3 block_size;                                               \
        block_size.x = BLOCK_W; /* col */                              \
        block_size.y = BLOCK_H; /* row */                              \
        block_size.z = 1;                                              \
        dim3 grid_size;                                                \
        grid_size.x = N_BLOCKS_NEEDED(W, BLOCK_W); /* col */           \
        grid_size.y = N_BLOCKS_NEEDED(H, BLOCK_H); /* row */           \
        grid_size.z = 1;                                               \
        kernel::applying<BLOCK_H, BLOCK_W, SUPPORT>                    \
            <<<grid_size, block_size, 0, STREAM>>>(H, W, __VA_ARGS__); \
    }

#define __KERNEL_APPLY_SWITCH_H_W(STREAM, SUPPORT, H, W, ...)                 \
    {                                                                         \
        constexpr int BLOCK_H = 16;                                           \
        constexpr int BLOCK_W = 32;                                           \
        __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, __VA_ARGS__); \
    }                                                                         \
    // if (H == 800 && W == 800) {                                            \
        /* nerf synthetic */                                                  \
        constexpr int BLOCK_H = 32;                                           \
        constexpr int BLOCK_W = 32;                                           \
        __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, __VA_ARGS__); \
    } else if (H == 756 && W == 1008) {                                       \
        /* llff */                                                            \
        constexpr int BLOCK_H = 16;                                           \
        constexpr int BLOCK_W = 32;                                           \
        __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, __VA_ARGS__); \
    } else if (H == 1080 && W == 1920) {                                      \
        /* tanks and temples */                                               \
        constexpr int BLOCK_H = 16;                                           \
        constexpr int BLOCK_W = 32;                                           \
        __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, __VA_ARGS__); \
    } else {                                                                  \
        /* default */                                                         \
        constexpr int BLOCK_H = 16;                                           \
        constexpr int BLOCK_W = 32;                                           \
        __KERNEL_APPLY(STREAM, SUPPORT, BLOCK_H, BLOCK_W, H, W, __VA_ARGS__); \
    }

#define __KERNEL_APPLY_SWITCH_SUPPORT(STREAM, support, H, W, ...)     \
    switch (support) {                                                \
        case 1:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 1, H, W, __VA_ARGS__);  \
            break;                                                    \
        case 2:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 2, H, W, __VA_ARGS__);  \
            break;                                                    \
        case 3:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 3, H, W, __VA_ARGS__);  \
            break;                                                    \
        case 4:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 4, H, W, __VA_ARGS__);  \
            break;                                                    \
        case 5:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 5, H, W, __VA_ARGS__);  \
            break;                                                    \
        case 6:                                                       \
            __KERNEL_APPLY_SWITCH_H_W(STREAM, 6, H, W, __VA_ARGS__);  \
            break;                                                    \
        default:                                                      \
            throw std::runtime_error(                                 \
                "Kernel size == " + std::to_string(support * 2 + 1) + \
                " not supported.");                                   \
    }

#define KERNEL_APPLY(STREAM, support, H, W, ...) \
    __KERNEL_APPLY_SWITCH_SUPPORT(STREAM, support, H, W, __VA_ARGS__)

template <typename float_n>
cudaTextureObject_t create_texture_from_tensor(
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

    cudaResourceDesc resDesc   = {};
    resDesc.resType            = cudaResourceTypeArray;
    resDesc.res.array.array    = cuArray;

    cudaTextureDesc texDesc    = {};
    texDesc.normalizedCoords   = false;
    texDesc.filterMode         = cudaFilterModePoint;
    texDesc.addressMode[0]     = cudaAddressModeWrap;
    texDesc.addressMode[1]     = cudaAddressModeWrap;
    texDesc.readMode           = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    return texObj;
}

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
        torch::Tensor imgs_in,     // [H, W, 4]
        bool          requires_grad) {

        const int L    = kernel_map.size(0);
        const int H    = kernel_map.size(1);
        const int W    = kernel_map.size(2);
        // const int SIZE = B * H * W;
        // std::cout << L << " " << H << " " << W << std::endl;
        // std::cout << (uint64_t)imgs_out.data_ptr() << std::endl;

        auto imgs_out = torch::zeros_like(imgs_in);  // [H, W, 4]
        imgs_out.select(-1, 3).fill_(1); // set alpha channel to 1
        // std::cout << "imgs_out" << imgs_out << std::endl;

        // // weight normalization
        // const int weights_softmax_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
        // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        //     kernel_map.type(), "forward_weights_softmax", ([&] {
        //         kernel::weights_softmax<scalar_t>
        //             <<<weights_softmax_blocks, N_THREADS, 0, stream>>>(
        //                 SIZE, L, input.data_ptr<scalar_t>());
        //     }));
        // // permute
        // input = input.permute({3, 0, 1, 2}).contiguous();  // [L * 2, H, W]

        // variables to save
        auto save = torch::autograd::variable_list{};
        if (requires_grad) {
            // std::cout << "save context" << std::endl;
            save.resize(SAVED_INPUTS_CNT + L * SAVED_PER_LEVEL_CNT);
            save[0] = weight_map;
            save[1] = kernel_map;
            save[2] = imgs_in;
        }

        // create imgs_in texture
        cudaArray_t imgs_in_cu;
        cudaTextureObject_t imgs_in_tex =
            create_texture_from_tensor<float4>(imgs_in_cu, H, W, imgs_in);
        // create kernel texture
        auto kernel_cus = std::vector<cudaArray_t>{};
        auto kernel_texs = std::vector<cudaTextureObject_t>{};
        for (int level = 0; level < L; level++) {
            cudaArray_t         kernel_cu;
            cudaTextureObject_t kernel_tex = create_texture_from_tensor<float>(
                kernel_cu, H, W, kernel_map.index({level}));
            kernel_cus.emplace_back(kernel_cu);
            kernel_texs.emplace_back(kernel_tex);
        }

        // std::cout << "imgs_in_tex" << tex2D<float>(imgs_in_tex, 400 * 3 + .5f, 400.5f) << std::endl;
        // std::cout << "kernel_tex"
        //           << tex2D<float>(kernel_texs[0], 400.5f, 400.5f)
        //           << std::endl;
        // std::cout << "all kernel_map.sizes()" << kernel_map.sizes() << std::endl;
        // std::cout << "all kernel_map.data_ptr()"
        //           << (uint64_t)kernel_map.data_ptr<float>() << std::endl;
        // applying & fusing
        auto stream = at::cuda::getCurrentCUDAStream();
        // auto streams = std::vector<at::cuda::CUDAStream>{};
        // streams.emplace_back(at::cuda::getCurrentCUDAStream());
        // streams.emplace_back(at::cuda::getStreamFromPool());
        // auto weight_cus = std::vector<cudaArray_t>{};
        for (int level = 0; level < L; level++) {
            // get current stream
            // auto stream = at::cuda::getStreamFromPool();
            // streams.emplace_back(stream);
            // auto stream = streams[level & 0x1];

            accumulate_one_level(
                save,
                stream,
                level,
                H,
                W,
                L,
                imgs_in_tex,
                kernel_texs[level],
                kernel_map.index({level}),
                weight_map.index({level}),
                imgs_out);
        }
        // for (auto& stream : streams) {
        //     cudaStreamSynchronize(stream);
        // }

        // free cuda array for texture object
        cudaFreeArray(imgs_in_cu);
        // for (auto& cu : weight_cus) {
        //     cudaFreeArray(cu);
        // }
        for (auto& cu : kernel_cus) {
            cudaFreeArray(cu);
        }

        if (requires_grad) {
            ctx->save_for_backward(save);
        }
        // std::cout << (uint64_t)imgs_out.data_ptr() << std::endl;
        return imgs_out;
    }

    static void accumulate_one_level(
        // clang-format off
        torch::autograd::variable_list& save,
        at::cuda::CUDAStream stream,
        // clang-format on
        const int           level,
        const int           H,
        const int           W,
        const int           L,
        cudaTextureObject_t imgs_in_tex,  // [H, W, 4]
        cudaTextureObject_t kernel_tex,   // [H, W]
        torch::Tensor       kernel_map,   // [H, W]
        torch::Tensor       weight_map,   // [H, W]
        torch::Tensor       imgs_out      // [H, W, 4]
    ) {

        // kernel size
        const int support = level + 1;
        const int K       = 1 + (support << 1);
        // const int K_SIZE  = K * K;

        // use max pooling to find maximum value in the kernel
        // std::cout << "kernel_map.sizes()"
        //           << kernel_map.sizes() << std::endl;
        // std::cout << "kernel_map.data_ptr()"
        //           << (uint64_t)kernel_map.data_ptr<float>() << std::endl;
        // std::cout << "kernel_map.unsqueeze(0)" << kernel_map.unsqueeze(0).sizes()
        //           << std::endl;

        // std::cout << "kernel_map.sizes()" << kernel_map.sizes() << std::endl;
        // std::cout << "kernel_map.unsqueeze(0)"
        //           << kernel_map.unsqueeze(0).sizes() << std::endl;
        // std::cout << "kernel_map" << kernel_map << std::endl;

        // kernel_map   = kernel_map.add(kernel_map);
        // std::cout << "kernel_map.sizes()" << kernel_map.sizes() << std::endl;

        // std::cout << "stream" << (uint64_t)stream.stream() << std::endl;
        at::cuda::CUDAStreamGuard stream_guard(stream);
        namespace F  = torch::nn::functional;
        auto max_map = F::max_pool2d(
            kernel_map.unsqueeze(0),  // [1, H, W]
            F::MaxPool2dFuncOptions(K).padding(support).stride(1));
        max_map = max_map.squeeze(0);  // [H, W]
        // std::cout << "max_map.sizes()" << max_map.sizes() << std::endl;
        // auto max_map = torch::ones_like(kernel_map);

        // tensors
        bool need_save     = !save.empty();
        auto rgb_filtered  = torch::Tensor();
        auto inv_kernel_sum = torch::Tensor();
        if (need_save) {
            rgb_filtered = torch::zeros_like(imgs_out);
            inv_kernel_sum = torch::zeros_like(kernel_map);
        }

        // apply kernel
        KERNEL_APPLY(
            stream.stream(),
            support,
            H,
            W,
            imgs_in_tex,
            kernel_tex,
            weight_map.data_ptr<float>(),
            max_map.data_ptr<float>(),
            imgs_out.data_ptr<float>(),
            (need_save ? rgb_filtered.data_ptr<float>() : nullptr),
            (need_save ? inv_kernel_sum.data_ptr<float>() : nullptr));

        // const int applying_blocks = N_BLOCKS_NEEDED(SIZE * K_SIZE, N_THREADS);
        // // clang-format off
        // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        //     kernel_map.type(), "forward_applying", ([&] {
        //         kernel::applying<scalar_t>
        //             <<<applying_blocks, N_THREADS, 0, stream.stream()>>>(
        //                 SIZE,
        //                 H,
        //                 W,
        //                 support,
        //                 kernel_map.data_ptr<scalar_t>(),
        //                 imgs_in.data_ptr<scalar_t>(),
        //                 max_map.data_ptr<scalar_t>(),
        //                 rgb_sum.data_ptr<scalar_t>(),
        //                 kernel_sum.data_ptr<scalar_t>());
        //     }));
        // // clang-format on



        // // normalize & accumulate
        // const int normalize_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
        // // clang-format off
        // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        //     kernel_map.type(), "forward_normalize", ([&] {
        //         kernel::normalize<scalar_t>
        //             <<<normalize_blocks, N_THREADS, 0, stream.stream()>>>(
        //                 SIZE,
        //                 weight_map.data_ptr<scalar_t>(),
        //                 rgb_sum.data_ptr<scalar_t>(),
        //                 kernel_sum.data_ptr<scalar_t>(),
        //                 imgs_out.data_ptr<scalar_t>());
        //     }));
        // // clang-format on

        // save tensors
        if (need_save) {
            int&& idx     = SAVED_INPUTS_CNT + level * SAVED_PER_LEVEL_CNT;
            save[idx]     = rgb_filtered;
            save[idx + 1] = max_map;
            save[idx + 2] = inv_kernel_sum;
        }
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list      grad_outputs  
    ) {
        auto saved       = ctx->get_saved_variables();
        auto weight_map  = saved[0];  // [L, H, W]
        auto kernel_map  = saved[1];  // [L, H, W]
        auto imgs_in     = saved[2];  // [H, W, 4]
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
                imgs_in,
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
        torch::Tensor imgs_in,         // [H, W, 4]
        torch::Tensor rgb_filtered,    // [H, W, 4]
        torch::Tensor max_map,         // [H, W]
        torch::Tensor inv_kernel_sum,  // [H, W]
        torch::Tensor weight_map,      // [H, W]
        torch::Tensor kernel_map,      // [H, W]
        torch::Tensor grad_weight,     // [H, W]
        torch::Tensor grad_kernel      // [H, W]
    ) {
        // std::cout << "grad_weight" << grad_weight << std::endl;

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

        // std::cout << "SIZE" << SIZE << std::endl;
        // std::cout << "grad_output" << grad_output.is_contiguous() << grad_output
        //           << std::endl;
        // std::cout << "weight_map" << weight_map << std::endl;
        // std::cout << "rgb_sum_map" << rgb_sum_map << std::endl;
        // std::cout << "grad_weight" << grad_weight << std::endl;

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
                        imgs_in.data_ptr<scalar_t>(),
                        weight_map.data_ptr<scalar_t>(),
                        rgb_filtered.data_ptr<scalar_t>(),
                        kernel_map.data_ptr<scalar_t>(),
                        max_map.data_ptr<scalar_t>(),
                        inv_kernel_sum.data_ptr<scalar_t>(),
                        grad_kernel.data_ptr<scalar_t>());
            }));
        // clang-format on

        // std::cout << "grad_output" << grad_output << std::endl;
        // std::cout << "weight_map" << weight_map << std::endl;
        // std::cout << "rgb_sum_map" << rgb_sum_map << std::endl;
        // std::cout << "kernel_map" << kernel_map << std::endl;
        // std::cout << "max_map" << max_map << std::endl;
        // std::cout << "kernel_sum" << max_map << std::endl;
        // std::cout << "grad_kernel" << grad_kernel << std::endl;
    }
};

torch::Tensor filtering(
    torch::Tensor weight_map,  // [L, H, W]
    torch::Tensor kernel_map,  // [L, H, W]
    torch::Tensor imgs_in,     // [H, W, 4]
    bool          requires_grad) {
    return Filtering::apply(
        weight_map, kernel_map, imgs_in, requires_grad);
}

}  // namespace denoiser