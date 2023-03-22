#include <ATen/cuda/CUDAContext.h>
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

#define CUDA_GET_THREAD_ID(tid, Q)                         \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q) return
#define N_BLOCKS_NEEDED(Q, N_CUDA_THREADS) ((Q - 1) / N_CUDA_THREADS + 1)

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

template <typename scalar_t>
__global__ void applying(
    const int SIZE,
    const int H,
    const int W,
    const int support,
    const scalar_t* __restrict__ kernel_map,  // [B, H, W]
    const scalar_t* __restrict__ imgs_in,     // [B, H, W, 3]
    const scalar_t* __restrict__ max_map,     // [B, H, W]
    scalar_t* rgb_sum,                        // [B, H, W, 3]
    scalar_t* kernel_sum                      // [B, H, W]
) {

    // locate
    const int K        = 1 + (support << 1);
    const int K_SIZE   = K * K;
    CUDA_GET_THREAD_ID(idx, SIZE * K_SIZE);
    const int pixel_idx    = idx / K_SIZE;
    const int kernel_idx   = idx % K_SIZE;

    const int lower_bound  = pixel_idx - pixel_idx % (H * W);
    const int pixel_y      = (pixel_idx - lower_bound) / W;
    const int neighbor_y   = pixel_y - support + kernel_idx / K;
    if (neighbor_y < 0 || neighbor_y >= H) return;
    const int pixel_x      = pixel_idx % W;
    const int neighbor_x   = pixel_x - support + kernel_idx % K;
    if (neighbor_x < 0 || neighbor_x >= W) return;

    const int neighbor_idx = lower_bound + neighbor_y * W + neighbor_x;
    kernel_map += neighbor_idx;
    imgs_in    += neighbor_idx * 3;
    max_map    += pixel_idx;
    rgb_sum    += pixel_idx * 3;
    kernel_sum += pixel_idx;

    // accumulate
    const scalar_t& k = _exp(kernel_map[0] - max_map[0]);
    // gpuAtomicAdd(&rgb_sum[0], k * imgs_in[0]);
    // gpuAtomicAdd(&rgb_sum[1], k * imgs_in[1]);
    // gpuAtomicAdd(&rgb_sum[2], k * imgs_in[2]);
    // gpuAtomicAdd(kernel_sum, k);
    at::native::fastAtomicAdd(rgb_sum, 0, 1, k * imgs_in[0], true);
    at::native::fastAtomicAdd(rgb_sum, 1, 1, k * imgs_in[1], true);
    at::native::fastAtomicAdd(rgb_sum, 2, 1, k * imgs_in[2], true);
    at::native::fastAtomicAdd(kernel_sum, 0, 1, k, true);
}

template <typename scalar_t>
__global__ void normalize(
    const int SIZE,
    const scalar_t* __restrict__ weight_map,  // [B, H, W]
    const scalar_t* __restrict__ rgb_sum,     // [B, H, W, 3]
    const scalar_t* __restrict__ kernel_sum,  // [B, H, W]
    scalar_t* imgs_out                        // [B, H, W, 3]
) {

    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    weight_map += idx;
    rgb_sum    += idx * 3;
    kernel_sum += idx;
    imgs_out   += idx * 3;

    // accumulate
    const scalar_t& k = weight_map[0] / kernel_sum[0];
    gpuAtomicAdd(&imgs_out[0], k * rgb_sum[0]);
    gpuAtomicAdd(&imgs_out[1], k * rgb_sum[1]);
    gpuAtomicAdd(&imgs_out[2], k * rgb_sum[2]);
}

template <typename scalar_t>
__global__ void grad_weight_accumulate(
    const int SIZE,
    const scalar_t* __restrict__ grad_output,  // [B, H, W, 3]
    const scalar_t* __restrict__ weight_map,   // [B, H, W]
    const scalar_t* __restrict__ rgb_sum_map,  // [B, H, W, 3]
    scalar_t* grad_weight                      // [B, H, W]
) {
    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    grad_output += idx * 3;
    weight_map  += idx;
    rgb_sum_map += idx * 3;
    grad_weight += idx;

    scalar_t& grad = grad_weight[0];
    grad += grad_output[0] * rgb_sum_map[0];
    grad += grad_output[1] * rgb_sum_map[1];
    grad += grad_output[2] * rgb_sum_map[2];

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
    const scalar_t* __restrict__ grad_output,  // [B, H, W, 3]
    const scalar_t* __restrict__ imgs_in,      // [B, H, W, 3]
    const scalar_t* __restrict__ weight_map,   // [B, H, W]
    const scalar_t* __restrict__ rgb_sum_map,  // [B, H, W, 3]
    const scalar_t* __restrict__ kernel_map,   // [B, H, W]
    const scalar_t* __restrict__ max_map,      // [B, H, W]
    const scalar_t* __restrict__ kernel_sum,   // [B, H, W]
    scalar_t* grad_kernel                      // [B, H, W]
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
    grad_output += pixel_idx * 3;
    imgs_in     += neighbor_idx * 3;
    weight_map  += pixel_idx;
    rgb_sum_map += pixel_idx * 3;
    kernel_map  += neighbor_idx;
    max_map     += pixel_idx;
    kernel_sum  += pixel_idx;
    grad_kernel += neighbor_idx;
    // clang-format on

    // accumulate grad
    const scalar_t& k = _exp(kernel_map[0] - max_map[0]) / kernel_sum[0];
    scalar_t res = 0;
    res += grad_output[0] * (imgs_in[0] - rgb_sum_map[0]);
    res += grad_output[1] * (imgs_in[0] - rgb_sum_map[0]);
    res += grad_output[2] * (imgs_in[0] - rgb_sum_map[0]);
    res *= weight_map[0] * k;
    
    gpuAtomicAdd(&grad_kernel[0], res);
}

}  // namespace kernel

class Filtering : public torch::autograd::Function<Filtering> {
public:
    constexpr static const int N_THREADS = 512;

    // forward
    static torch::Tensor forward(
        // clang-format off
        torch::autograd::AutogradContext* ctx,
        // clang-format on
        torch::Tensor weight_map,  // [L, B, H, W]
        torch::Tensor kernel_map,  // [L, B, H, W]
        torch::Tensor imgs_in,     // [B, H, W, 3]
        bool          requires_grad) {

        const int L    = kernel_map.size(0);
        const int B    = kernel_map.size(1);
        const int H    = kernel_map.size(2);
        const int W    = kernel_map.size(3);
        const int SIZE = B * H * W;
        // std::cout << L << " " << H << " " << W << std::endl;
        // std::cout << (uint64_t)imgs_out.data_ptr() << std::endl;

        auto imgs_out = torch::zeros_like(imgs_in);  // [B, H, W, 3]

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
            save.resize((L + 1) * 3);
            save[0] = weight_map;
            save[1] = kernel_map;
            save[2] = imgs_in;
        }

        // std::cout << "all kernel_map.sizes()" << kernel_map.sizes() << std::endl;
        // std::cout << "all kernel_map.data_ptr()"
        //           << (uint64_t)kernel_map.data_ptr<float>() << std::endl;
        // applying & fusing
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        for (int level = 0; level < L; level++) {
            accumulate_one_level(
                save,
                stream,
                level,
                SIZE,
                H,
                W,
                L,
                weight_map.index({level}),  // weight map
                kernel_map.index({level}),  // kernel map
                imgs_in,
                imgs_out);
        }

        ctx->save_for_backward(save);
        // std::cout << (uint64_t)imgs_out.data_ptr() << std::endl;
        return imgs_out;
    }

    static void accumulate_one_level(
        // clang-format off
        torch::autograd::variable_list& save,
        // clang-format on
        cudaStream_t  stream,
        const int     level,
        const int     SIZE,
        const int     H,
        const int     W,
        const int     L,
        torch::Tensor weight_map,  // [B, H, W]
        torch::Tensor kernel_map,  // [B, H, W]
        torch::Tensor imgs_in,     // [B, H, W, 3]
        torch::Tensor imgs_out     // [B, H, W, 3]
    ) {

        // kernel size
        const int support = level + 1;
        const int K       = 1 + (support << 1);
        const int K_SIZE  = K * K;

        // use max pooling to find maximum value in the kernel
        namespace F = torch::nn::functional;
        // std::cout << "kernel_map.sizes()"
        //           << kernel_map.sizes() << std::endl;
        // std::cout << "kernel_map.data_ptr()"
        //           << (uint64_t)kernel_map.data_ptr<float>() << std::endl;
        // std::cout << "kernel_map.unsqueeze(0)" << kernel_map.unsqueeze(0).sizes()
        //           << std::endl;
        auto max_map = F::max_pool2d(
            kernel_map,
            F::MaxPool2dFuncOptions(K).padding(support).stride(1));
        max_map = max_map.squeeze(0);  // [B, H, W]

        // buffer
        auto rgb_sum    = torch::zeros_like(imgs_out);
        auto kernel_sum = torch::zeros_like(max_map);

        // apply kernel
        const int applying_blocks = N_BLOCKS_NEEDED(SIZE * K_SIZE, N_THREADS);
        // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            kernel_map.type(), "forward_applying", ([&] {
                kernel::applying<scalar_t>
                    <<<applying_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        H,
                        W,
                        support,
                        kernel_map.data_ptr<scalar_t>(),
                        imgs_in.data_ptr<scalar_t>(),
                        max_map.data_ptr<scalar_t>(),
                        rgb_sum.data_ptr<scalar_t>(),
                        kernel_sum.data_ptr<scalar_t>());
            }));
        // clang-format on

        // normalize & accumulate
        const int normalize_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
        // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            kernel_map.type(), "forward_normalize", ([&] {
                kernel::normalize<scalar_t>
                    <<<normalize_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        weight_map.data_ptr<scalar_t>(),
                        rgb_sum.data_ptr<scalar_t>(),
                        kernel_sum.data_ptr<scalar_t>(),
                        imgs_out.data_ptr<scalar_t>());
            }));
        // clang-format on

        // save tensors
        if (!save.empty()) {
            int&& idx     = (level + 1) * 3;
            save[idx]     = rgb_sum.div_(kernel_sum.unsqueeze(-1));
            save[idx + 1] = max_map;
            save[idx + 2] = kernel_sum;
        }
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list      grad_outputs  
    ) {
        auto saved       = ctx->get_saved_variables();
        auto weight_map  = saved[0];  // [L, B, H, W]
        auto kernel_map  = saved[1];  // [L, B, H, W]
        auto imgs_in     = saved[2];  // [B, H, W, 3]
        auto grad_output = grad_outputs[0].contiguous();  // [B, H, W, 3]

        // get dimensions
        const int L    = kernel_map.size(0);
        const int B    = kernel_map.size(1);
        const int H    = kernel_map.size(2);
        const int W    = kernel_map.size(3);
        const int SIZE = B * H * W;

        // create buffer
        auto grad_weight = torch::zeros_like(weight_map);
        auto grad_kernel = torch::zeros_like(kernel_map);
        // std::cout << "grad_input" << grad_input << std::endl;

        // accumulate grads for each level
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        for (int level = 0; level < L; level++) {
            int&& save_idx = (level + 1) * 3;
            grad_accumulate_one_level(
                stream,
                level,
                SIZE,
                H,
                W,
                L,
                grad_output,
                imgs_in,
                saved[save_idx],             // rgb_sum_map
                saved[save_idx + 1],         // max_map
                saved[save_idx + 2],         // kernel_sum
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
        torch::Tensor grad_output,  // [B, H, W, 3]
        torch::Tensor imgs_in,      // [B, H, W, 3]
        torch::Tensor rgb_sum_map,  // [B, H, W, 3]
        torch::Tensor max_map,      // [B, H, W]
        torch::Tensor kernel_sum,   // [B, H, W]
        torch::Tensor weight_map,   // [B, H, W]
        torch::Tensor kernel_map,   // [B, H, W]
        torch::Tensor grad_weight,  // [B, H, W]
        torch::Tensor grad_kernel   // [B, H, W]
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
                        rgb_sum_map.data_ptr<scalar_t>(),
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
                        rgb_sum_map.data_ptr<scalar_t>(),
                        kernel_map.data_ptr<scalar_t>(),
                        max_map.data_ptr<scalar_t>(),
                        kernel_sum.data_ptr<scalar_t>(),
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
    torch::Tensor weight_map,  // [L, B, H, W]
    torch::Tensor kernel_map,  // [L, B, H, W]
    torch::Tensor imgs_in,     // [B, H, W, 3]
    bool          requires_grad) {
    return Filtering::apply(
        weight_map, kernel_map, imgs_in, requires_grad);
}

}  // namespace denoiser