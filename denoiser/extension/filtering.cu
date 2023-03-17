#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <ATen/cuda/Atomic.cuh>

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

template <typename scalar_t>
__global__ void weights_softmax(const int SIZE, const int L, scalar_t* input) {
    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    input += idx * (L << 1);

    scalar_t max_val = input[0];
    for (int i = 1; i < L; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    scalar_t sum = 0;
    for (int i = 0; i < L; i++) {
        input[i] = __expf(input[i] - max_val);
        sum += input[i];
    }

    const scalar_t inv_sum = 1 / sum;
    for (int i = 0; i < L; i++) {
        input[i] *= inv_sum;
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
    scalar_t* kernel_sum                      // [B, H, W, 1]
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
    imgs_in += neighbor_idx * 3;
    max_map += pixel_idx;
    rgb_sum += pixel_idx * 3;
    kernel_sum += pixel_idx;

    // accumulate
    const scalar_t& k = __expf(kernel_map[0] - max_map[0]);
    gpuAtomicAdd(&rgb_sum[0], k * imgs_in[0]);
    gpuAtomicAdd(&rgb_sum[1], k * imgs_in[1]);
    gpuAtomicAdd(&rgb_sum[2], k * imgs_in[2]);
    gpuAtomicAdd(kernel_sum, k);
}

template <typename scalar_t>
__global__ void normalize(
    const int SIZE,
    const scalar_t* __restrict__ weight_map,  // [B, H, W]
    const scalar_t* __restrict__ rgb_sum,     // [B, H, W, 3]
    const scalar_t* __restrict__ kernel_sum,  // [B, H, W, 1]
    scalar_t* imgs_out                        // [B, H, W, 3]
) {

    // locate
    CUDA_GET_THREAD_ID(idx, SIZE);
    weight_map += idx;
    rgb_sum += idx * 3;
    kernel_sum += idx;
    imgs_out += idx * 3;

    // accumulate
    const scalar_t& k = weight_map[0] / kernel_sum[0];
    gpuAtomicAdd(&imgs_out[0], k * rgb_sum[0]);
    gpuAtomicAdd(&imgs_out[1], k * rgb_sum[1]);
    gpuAtomicAdd(&imgs_out[2], k * rgb_sum[2]);
}

}  // namespace kernel

class Filtering : public torch::autograd::Function<Filtering> {
public:
    constexpr static const int N_THREADS = 512;

    // forward
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor                     input,    // [B, H, W, L * 2]
        torch::Tensor                     imgs_in,  // [B, H, W, 3]
        torch::Tensor                     imgs_out  // [B, H, W, 3]
    ) {
        const int B        = input.size(0);
        const int H        = input.size(1);
        const int W        = input.size(2);
        const int L        = input.size(3) >> 1;
        const int SIZE     = B * H * W;

        // init
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        cudaMemset(
            imgs_out.data_ptr(), 0,
            imgs_out.numel() * torch::elementSize(torch::typeMetaToScalarType(
                                   imgs_out.dtype())));

        // weight normalization
        const int weights_softmax_blocks = N_BLOCKS_NEEDED(SIZE, N_THREADS);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "forward_weights_softmax", ([&] {
                kernel::weights_softmax<scalar_t>
                    <<<weights_softmax_blocks, N_THREADS, 0, stream>>>(
                        SIZE, L, input.data_ptr<scalar_t>());
            }));
        ctx->save_for_backward({input, imgs_in});

        // permute
        input = input.permute({3, 0, 1, 2}).contiguous();  // [L * 2, B, H, W]

        // applying & fusing
        for (int level = 0; level < L; level++) {
            accumulate_one_level(
                stream, level, SIZE, H, W, L, input, imgs_in, imgs_out);
        }

        return imgs_out;  // [B, H, W, 3]
    }

    static void accumulate_one_level(
        cudaStream_t  stream,
        const int     level,
        const int     SIZE,
        const int     H,
        const int     W,
        const int     L,
        torch::Tensor input,
        torch::Tensor imgs_in,
        torch::Tensor imgs_out) {

        // kernel size
        const int support = level + 1;
        const int K       = 1 + (support << 1);
        const int K_SIZE  = K * K;

        // use max pooling to find maximum value in the kernel
        // max_map [B, H, W]
        namespace F = torch::nn::functional;
        auto max_map = F::max_pool2d(
            input.index({level + L}),
            F::MaxPool2dFuncOptions(K).padding(support).stride(1));

        // buffer
        torch::TensorOptions tensor_options =
            torch::TensorOptions().dtype(input.dtype()).device(torch::kCUDA);
        auto rgb_sum    = torch::zeros_like(imgs_out, tensor_options);
        auto kernel_sum = torch::zeros_like(max_map, tensor_options);

        // apply kernel
        const int applying_blocks = N_BLOCKS_NEEDED(SIZE * K_SIZE, N_THREADS);
        // clang-format off
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "forward_applying", ([&] {
                kernel::applying<scalar_t>
                    <<<applying_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        H,
                        W,
                        support,
                        input.data_ptr<scalar_t>() + (level + L) * SIZE,  // kernel map
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
            input.type(), "forward_normalize", ([&] {
                kernel::normalize<scalar_t>
                    <<<normalize_blocks, N_THREADS, 0, stream>>>(
                        SIZE,
                        input.data_ptr<scalar_t>() + level * SIZE,        // weight map
                        rgb_sum.data_ptr<scalar_t>(),
                        kernel_sum.data_ptr<scalar_t>(),
                        imgs_out.data_ptr<scalar_t>());
            }));
        // clang-format on
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list      grad_outputs) {
        auto saved       = ctx->get_saved_variables();
        auto input       = saved[0]; // with normalized weights
        auto imgs_in     = saved[1];

        // auto grad_output = grad_outputs[0];
        // auto grad_input  = grad_output.mm(weight);
        // auto grad_weight = grad_output.t().mm(input);
        // auto grad_bias   = torch::Tensor();
        // if (bias.defined()) {
        //     grad_bias = grad_output.sum(0);
        // }

        return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

void filtering(
    torch::Tensor input,
    torch::Tensor imgs_in,
    torch::Tensor imgs_out) {
    
    Filtering::apply(input, imgs_in, imgs_out);
}

}  // namespace denoiser