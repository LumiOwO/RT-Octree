#include "filtering.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filtering", &denoiser::filtering, "Autograd filtering (CUDA)");
}