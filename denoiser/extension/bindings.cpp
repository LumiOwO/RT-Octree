#include <torch/extension.h>

#include "filtering.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "filtering_autograd",
        &denoiser::filtering_autograd,
        "Autograd filtering (CUDA)",
        py::arg("weight_map"),
        py::arg("kernel_map"),
        py::arg("imgs_in"),
        py::arg("requires_grad") = false);
}