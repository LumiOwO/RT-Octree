#include "filtering.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "filtering",
        &denoiser::filtering,
        "Autograd filtering (CUDA)",
        py::arg("input"),
        py::arg("imgs_in"),
        py::arg("imgs_out"),
        py::arg("requires_grad") = false);
}