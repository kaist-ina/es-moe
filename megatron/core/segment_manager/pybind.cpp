
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "local_segment_manager.h"

namespace py = pybind11;

PYBIND11_MODULE(segment_manager, m) {
    m.doc() = "pybind11 global segment implementation";
    m.attr("__version__") = "0.0.1";

    m.def("is_initialized", [] () {
        return LocalSegmentManager::getInstance().is_initialized();
    });

    m.def("initialize", [] (int rank, py::dict options) {
        LocalSegmentManager::getInstance().initialize(rank);
    });

    m.def("pre_forward_hook", [] (int layer, int expert) {
        LocalSegmentManager::getInstance().pre_forward_hook(layer, expert);
    });

    m.def("post_forward_hook", [] (int layer, int expert, uint64_t stream_uint_ptr) {
        LocalSegmentManager::getInstance().post_forward_hook(layer, expert, reinterpret_cast<cudaStream_t>(stream_uint_ptr));
    });

    m.def("pre_backward_hook", [] (int layer, int expert) {
        LocalSegmentManager::getInstance().pre_backward_hook(layer, expert);
    });

    m.def("post_backward_hook", [] (int layer, int expert, uint64_t stream_uint_ptr) {
        LocalSegmentManager::getInstance().post_backward_hook(layer, expert, reinterpret_cast<cudaStream_t>(stream_uint_ptr));
    });
    
    m.def("pre_optimize_hook", [] (int layer, int expert) {
        LocalSegmentManager::getInstance().pre_optimize_hook(layer, expert);
    });

    m.def("post_optimize_hook", [] (int layer, int expert) {
        LocalSegmentManager::getInstance().post_optimize_hook(layer, expert);
    });

    m.def("shared_pinned_memory", [] (torch::Tensor tensor, int rank, int layer, int expert, int order, int type, bool pinning = true) {
        return LocalSegmentManager::getInstance().shared_pinned_memory(tensor, rank, layer, expert, order, type, pinning);
    });
}