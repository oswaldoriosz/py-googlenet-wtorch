#ifndef TORCH_IMAGE_PREPROCESSOR_HPP
#define TORCH_IMAGE_PREPROCESSOR_HPP

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace adapters {

class TorchImagePreprocessor {
public:
    void preprocess(pybind11::object image, const std::string& output_path);
};

} // namespace adapters

#endif // TORCH_IMAGE_PREPROCESSOR_HPP