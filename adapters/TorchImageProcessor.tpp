#include "TorchImageProcessor.hpp"
#include <torch/torch.h>
#include <stdexcept>

namespace adapters {

torch::Tensor TorchImageProcessor::preprocess(const std::string& tensor_path) {
    try {
        torch::Tensor tensor;
        torch::load(tensor, tensor_path);
        if (tensor.sizes() != torch::IntArrayRef({1, 3, 224, 224})) {
            throw std::runtime_error("Tensor tiene forma incorrecta: esperado [1, 3, 224, 224]");
        }
        return tensor;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error al cargar el tensor: " + std::string(e.what()));
    }
}

} // namespace adapters