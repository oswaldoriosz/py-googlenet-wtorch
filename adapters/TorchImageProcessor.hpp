#ifndef TORCH_IMAGE_PROCESSOR_HPP
#define TORCH_IMAGE_PROCESSOR_HPP

#include "../ports/ImageProcessorPort.hpp"
#include <torch/torch.h>
#include <stdexcept>
#include <string>

namespace adapters {

class TorchImageProcessor : public ports::ImageProcessorPort {
public:
    torch::Tensor preprocess(const std::string& tensor_path) override {
        try {
            // Abrir el archivo como flujo binario
            std::ifstream stream(tensor_path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("No se pudo abrir el archivo: " + tensor_path);
            }

            // Cargar el tensor directamente con torch::load
            torch::Tensor tensor;
            torch::load(tensor, tensor_path);

            // Verificar que el tensor sea válido
            if (!tensor.defined()) {
                throw std::runtime_error("El archivo no contiene un tensor válido");
            }

            // Verificar la forma del tensor
            if (tensor.sizes() != torch::IntArrayRef({1, 3, 224, 224})) {
                throw std::runtime_error("Tensor tiene forma incorrecta: esperado [1, 3, 224, 224], encontrado " +
                                        std::to_string(tensor.sizes()[0]) + "," +
                                        std::to_string(tensor.sizes()[1]) + "," +
                                        std::to_string(tensor.sizes()[2]) + "," +
                                        std::to_string(tensor.sizes()[3]));
            }

            return tensor.to(torch::kCPU);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error al cargar el tensor: " + std::string(e.what()));
        }
    }
};

} // namespace adapters

#endif // TORCH_IMAGE_PROCESSOR_HPP