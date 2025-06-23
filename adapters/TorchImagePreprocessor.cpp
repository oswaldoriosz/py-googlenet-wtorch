#include "TorchImagePreprocessor.hpp"
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <fstream> // Agregar esta cabecera para std::ofstream

namespace py = pybind11;

namespace adapters {

void TorchImagePreprocessor::preprocess(py::object image, const std::string& output_path) {
    try {
        // Extraer datos de la imagen PIL (RGB)
        py::object getdata = image.attr("getdata");
        py::list data = getdata();
        int width = py::cast<int>(image.attr("width"));
        int height = py::cast<int>(image.attr("height"));

        // Verificar que la imagen sea RGB
        if (py::len(data) != width * height) {
            throw std::runtime_error("La imagen no está en formato RGB");
        }

        // Convertir los datos a un vector de floats (RGB)
        std::vector<float> pixels;
        pixels.reserve(width * height * 3);
        for (auto item : data) {
            py::tuple rgb = py::cast<py::tuple>(item);
            pixels.push_back(py::cast<float>(rgb[0]) / 255.0f); // R
            pixels.push_back(py::cast<float>(rgb[1]) / 255.0f); // G
            pixels.push_back(py::cast<float>(rgb[2]) / 255.0f); // B
        }

        // Crear tensor [H, W, C]
        torch::Tensor tensor = torch::from_blob(pixels.data(), {height, width, 3}, torch::kFloat32);
        tensor = tensor.permute({2, 0, 1}); // [C, H, W]

        // Transformación: Resize a 256
        tensor = torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{256, 256}).mode(torch::kBilinear)
        ).squeeze(0);

        // Transformación: CenterCrop a 224
        int crop_size = 224;
        int h_offset = (256 - crop_size) / 2;
        int w_offset = (256 - crop_size) / 2;
        tensor = tensor.slice(1, h_offset, h_offset + crop_size).slice(2, w_offset, w_offset + crop_size);

        // Transformación: Normalizar
        torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
        torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
        tensor = (tensor - mean) / std;

        // Añadir dimensión batch: [1, C, H, W]
        tensor = tensor.unsqueeze(0);

        // Asegurar que el tensor esté en CPU
        tensor = tensor.to(torch::kCPU);

        // Guardar el tensor
        std::ofstream stream(output_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("No se pudo abrir el archivo para escritura: " + output_path);
        }
        torch::save(tensor, stream);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error al preprocesar la imagen: " + std::string(e.what()));
    }
}

} // namespace adapters