#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "domain/GoogLeNetService.hpp"
#include "adapters/TorchModelLoader.hpp"
#include "adapters/TorchImageProcessor.hpp"
#include "adapters/TorchInference.hpp"
#include "adapters/TorchImagePreprocessor.hpp"

namespace py = pybind11;

// Forzar la generación de la vtable
namespace adapters {
    void force_vtable_generation() {
        TorchImageProcessor dummy_processor;
        TorchModelLoader dummy_loader;
        TorchInference dummy_inference;
        TorchImagePreprocessor dummy_preprocessor;
    }
}

PYBIND11_MODULE(googlenet, m) {
    // Llamar a force_vtable_generation para evitar eliminación de código
    adapters::force_vtable_generation();

    py::class_<domain::GoogLeNetService>(m, "GoogLeNetService")
        .def(py::init([](const std::string& /* dummy */) {
            return std::make_unique<domain::GoogLeNetService>(
                std::make_unique<adapters::TorchModelLoader>(),
                std::make_unique<adapters::TorchImageProcessor>(),
                std::make_unique<adapters::TorchInference>()
            );
        }))
        .def("classify", &domain::GoogLeNetService::classify);

    m.def("preprocess_image", [](py::object image, const std::string& output_path) {
        adapters::TorchImagePreprocessor preprocessor;
        preprocessor.preprocess(image, output_path);
    }, "Preprocesa una imagen PIL y guarda el tensor en el archivo especificado");
}
