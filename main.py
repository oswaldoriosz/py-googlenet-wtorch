from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import googlenet
import json
import shutil
import os

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Directorio del PVC montado en OpenShift
UPLOAD_FOLDER = "/opt/app-root/src/models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Preprocesa una imagen para el modelo GoogleNet usando C++."""
    try:
        # Cargar la imagen
        image = Image.open(image_path).convert("RGB")

        # Llamar al método C++ para preprocesar y guardar el tensor
        tensor_path = os.path.join(UPLOAD_FOLDER, "preprocessed_tensor.pt")
        googlenet.preprocess_image(image, tensor_path)
        return tensor_path
    except Exception as e:
        raise Exception(f"Error al preprocesar la imagen: {str(e)}")

def load_imagenet_labels():
    """Carga las etiquetas de ImageNet desde un archivo o genera etiquetas simuladas."""
    labels_path = os.path.join(UPLOAD_FOLDER, "imagenet_classes.txt")
    try:
        with open(labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Advertencia: {labels_path} no encontrado. Usando etiquetas simuladas.")
        return [f"simulated_class_{i}" for i in range(1000)]

@app.post("/pre-procesar-imagen")
async def preprocesar(file: UploadFile = File(...)):
    """Procesa una imagen subida y guarda el tensor preprocesado."""
    try:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        preprocess_image(file_location)
        return {"filename": file.filename, "message": "Imagen procesada exitosamente"}
    except Exception as e:
        return {"error": f"Error al procesar la imagen: {str(e)}"}

@app.post("/clasificador")
def calculo():
    """Realiza la clasificación con GoogleNet y genera una gráfica de resultados."""
    output_file = os.path.join(UPLOAD_FOLDER, 'googlenet_results.png')
    model_path = os.path.join(UPLOAD_FOLDER, "googlenet.pt")
    tensor_path = os.path.join(UPLOAD_FOLDER, "preprocessed_tensor.pt")
    
    try:
        # Verificar que los archivos necesarios existan
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Tensor preprocesado no encontrado: {tensor_path}")

        class_names = load_imagenet_labels()
        service = googlenet.GoogLeNetService("clasificador")
        results = service.classify(model_path, tensor_path, class_names)
        
        print("Top-5 predicciones:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")
        
        labels = [label for label, prob in results]
        probs = [prob for label, prob in results]
        
        # Usar un colormap para las barras
        colors = cm.viridis([i/len(probs) for i in range(len(probs))])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, probs, color=colors, edgecolor='black')
        plt.title("Top-5 Predicciones de GoogLeNet", fontsize=14, pad=15)
        plt.xlabel("Clase", fontsize=12)
        plt.ylabel("Probabilidad", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        response = {"Grafica generada": output_file}
        return json.dumps(response)
    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return {"error": str(e)}

@app.get("/googlenet-graph")
def getGraph():
    """Devuelve la gráfica generada como imagen PNG."""
    output_file = os.path.join(UPLOAD_FOLDER, 'googlenet_results.png')
    if not os.path.exists(output_file):
        return {"error": f"Gráfica no encontrada: {output_file}"}
    return FileResponse(output_file, media_type="image/png", filename=os.path.basename(output_file))
