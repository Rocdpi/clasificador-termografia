import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pillow_heif
import gradio as gr

# Soporte para fotos HEIC
pillow_heif.register_heif_opener()

# Configuracion de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases del modelo
CLASS_NAMES = ["Imagen_Fuera_de_Dominio_NA", "Termografia_Descalibrada_Saturada", "Termografia_Valida_OK"]

# Transformaciones de inferencia
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Carga de la arquitectura
def load_model(weights_path="model.pth"):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Aviso: No se encontro '{weights_path}'. Se utilizara la arquitectura base.")
        
    model.to(device)
    model.eval()
    return model

model = load_model("model.pth")

# Prediccion
def predict(image):
    if image is None:
        return {}
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    
    tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

# Interfaz Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Cargar Imagen"),
    outputs=gr.Label(num_top_classes=3, label="Resultado"),
    title="Clasificador de Imagenes Termograficas",
    description="Validacion y clasificacion de capturas termicas."
)

if __name__ == "__main__":
    demo.launch()
