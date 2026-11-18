import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pillow_heif
import gradio as gr

# Soporte para fotos HEIC
pillow_heif.register_heif_opener()

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases del modelo
CLASS_NAMES = [
    "Imagen_Fuera_de_Dominio_NA",
    "Termografia_Descalibrada_Saturada",
    "Termografia_Valida_OK"
]

# Transformaciones de inferencia para MobileNetV2
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Carga de la arquitectura MobileNetV2
def load_model(weights_path="model.pth"):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Pesos cargados exitosamente desde '{weights_path}'.")
    else:
        print(f"Aviso: No se encontró '{weights_path}'. Se utilizará la arquitectura base.")
        
    model.to(device)
    model.eval()
    return model

model = load_model("model.pth")

# Función de degradación sintética para pruebas de control de fallas
def simular_camara_averiada(image_input, tipo_falla="saturacion_extrema", offset_saturacion=140):
    """
    Modifica una termografía para inducir síntomas característicos de falla
    física en el microbolómetro o fallas de corrección de no uniformidad (NUC).
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input, dtype=np.float32)
    else:
        img_np = image_input.astype(np.float32)

    h, w, c = img_np.shape

    if tipo_falla == "saturacion_extrema":
        # Desfase térmico global y saturación
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] + offset_saturacion, 0, 255)
        img_np[:, :, 1] = np.clip(img_np[:, :, 1] + (offset_saturacion * 0.35), 0, 255)

    elif tipo_falla == "falla_nuc":
        # Gradiente parásito continuo por fallo en obturador
        gradiente = np.linspace(0, 110, w).reshape(1, w, 1)
        img_np = np.clip(img_np + gradiente, 0, 255)

    elif tipo_falla == "pixeles_muertos":
        # Ruido de patrón fijo (líneas y píxeles ciegos)
        for _ in range(6):
            row = np.random.randint(0, h)
            img_np[row, :, :] = [255, 255, 0]
        num_muertos = int(h * w * 0.04)
        ys = np.random.randint(0, h, num_muertos)
        xs = np.random.randint(0, w, num_muertos)
        img_np[ys, xs] = [255, 0, 0]

    img_out = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_out)

# Pipeline de inferencia y emisión de dictamen
def predict(image, threshold, modo_simulacion):
    if image is None:
        empty_box = """
        <div style="background-color: #242424; border: 1px solid #444; border-radius: 8px; padding: 16px; color: #ccc;">
            ⚠️ <b>Sin entrada:</b> Cargue una captura térmica para iniciar el diagnóstico de calibración.
        </div>
        """
        return None, {}, empty_box

    # Aplicar alteración física si se seleccionó en el modo de prueba
    if modo_simulacion != "Ninguna (Normal)":
        if modo_simulacion == "Saturación Térmica / Offset":
            image_procesada = simular_camara_averiada(image, tipo_falla="saturacion_extrema", offset_saturacion=150)
        elif modo_simulacion == "Falla Obturador NUC (Gradiente)":
            image_procesada = simular_camara_averiada(image, tipo_falla="falla_nuc")
        elif modo_simulacion == "Píxeles Muertos / FPN":
            image_procesada = simular_camara_averiada(image, tipo_falla="pixeles_muertos")
        else:
            image_procesada = image
    else:
        image_procesada = image

    # Preprocesamiento de tensor
    if not isinstance(image_procesada, Image.Image):
        image_procesada = Image.fromarray(image_procesada)
    image_rgb = image_procesada.convert("RGB")
    
    tensor = inference_transforms(image_rgb).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    best_idx = torch.argmax(probabilities).item()
    best_class = CLASS_NAMES[best_idx]
    best_confidence = float(probabilities[best_idx])
    threshold_decimal = threshold / 100.0

    # Construcción de tarjeta de reporte técnico
    if best_class == "Termografia_Valida_OK" and best_confidence >= threshold_decimal:
        status_box = f"""
        <div style="background-color: #0b2e17; border: 2px solid #2ecc71; border-radius: 8px; padding: 18px; color: #ffffff; font-family: system-ui, sans-serif;">
            <div style="font-size: 1.25em; font-weight: bold; color: #2ecc71; margin-bottom: 8px;">
                ✅ DICTAMEN: CÁMARA OPERATIVA Y CALIBRADA
            </div>
            <div style="font-size: 1.0em; line-height: 1.6; margin-bottom: 12px;">
                • <b>Clase detectada:</b> <code>{best_class}</code><br>
                • <b>Confianza obtenida:</b> <span style="color: #2ecc71; font-weight: bold;">{best_confidence * 100:.2f}%</span><br>
                • <b>Umbral exigido:</b> {threshold:.1f}%
            </div>
            <div style="background-color: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ecc71; padding: 10px 14px; border-radius: 4px; font-size: 0.95em;">
                <b>Conclusión técnica:</b> La imagen muestra que <b>la cámara cumple con los criterios técnicos que avalan un buen funcionamiento</b>, preservando contraste adecuado, sensibilidad en rango dinámico y calibración térmica operativa.
            </div>
        </div>
        """
    elif best_class == "Termografia_Valida_OK" and best_confidence < threshold_decimal:
        status_box = f"""
        <div style="background-color: #332600; border: 2px solid #f39c12; border-radius: 8px; padding: 18px; color: #ffffff; font-family: system-ui, sans-serif;">
            <div style="font-size: 1.25em; font-weight: bold; color: #f39c12; margin-bottom: 8px;">
                ⚠️ DICTAMEN: VALIDACIÓN TÉCNICA AMBIGUA
            </div>
            <div style="font-size: 1.0em; line-height: 1.6; margin-bottom: 12px;">
                • <b>Clase detectada:</b> <code>{best_class}</code><br>
                • <b>Confianza obtenida:</b> <span style="color: #f39c12; font-weight: bold;">{best_confidence * 100:.2f}%</span><br>
                • <b>Umbral exigido:</b> {threshold:.1f}%
            </div>
            <div style="background-color: rgba(243, 156, 18, 0.15); border-left: 4px solid #f39c12; padding: 10px 14px; border-radius: 4px; font-size: 0.95em;">
                <b>Observación:</b> Aunque la toma se asemeja a una termografía estándar, la certeza no supera el umbral configurado ({threshold:.1f}%). Se sugiere repetir el disparo ajustando enfoque manual y distancia de medición.
            </div>
        </div>
        """
    elif best_class == "Termografia_Descalibrada_Saturada":
        status_box = f"""
        <div style="background-color: #380d0d; border: 2px solid #e74c3c; border-radius: 8px; padding: 18px; color: #ffffff; font-family: system-ui, sans-serif;">
            <div style="font-size: 1.25em; font-weight: bold; color: #e74c3c; margin-bottom: 8px;">
                ❌ DICTAMEN: CÁMARA FUERA DE SERVICIO / DESCALIBRADA
            </div>
            <div style="font-size: 1.0em; line-height: 1.6; margin-bottom: 12px;">
                • <b>Clase detectada:</b> <code>{best_class}</code><br>
                • <b>Confianza de falla:</b> <span style="color: #e74c3c; font-weight: bold;">{best_confidence * 100:.2f}%</span>
            </div>
            <div style="background-color: rgba(231, 76, 60, 0.15); border-left: 4px solid #e74c3c; padding: 10px 14px; border-radius: 4px; font-size: 0.95em;">
                <b>Criterio de mantenimiento:</b> Se detecta <b>saturación térmica destructiva o asimetría grave en el sensor</b>. El dispositivo debe ser retirado de operación y enviado a calibración técnica certificada.
            </div>
        </div>
        """
    else:
        status_box = f"""
        <div style="background-color: #1e1e1e; border: 2px solid #7f8c8d; border-radius: 8px; padding: 18px; color: #ffffff; font-family: system-ui, sans-serif;">
            <div style="font-size: 1.25em; font-weight: bold; color: #bdc3c7; margin-bottom: 8px;">
                🚫 DICTAMEN: IMAGEN FUERA DE DOMINIO
            </div>
            <div style="font-size: 1.0em; line-height: 1.6; margin-bottom: 12px;">
                • <b>Clase detectada:</b> <code>{best_class}</code><br>
                • <b>Confianza:</b> {best_confidence * 100:.2f}%
            </div>
            <div style="background-color: rgba(127, 140, 141, 0.15); border-left: 4px solid #7f8c8d; padding: 10px 14px; border-radius: 4px; font-size: 0.95em;">
                <b>Detalle:</b> El archivo analizado no presenta características de radiometría o espectro infrarrojo procesable.
            </div>
        </div>
        """

    return image_procesada, confidences, status_box

# Configuración de interfaz Gradio
with gr.Blocks(title="Validador y Clasificador de Cámaras Termográficas") as demo:
    gr.Markdown("# Clasificador y Validador de Cámaras Termográficas")
    gr.Markdown("Inspección de calidad de capturas térmicas para comprobar operatividad del sensor y criterios de calibración.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Cargar Captura Térmica")
            threshold_slider = gr.Slider(
                minimum=50.0,
                maximum=99.0,
                value=85.0,
                step=1.0,
                label="Umbral de Confianza Requerido (%)"
            )
            sim_dropdown = gr.Dropdown(
                choices=[
                    "Ninguna (Normal)",
                    "Saturación Térmica / Offset",
                    "Falla Obturador NUC (Gradiente)",
                    "Píxeles Muertos / FPN"
                ],
                value="Ninguna (Normal)",
                label="Simulación de Fallas de Campo (Demostración de Mantenimiento)"
            )
            submit_btn = gr.Button("Validar Termografía", variant="primary")

        with gr.Column(scale=1):
            processed_image_view = gr.Image(type="pil", label="Imagen Analizada por el Modelo")
            label_output = gr.Label(num_top_classes=3, label="Distribución de Probabilidades")
            report_box = gr.HTML(label="Certificado de Operatividad")

    submit_btn.click(
        fn=predict,
        inputs=[input_image, threshold_slider, sim_dropdown],
        outputs=[processed_image_view, label_output, report_box]
    )

if __name__ == "__main__":
    demo.launch()