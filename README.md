<p align="center">
  <img src="assets/social-preview.png" alt="Prototipo de Clasificación Infrarroja" width="100%"/>
</p>

# Clasificador de Calidad y Calibración en Imágenes Termográficas Infrarrojas (IRT)

Este repositorio contiene un prototipo de visión por computadora desarrollado en Python para evaluar automáticamente la calidad y calibración de capturas termográficas infrarrojas (IRT) mediante Transfer Learning. El sistema implementa una arquitectura agnóstica de hardware para discriminar si un sensor presenta calibración válida, saturación cromática o si la captura se encuentra fuera de dominio.

---

## 1. Alcance del Proyecto y Clases del Modelo

El problema se abarca como una clasificación supervisada multiclase distribuyendo las capturas en 3 categorías mutuamente excluyentes:

1. **`Termografia_Valida`**: Capturas térmicas conformes con gradiente cromático continuo y barra lateral de temperatura/gradiente medible.
2. **`Termografia_Descalibrada_Saturada`**: Muestras térmicas con pérdida de rango, paleta desajustada o saturación cromática que altera las isotermas.
3. **`Imagen_Fuera_de_Dominio_NA`**: Fotografías ópticas estándar, tomas borrosas o elementos ajenos al protocolo. Adopción de la clase de control (`Unqualified`).

---

## 2. Dataset Piloto y Hardware de Campo

* **Equipo de Inspección**: Cámara Termográfica Profesional (resolución infrarroja 160 × 120 píxeles, sensibilidad térmica < 0.05 °C).
* **Preprocesamiento**: Conversión desde el formato radiométrico del fabricante a formato abierto (`.bmp` / `.jpg`) y resolución a tensores de 224 × 224 px.

---

## 3. Fundamentación Teórica y Stack Técnico

* **Gestión de Entorno**: `uv` (`pyproject.toml`, `uv.lock`).
* **Framework de Aprendizaje**: PyTorch (`torch`, `torchvision`).
* **Modelo Base**: MobileNetV2 preentrenado en ImageNet.
* **Interfaz Web**: Gradio (`app.py`).

### Justificación de MobileNetV2 frente a Vision Transformers (ViT)
Se seleccionó MobileNetV2 por su sesgo inductivo de localidad espacial e invariancia a la traslación. Las capas convolucionales presuponen que los píxeles adyacentes forman bordes y gradientes térmicos contiguos, lo que permite generalizar con datasets reducidos de laboratorio sin caer en sobreajuste (*overfitting*), a diferencia de los Transformers (ViT) que requieren conjuntos masivos de datos para aprender relaciones espaciales.

---

## 4. Instalación y Ejecución

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/Rocdpi/clasificador-termografia.git](https://github.com/Rocdpi/clasificador-termografia.git)
   cd clasificador-termografia
