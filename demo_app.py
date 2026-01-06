import gradio as gr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Cargar modelo y procesador
print("🔄 Cargando modelo...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("lucas-mella/llarri-spanish-htr-model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Modelo cargado en {device}")

def recognize_text(image):
    """Reconoce texto manuscrito en español"""
    if image is None:
        return "⚠️ Por favor sube una imagen"
    
    # Convertir a RGB si es necesario
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Preprocesar
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generar texto
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)
    
    # Decodificar
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text

# Interfaz Gradio
demo = gr.Interface(
    fn=recognize_text,
    inputs=gr.Image(type="pil", label="📷 Imagen de texto manuscrito"),
    outputs=gr.Textbox(label="📝 Texto reconocido", lines=3),
    title="🇪🇸 LLARRI - OCR Manuscrito Español",
    description="""
    **Reconocimiento de texto manuscrito en español**
    
    Sube una imagen con texto escrito a mano y el modelo intentará reconocerlo.
    
    📦 Modelo: [lucas-mella/llarri-spanish-htr-model](https://huggingface.co/lucas-mella/llarri-spanish-htr-model)
    """,
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
