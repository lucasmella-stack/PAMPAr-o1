#!/usr/bin/env python3
"""
demo_inference.py - Demostración de inferencia OCR con TrOCR pre-entrenado

Este script muestra cómo usar el modelo para reconocer texto en imágenes.
Funciona con el modelo pre-entrenado de Microsoft sin necesidad de entrenar.

Uso:
    # Con imagen de ejemplo generada:
    python scripts/demo_inference.py
    
    # Con tu propia imagen:
    python scripts/demo_inference.py --image path/to/image.png
    
    # Con carpeta de imágenes:
    python scripts/demo_inference.py --folder data/user_samples/images/
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_sample_image(text: str = "Hello World", output_path: str = None) -> Image.Image:
    """Crea una imagen de ejemplo con texto para probar OCR."""
    # Crear imagen blanca
    width, height = 400, 80
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Intentar usar una fuente manuscrita, si no usar default
    try:
        # Intentar fuentes comunes en Linux
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        font = None
        for fp in font_paths:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, 40)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Dibujar texto centrado
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill='black', font=font)
    
    if output_path:
        img.save(output_path)
        print(f"Imagen guardada en: {output_path}")
    
    return img


def load_pretrained_model():
    """Carga el modelo TrOCR pre-entrenado de Microsoft."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    
    print("Cargando modelo TrOCR pre-entrenado...")
    print("(Primera vez descargará ~1GB de pesos)")
    
    # Cargar processor (tokenizer + image processor)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Cargar modelo completo pre-entrenado
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Mover a GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Modelo cargado en: {device}")
    return model, processor, device


def run_inference(model, processor, image: Image.Image, device: str) -> str:
    """Ejecuta inferencia OCR en una imagen."""
    # Preprocesar imagen
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generar texto
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    
    # Decodificar
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def process_single_image(model, processor, device, image_path: str):
    """Procesa una sola imagen."""
    print(f"\n📷 Procesando: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    text = run_inference(model, processor, img, device)
    
    print(f"📝 Texto reconocido: '{text}'")
    return text


def process_folder(model, processor, device, folder_path: str):
    """Procesa todas las imágenes en una carpeta."""
    folder = Path(folder_path)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    
    if not images:
        print(f"No se encontraron imágenes en: {folder}")
        return []
    
    print(f"\n📁 Procesando {len(images)} imágenes de: {folder}")
    print("-" * 50)
    
    results = []
    for img_path in sorted(images):
        text = process_single_image(model, processor, device, str(img_path))
        results.append({"file": img_path.name, "text": text})
    
    return results


def demo_with_generated_images(model, processor, device):
    """Demostración con imágenes generadas."""
    print("\n" + "="*60)
    print("DEMO: Inferencia con imágenes generadas")
    print("="*60)
    
    test_texts = [
        "Hello World",
        "Testing OCR",
        "12345",
        "OpenAI GPT",
    ]
    
    results = []
    for original_text in test_texts:
        print(f"\n🎯 Texto original: '{original_text}'")
        
        # Crear imagen
        img = create_sample_image(original_text)
        
        # OCR
        recognized = run_inference(model, processor, img, device)
        print(f"📝 Texto reconocido: '{recognized}'")
        
        # Comparar
        match = "✅" if recognized.lower().strip() == original_text.lower().strip() else "❌"
        print(f"   {match} Match: {recognized.lower().strip() == original_text.lower().strip()}")
        
        results.append({
            "original": original_text,
            "recognized": recognized,
            "match": recognized.lower().strip() == original_text.lower().strip()
        })
    
    # Resumen
    correct = sum(1 for r in results if r["match"])
    print(f"\n📊 Resultados: {correct}/{len(results)} correctos")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Demo de inferencia OCR con TrOCR")
    parser.add_argument("--image", "-i", type=str, help="Ruta a una imagen para OCR")
    parser.add_argument("--folder", "-f", type=str, help="Carpeta con imágenes para OCR")
    parser.add_argument("--demo", "-d", action="store_true", help="Ejecutar demo con imágenes generadas")
    args = parser.parse_args()
    
    # Info del sistema
    print("="*60)
    print("LLARRI-OCR - Demo de Inferencia")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar modelo
    model, processor, device = load_pretrained_model()
    
    # Ejecutar según argumentos
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Error: No se encontró la imagen: {args.image}")
            sys.exit(1)
        process_single_image(model, processor, device, args.image)
        
    elif args.folder:
        if not Path(args.folder).exists():
            print(f"❌ Error: No se encontró la carpeta: {args.folder}")
            sys.exit(1)
        results = process_folder(model, processor, device, args.folder)
        
        # Guardar resultados
        output_file = Path(args.folder) / "ocr_results.txt"
        with open(output_file, "w") as f:
            for r in results:
                f.write(f"{r['file']}\t{r['text']}\n")
        print(f"\n💾 Resultados guardados en: {output_file}")
        
    else:
        # Demo por defecto
        demo_with_generated_images(model, processor, device)
    
    print("\n" + "="*60)
    print("✅ Demo completada")
    print("="*60)
    print("\nPara usar con tus propias imágenes:")
    print("  python scripts/demo_inference.py --image tu_imagen.png")
    print("  python scripts/demo_inference.py --folder data/user_samples/images/")


if __name__ == "__main__":
    main()
