#!/usr/bin/env python3
"""
train_on_huggingface.py - Entrenar modelo en Hugging Face Spaces/AutoTrain

Con HF Pro tienes acceso a:
- GPUs: T4, A10G, A100
- Hasta 8x A100 80GB para entrenamiento pesado
- AutoTrain para entrenamiento sin código
- Spaces con GPU dedicada

Opciones:
1. AutoTrain (más fácil) - UI web
2. Training Space (más control) - Script personalizado
3. API de entrenamiento (programático)
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, SpaceHardware

# Tu dataset
DATASET_REPO = "lucas-mella/llarri-spanish-htr"
MODEL_REPO = "lucas-mella/llarri-spanish-ocr"


def create_training_space():
    """Crea un Space de entrenamiento con GPU."""
    
    api = HfApi()
    
    # Código del Space para entrenamiento
    app_code = '''
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gradio as gr
import torch
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from PIL import Image
from huggingface_hub import login, HfApi
import threading

# Login con token del Space
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Variables globales
training_status = {"running": False, "progress": 0, "logs": []}
model = None
processor = None

def log(msg):
    training_status["logs"].append(msg)
    print(msg)
    if len(training_status["logs"]) > 100:
        training_status["logs"] = training_status["logs"][-50:]

def start_training(epochs, batch_size, learning_rate, max_samples):
    global model, processor, training_status
    
    if training_status["running"]:
        return "⚠️ Entrenamiento ya en curso"
    
    training_status["running"] = True
    training_status["progress"] = 0
    training_status["logs"] = []
    
    try:
        log("🚀 Iniciando entrenamiento LLARRI...")
        log(f"📊 Config: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
        
        # Verificar GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            log(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            log(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            log("⚠️ Sin GPU, entrenamiento será lento")
        
        # Cargar dataset
        log("📂 Cargando dataset...")
        dataset = load_dataset("lucas-mella/llarri-spanish-htr")
        
        if max_samples and max_samples > 0:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            log(f"   Usando {len(dataset['train'])} muestras de entrenamiento")
        
        # Cargar modelo
        log("🤖 Cargando TrOCR...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Configurar
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.max_length = 64
        model.config.num_beams = 4
        
        # Preprocesar
        log("🔄 Preprocesando datos...")
        
        def preprocess(examples):
            images = [Image.open(img).convert("RGB") for img in examples["image"]]
            pixel_values = processor(images, return_tensors="pt").pixel_values
            
            labels = processor.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            return {"pixel_values": pixel_values, "labels": labels}
        
        train_dataset = dataset["train"].map(
            preprocess, 
            batched=True, 
            batch_size=32,
            remove_columns=["image", "text"]
        )
        
        eval_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"])))).map(
            preprocess,
            batched=True,
            batch_size=32,
            remove_columns=["image", "text"]
        )
        
        log(f"   Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
        
        # Argumentos de entrenamiento
        training_args = Seq2SeqTrainingArguments(
            output_dir="./llarri-output",
            num_train_epochs=int(epochs),
            per_device_train_batch_size=int(batch_size),
            per_device_eval_batch_size=int(batch_size),
            gradient_accumulation_steps=4,
            learning_rate=float(learning_rate),
            warmup_ratio=0.1,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            fp16=torch.cuda.is_available(),
            push_to_hub=True,
            hub_model_id="lucas-mella/llarri-spanish-ocr",
            hub_token=HF_TOKEN,
            report_to="none",
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )
        
        # Entrenar
        log("🏋️ Entrenando...")
        trainer.train()
        
        # Guardar
        log("💾 Guardando modelo...")
        trainer.push_to_hub()
        processor.push_to_hub("lucas-mella/llarri-spanish-ocr")
        
        log("✅ ¡Entrenamiento completado!")
        log(f"🔗 Modelo en: https://huggingface.co/lucas-mella/llarri-spanish-ocr")
        
        training_status["progress"] = 100
        
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        training_status["running"] = False
    
    return "\\n".join(training_status["logs"])

def get_status():
    return "\\n".join(training_status["logs"][-20:])

def test_inference(image):
    global model, processor
    
    if model is None or processor is None:
        # Cargar modelo entrenado
        try:
            processor = TrOCRProcessor.from_pretrained("lucas-mella/llarri-spanish-ocr")
            model = VisionEncoderDecoderModel.from_pretrained("lucas-mella/llarri-spanish-ocr")
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
        except:
            return "Modelo no disponible. Entrena primero."
    
    if image is None:
        return "Sube una imagen"
    
    # Inferencia
    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# UI
with gr.Blocks(title="LLARRI Training") as demo:
    gr.Markdown("# 🚀 LLARRI - Entrenamiento OCR Español")
    gr.Markdown("Entrena TrOCR con tu dataset de escritura en español")
    
    with gr.Tab("🏋️ Entrenar"):
        with gr.Row():
            with gr.Column():
                epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                learning_rate = gr.Number(value=5e-5, label="Learning Rate")
                max_samples = gr.Number(value=50000, label="Max Samples (0=todos)")
                
                train_btn = gr.Button("🚀 Iniciar Entrenamiento", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(label="Logs", lines=20, max_lines=30)
                refresh_btn = gr.Button("🔄 Actualizar Logs")
        
        train_btn.click(
            start_training,
            inputs=[epochs, batch_size, learning_rate, max_samples],
            outputs=output
        )
        refresh_btn.click(get_status, outputs=output)
    
    with gr.Tab("🔍 Probar"):
        gr.Markdown("Prueba el modelo entrenado")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Imagen de texto")
            output_text = gr.Textbox(label="Texto reconocido")
        
        test_btn = gr.Button("Reconocer texto")
        test_btn.click(test_inference, inputs=input_image, outputs=output_text)

demo.launch()
'''
    
    requirements = """
gradio>=4.0.0
transformers>=4.35.0
datasets>=2.14.0
torch>=2.0.0
accelerate>=0.24.0
Pillow
huggingface_hub
hf_transfer
"""
    
    # Crear repo del Space
    space_id = "lucas-mella/llarri-trainer"
    
    print(f"\n📦 Creando Space: {space_id}")
    
    try:
        create_repo(
            space_id,
            repo_type="space",
            space_sdk="gradio",
            private=True,
            exist_ok=True,
        )
        print("✅ Space creado")
    except Exception as e:
        print(f"⚠️ {e}")
    
    # Subir archivos
    print("📤 Subiendo archivos...")
    
    # app.py
    upload_file(
        path_or_fileobj=app_code.encode(),
        path_in_repo="app.py",
        repo_id=space_id,
        repo_type="space",
    )
    
    # requirements.txt
    upload_file(
        path_or_fileobj=requirements.encode(),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
    )
    
    # README
    readme = f"""---
title: LLARRI Trainer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# LLARRI - Entrenador de OCR en Español

Entrena TrOCR con el dataset de escritura manuscrita en español.

## Uso

1. Ve a la pestaña "Entrenar"
2. Configura epochs, batch size, etc.
3. Click en "Iniciar Entrenamiento"
4. El modelo se guardará en `lucas-mella/llarri-spanish-ocr`

## Dataset

Usa el dataset privado: `lucas-mella/llarri-spanish-htr`
"""
    
    upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
    )
    
    print("✅ Archivos subidos")
    
    # Configurar GPU
    print("\n⚙️ Configurando hardware...")
    
    try:
        api.request_space_hardware(
            repo_id=space_id,
            hardware=SpaceHardware.T4_MEDIUM,  # GPU T4 (incluida en Pro)
        )
        print("✅ GPU T4 asignada")
    except Exception as e:
        print(f"⚠️ No se pudo asignar GPU automáticamente: {e}")
        print("   Ve a Settings del Space y selecciona GPU manualmente")
    
    # Agregar secreto del token
    print("\n🔐 Configurando token...")
    try:
        api.add_space_secret(
            repo_id=space_id,
            key="HF_TOKEN",
            value=os.environ.get("HF_TOKEN", ""),
        )
        print("✅ Token configurado")
    except Exception as e:
        print(f"⚠️ Configura HF_TOKEN manualmente en Settings > Secrets")
    
    print("\n" + "="*60)
    print("✅ SPACE DE ENTRENAMIENTO CREADO")
    print("="*60)
    print(f"🔗 URL: https://huggingface.co/spaces/{space_id}")
    print(f"\n📋 Pasos:")
    print(f"   1. Ve al Space")
    print(f"   2. Settings > Variables and secrets > New secret")
    print(f"      - Name: HF_TOKEN")
    print(f"      - Value: tu token de HF con permisos write")
    print(f"   3. Settings > Hardware > Selecciona T4 o A10G")
    print(f"   4. ¡Listo! Usa la UI para entrenar")
    
    return space_id


def show_options():
    """Muestra opciones de entrenamiento en HF."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     🚀 OPCIONES DE ENTRENAMIENTO EN HUGGING FACE PRO         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. AutoTrain (Más fácil - UI web)                          ║
║     → https://huggingface.co/autotrain                      ║
║     → Selecciona "Image to Text"                            ║
║     → Usa tu dataset: lucas-mella/llarri-spanish-htr        ║
║                                                              ║
║  2. Training Space (Este script)                             ║
║     → Crea un Space con Gradio + GPU                        ║
║     → Control total sobre el entrenamiento                   ║
║     → Ejecuta: python train_on_huggingface.py --create      ║
║                                                              ║
║  3. Notebooks en HF                                          ║
║     → https://huggingface.co/spaces/notebooks               ║
║     → Jupyter con GPU T4/A10G                               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  💰 GPUs disponibles con Pro:                                ║
║     • T4 (16GB)    - Gratis 2hr/día, luego ~$0.60/hr        ║
║     • A10G (24GB)  - ~$1.05/hr                              ║
║     • A100 (40GB)  - ~$4.13/hr                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Crear Space de entrenamiento")
    parser.add_argument("--options", action="store_true", help="Mostrar opciones")
    
    args = parser.parse_args()
    
    if args.create:
        create_training_space()
    else:
        show_options()
