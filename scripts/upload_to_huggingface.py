#!/usr/bin/env python3
"""
upload_to_huggingface.py - Sube el dataset a Hugging Face Hub

Uso:
    # Login primero (solo una vez):
    huggingface-cli login
    
    # Luego ejecutar:
    python scripts/upload_to_huggingface.py --dataset spanish_synthetic_4gb --repo tu-usuario/llarri-spanish-htr
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import HfApi, create_repo, upload_folder
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external"


def create_hf_dataset(dataset_name: str) -> DatasetDict:
    """Convierte el dataset local a formato Hugging Face."""
    
    dataset_path = DATA_DIR / dataset_name
    
    print(f"📂 Cargando dataset desde {dataset_path}...")
    
    # Cargar splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = dataset_path / f"{split_name}.json"
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Preparar datos para HF Dataset
            images = []
            texts = []
            
            for sample in tqdm(data, desc=f"Procesando {split_name}"):
                img_path = dataset_path / "images" / sample["image"]
                if img_path.exists():
                    images.append(str(img_path))
                    texts.append(sample["text"])
            
            # Crear Dataset
            splits[split_name] = Dataset.from_dict({
                "image": images,
                "text": texts,
            })
            
            print(f"   {split_name}: {len(splits[split_name])} muestras")
    
    # Renombrar val -> validation (convención HF)
    if "val" in splits:
        splits["validation"] = splits.pop("val")
    
    return DatasetDict(splits)


def upload_dataset(
    dataset_name: str,
    repo_name: str,
    private: bool = False,
):
    """Sube el dataset a Hugging Face Hub."""
    
    print("\n" + "="*60)
    print("📤 SUBIENDO DATASET A HUGGING FACE HUB")
    print("="*60)
    
    api = HfApi()
    
    # Verificar autenticación
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Autenticado como: {username}")
    except Exception as e:
        print("❌ No autenticado. Ejecuta: huggingface-cli login")
        return
    
    # Construir nombre del repo
    if "/" not in repo_name:
        repo_name = f"{username}/{repo_name}"
    
    print(f"\n📦 Repositorio: {repo_name}")
    print(f"🔒 Privado: {private}")
    
    # Crear repo
    try:
        create_repo(
            repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"✅ Repositorio creado/verificado")
    except Exception as e:
        print(f"⚠️  Error creando repo: {e}")
    
    # Crear dataset HF
    print(f"\n🔄 Convirtiendo dataset...")
    dataset = create_hf_dataset(dataset_name)
    
    print(f"\n📊 Dataset info:")
    print(dataset)
    
    # Subir
    print(f"\n📤 Subiendo a Hugging Face...")
    dataset.push_to_hub(
        repo_name,
        private=private,
    )
    
    print("\n" + "="*60)
    print("✅ DATASET SUBIDO EXITOSAMENTE")
    print("="*60)
    print(f"🔗 URL: https://huggingface.co/datasets/{repo_name}")
    print(f"\n📝 Para usar en entrenamiento:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{repo_name}')")
    
    return repo_name


def create_training_notebook(repo_name: str):
    """Crea un notebook de Colab/Kaggle para entrenamiento."""
    
    notebook_content = f'''{{
  "cells": [
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": [
        "# 🚀 Entrenamiento LLARRI - OCR Español\\n",
        "\\n",
        "Notebook para entrenar TrOCR con el dataset de escritura en español."
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Instalar dependencias\\n",
        "!pip install -q transformers datasets accelerate evaluate python-Levenshtein"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Verificar GPU\\n",
        "import torch\\n",
        "print(f'GPU: {{torch.cuda.get_device_name(0)}}')\\n",
        "print(f'VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Cargar dataset desde Hugging Face\\n",
        "from datasets import load_dataset\\n",
        "\\n",
        "dataset = load_dataset('{repo_name}')\\n",
        "print(dataset)"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Cargar modelo y procesador\\n",
        "from transformers import TrOCRProcessor, VisionEncoderDecoderModel\\n",
        "\\n",
        "processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')\\n",
        "model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')\\n",
        "\\n",
        "# Configurar\\n",
        "model.config.decoder_start_token_id = processor.tokenizer.cls_token_id\\n",
        "model.config.pad_token_id = processor.tokenizer.pad_token_id\\n",
        "model.config.vocab_size = model.config.decoder.vocab_size\\n",
        "model.config.eos_token_id = processor.tokenizer.sep_token_id\\n",
        "model.config.max_length = 64\\n",
        "model.config.early_stopping = True\\n",
        "model.config.num_beams = 4"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Preparar dataset para entrenamiento\\n",
        "from PIL import Image\\n",
        "import torch\\n",
        "\\n",
        "def preprocess(examples):\\n",
        "    images = [Image.open(img).convert('RGB') for img in examples['image']]\\n",
        "    pixel_values = processor(images, return_tensors='pt').pixel_values\\n",
        "    \\n",
        "    labels = processor.tokenizer(\\n",
        "        examples['text'],\\n",
        "        padding='max_length',\\n",
        "        max_length=64,\\n",
        "        truncation=True,\\n",
        "        return_tensors='pt'\\n",
        "    ).input_ids\\n",
        "    \\n",
        "    labels[labels == processor.tokenizer.pad_token_id] = -100\\n",
        "    \\n",
        "    return {{'pixel_values': pixel_values, 'labels': labels}}\\n",
        "\\n",
        "# Procesar\\n",
        "train_dataset = dataset['train'].map(preprocess, batched=True, batch_size=32, remove_columns=['image', 'text'])\\n",
        "eval_dataset = dataset['validation'].map(preprocess, batched=True, batch_size=32, remove_columns=['image', 'text'])"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Configuración de entrenamiento\\n",
        "from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments\\n",
        "\\n",
        "training_args = Seq2SeqTrainingArguments(\\n",
        "    output_dir='./llarri-spanish',\\n",
        "    num_train_epochs=5,\\n",
        "    per_device_train_batch_size=8,  # Ajustar según GPU\\n",
        "    per_device_eval_batch_size=8,\\n",
        "    gradient_accumulation_steps=2,\\n",
        "    learning_rate=5e-5,\\n",
        "    warmup_ratio=0.1,\\n",
        "    logging_steps=100,\\n",
        "    eval_strategy='steps',\\n",
        "    eval_steps=500,\\n",
        "    save_strategy='steps',\\n",
        "    save_steps=500,\\n",
        "    fp16=True,\\n",
        "    predict_with_generate=True,\\n",
        "    report_to='none',\\n",
        ")\\n",
        "\\n",
        "trainer = Seq2SeqTrainer(\\n",
        "    model=model,\\n",
        "    args=training_args,\\n",
        "    train_dataset=train_dataset,\\n",
        "    eval_dataset=eval_dataset,\\n",
        "    processing_class=processor,\\n",
        ")"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# ¡Entrenar!\\n",
        "trainer.train()"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Guardar modelo\\n",
        "trainer.save_model('./llarri-spanish-final')\\n",
        "processor.save_pretrained('./llarri-spanish-final')\\n",
        "\\n",
        "# Subir a Hugging Face (opcional)\\n",
        "# trainer.push_to_hub('llarri-spanish-ocr')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Evaluar\\n",
        "import Levenshtein\\n",
        "from tqdm import tqdm\\n",
        "\\n",
        "model.eval()\\n",
        "results = {{'cer': [], 'accuracy': []}}\\n",
        "\\n",
        "for sample in tqdm(dataset['test'][:1000]):\\n",
        "    image = Image.open(sample['image']).convert('RGB')\\n",
        "    pixel_values = processor(image, return_tensors='pt').pixel_values.cuda()\\n",
        "    \\n",
        "    with torch.no_grad():\\n",
        "        generated_ids = model.generate(pixel_values)\\n",
        "    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\\n",
        "    target = sample['text']\\n",
        "    \\n",
        "    cer = Levenshtein.distance(pred, target) / max(len(target), 1)\\n",
        "    results['cer'].append(cer)\\n",
        "    results['accuracy'].append(1.0 if pred.strip() == target.strip() else 0.0)\\n",
        "\\n",
        "print(f'CER: {{sum(results[\\'cer\\'])/len(results[\\'cer\\'])*100:.2f}}%')\\n",
        "print(f'Accuracy: {{sum(results[\\'accuracy\\'])/len(results[\\'accuracy\\'])*100:.2f}}%')"
      ]
    }}
  ],
  "metadata": {{
    "accelerator": "GPU",
    "colab": {{
      "gpuType": "T4",
      "provenance": []
    }},
    "kernelspec": {{
      "display_name": "Python 3",
      "name": "python3"
    }}
  }},
  "nbformat": 4,
  "nbformat_minor": 0
}}'''
    
    notebook_path = PROJECT_ROOT / "notebooks" / "train_llarri_colab.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"\n📓 Notebook creado: {notebook_path}")
    print("   Súbelo a Google Colab o Kaggle para entrenar con GPU gratis")
    
    return notebook_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sube dataset a Hugging Face")
    parser.add_argument("--dataset", type=str, default="spanish_synthetic_4gb",
                       help="Nombre del dataset local")
    parser.add_argument("--repo", type=str, default="llarri-spanish-htr",
                       help="Nombre del repositorio en HF")
    parser.add_argument("--private", action="store_true",
                       help="Hacer el dataset privado")
    parser.add_argument("--create-notebook", action="store_true",
                       help="Crear notebook de Colab")
    
    args = parser.parse_args()
    
    repo_name = upload_dataset(
        dataset_name=args.dataset,
        repo_name=args.repo,
        private=args.private,
    )
    
    if args.create_notebook and repo_name:
        create_training_notebook(repo_name)
