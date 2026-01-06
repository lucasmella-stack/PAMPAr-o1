#!/usr/bin/env python3
"""
train_llarri_real.py - Entrenamiento real del modelo LLARRI

Entrena el modelo TrOCR con el dataset sintético en español.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from tqdm import tqdm
import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external" / "spanish_synthetic_4gb"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "llarri_spanish"


class OCRDataset(Dataset):
    """Dataset para OCR."""
    
    def __init__(self, data_path: Path, processor: TrOCRProcessor, max_samples: int = None):
        self.processor = processor
        self.images_dir = data_path / "images"
        
        # Cargar labels
        labels_path = data_path / "train.json"
        if not labels_path.exists():
            labels_path = data_path / "labels.json"
        
        with open(labels_path, 'r') as f:
            self.samples = json.load(f)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Dataset cargado: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Cargar imagen
        img_path = self.images_dir / sample["image"]
        image = Image.open(img_path).convert("RGB")
        
        # Procesar imagen
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Procesar texto (labels)
        text = sample["text"]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Reemplazar padding token id por -100 para ignorar en loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class ValDataset(Dataset):
    """Dataset de validación."""
    
    def __init__(self, data_path: Path, processor: TrOCRProcessor, max_samples: int = 1000):
        self.processor = processor
        self.images_dir = data_path / "images"
        
        val_path = data_path / "val.json"
        with open(val_path, 'r') as f:
            self.samples = json.load(f)[:max_samples]
        
        logger.info(f"Val dataset: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = self.images_dir / sample["image"]
        image = Image.open(img_path).convert("RGB")
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        text = sample["text"]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def compute_metrics(pred):
    """Calcula métricas de evaluación."""
    cer_metric = evaluate.load("cer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Reemplazar -100 por pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decodificar
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Calcular CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}


def train(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_train_samples: int = None,
    gradient_accumulation: int = 4,
    fp16: bool = True,
):
    """Entrena el modelo LLARRI."""
    
    global processor  # Para compute_metrics
    
    print("\n" + "="*60)
    print("🚀 ENTRENAMIENTO LLARRI - MODELO OCR EN ESPAÑOL")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Cargar modelo y procesador
    print("\n🤖 Cargando modelo TrOCR-base-handwritten...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Configurar para generación
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64  # Reducido para ahorrar memoria
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 1  # Greedy decoding para ahorrar memoria
    
    # Habilitar gradient checkpointing para ahorrar VRAM
    model.gradient_checkpointing_enable()
    
    # Congelar encoder para usar menos memoria
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Datasets
    print(f"\n📂 Cargando datasets desde {DATA_DIR}...")
    train_dataset = OCRDataset(DATA_DIR, processor, max_samples=max_train_samples)
    val_dataset = ValDataset(DATA_DIR, processor, max_samples=1000)
    
    # Configuración de entrenamiento
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # Reducido para GPU de 4GB
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Compensar batch pequeño
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=False,  # Desactivar para ahorrar memoria
        fp16=fp16 and device.type == "cuda",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch_fused",  # Optimizador más eficiente
        gradient_checkpointing=True,
    )
    
    print(f"\n⚙️  Configuración:")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Gradient accumulation: {gradient_accumulation}")
    print(f"   • Effective batch: {batch_size * gradient_accumulation}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • FP16: {fp16 and device.type == 'cuda'}")
    print(f"   • Train samples: {len(train_dataset):,}")
    print(f"   • Val samples: {len(val_dataset):,}")
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        # compute_metrics=compute_metrics,  # Requiere evaluate
    )
    
    # Entrenar
    print(f"\n🏋️ Iniciando entrenamiento...")
    print("-"*60)
    
    start_time = datetime.now()
    trainer.train()
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Tiempo total: {elapsed}")
    
    # Guardar modelo final
    final_path = OUTPUT_DIR / "final"
    print(f"\n💾 Guardando modelo en {final_path}...")
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"📁 Modelo guardado en: {final_path}")
    
    return trainer, processor


def evaluate_model(model_path: str = None):
    """Evalúa el modelo entrenado."""
    import Levenshtein
    
    if model_path is None:
        model_path = OUTPUT_DIR / "final"
    
    print(f"\n📊 Evaluando modelo desde {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Cargar test set
    test_path = DATA_DIR / "test.json"
    with open(test_path, 'r') as f:
        test_data = json.load(f)[:1000]  # 1000 muestras para eval rápida
    
    print(f"   Evaluando {len(test_data)} muestras...")
    
    results = {"cer": [], "wer": [], "accuracy": []}
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluando"):
            img_path = DATA_DIR / "images" / sample["image"]
            target = sample["text"]
            
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            
            generated_ids = model.generate(pixel_values, max_length=128)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # CER
            cer = Levenshtein.distance(pred, target) / max(len(target), 1)
            results["cer"].append(cer)
            
            # WER
            pred_words = pred.split()
            target_words = target.split()
            wer = Levenshtein.distance(pred_words, target_words) / max(len(target_words), 1)
            results["wer"].append(wer)
            
            # Accuracy
            results["accuracy"].append(1.0 if pred.strip() == target.strip() else 0.0)
    
    # Promedios
    avg_cer = sum(results["cer"]) / len(results["cer"]) * 100
    avg_wer = sum(results["wer"]) / len(results["wer"]) * 100
    avg_acc = sum(results["accuracy"]) / len(results["accuracy"]) * 100
    
    print("\n" + "="*60)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("="*60)
    print(f"   • CER:      {avg_cer:.2f}%")
    print(f"   • WER:      {avg_wer:.2f}%")
    print(f"   • Accuracy: {avg_acc:.2f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Máximo de muestras")
    parser.add_argument("--eval-only", action="store_true", help="Solo evaluar")
    parser.add_argument("--model-path", type=str, default=None, help="Path al modelo para evaluar")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_model(args.model_path)
    else:
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_train_samples=args.max_samples,
        )
