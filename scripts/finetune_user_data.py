#!/usr/bin/env python3
"""
finetune_user_data.py - Fine-tuning del modelo con tus propios datos

Este script permite entrenar el modelo OCR con tus propias imágenes
de texto manuscrito.

Requisitos:
    1. Imágenes en: data/user_samples/images/
    2. Anotaciones en: data/user_samples/annotations.txt
       Formato: nombre_imagen<TAB>texto

Uso:
    python scripts/finetune_user_data.py
    python scripts/finetune_user_data.py --epochs 10 --lr 1e-5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class UserDataset(Dataset):
    """Dataset para imágenes de usuario con anotaciones."""
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        processor: TrOCRProcessor,
        max_length: int = 64,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Cargar anotaciones
        self.samples = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    img_name, text = parts[0], parts[1]
                    img_path = self.images_dir / img_name
                    if img_path.exists():
                        self.samples.append((img_path, text))
                    else:
                        print(f"⚠️ Imagen no encontrada: {img_path}")
        
        print(f"📊 Dataset cargado: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert("RGB")
        
        # Procesar imagen
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenizar texto
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class OCRFineTuner(pl.LightningModule):
    """Módulo de PyTorch Lightning para fine-tuning de TrOCR."""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        learning_rate: float = 5e-5,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # Cargar modelo pre-entrenado
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Opcionalmente congelar encoder (más rápido, menos memoria)
        if freeze_encoder:
            print("🔒 Encoder congelado (solo entrena decoder)")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values, labels=None):
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        
        # Calcular accuracy (opcional)
        generated_ids = self.model.generate(batch["pixel_values"], max_length=64)
        pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Decodificar labels
        labels = batch["labels"]
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        target_texts = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Calcular exact match
        correct = sum(1 for p, t in zip(pred_texts, target_texts) if p.strip() == t.strip())
        accuracy = correct / len(pred_texts)
        self.log("val_accuracy", accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def generate(self, image: Image.Image) -> str:
        """Genera texto a partir de una imagen."""
        self.eval()
        with torch.no_grad():
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, max_length=64)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning de OCR con datos propios")
    parser.add_argument("--images", default="data/user_samples/images", help="Carpeta de imágenes")
    parser.add_argument("--annotations", default="data/user_samples/annotations.txt", help="Archivo de anotaciones")
    parser.add_argument("--epochs", type=int, default=10, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Congelar encoder")
    parser.add_argument("--output", default="outputs/finetuned_model", help="Carpeta de salida")
    args = parser.parse_args()
    
    print("="*60)
    print("LLARRI-OCR - Fine-Tuning con Datos Propios")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Verificar que existen datos
    if not Path(args.annotations).exists():
        print(f"\n❌ Error: No se encontró el archivo de anotaciones: {args.annotations}")
        print("\nPara crear el archivo de anotaciones:")
        print("1. Pon tus imágenes en: data/user_samples/images/")
        print("2. Crea un archivo annotations.txt con formato:")
        print("   imagen1.png<TAB>texto de la imagen 1")
        print("   imagen2.png<TAB>texto de la imagen 2")
        sys.exit(1)
    
    # Crear modelo
    model = OCRFineTuner(
        learning_rate=args.lr,
        freeze_encoder=args.freeze_encoder,
    )
    
    # Crear dataset
    dataset = UserDataset(
        images_dir=args.images,
        annotations_file=args.annotations,
        processor=model.processor,
    )
    
    if len(dataset) == 0:
        print("\n❌ Error: No se encontraron muestras válidas")
        print("Verifica que:")
        print("1. Las imágenes existen en la carpeta especificada")
        print("2. El archivo de anotaciones tiene el formato correcto")
        sys.exit(1)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
        ),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        default_root_dir=args.output,
        enable_progress_bar=True,
    )
    
    print(f"\n🚀 Iniciando entrenamiento...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} (effective: {args.batch_size * 8})")
    print(f"   Learning rate: {args.lr}")
    
    # Entrenar
    trainer.fit(model, train_loader, val_loader)
    
    # Guardar modelo final
    final_path = Path(args.output) / "final_model"
    model.model.save_pretrained(final_path)
    model.processor.save_pretrained(final_path)
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {final_path}")
    print(f"\nPara usar el modelo entrenado:")
    print(f"  from transformers import TrOCRProcessor, VisionEncoderDecoderModel")
    print(f"  processor = TrOCRProcessor.from_pretrained('{final_path}')")
    print(f"  model = VisionEncoderDecoderModel.from_pretrained('{final_path}')")


if __name__ == "__main__":
    main()
