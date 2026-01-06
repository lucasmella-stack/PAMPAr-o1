"""
train_selector.py - Entrenamiento del Selector de Estilo

Script completo para entrenar el clasificador que determina qué experto usar.

Flujo:
1. Preparar dataset con imágenes etiquetadas por estilo
2. Crear StyleSelector (CNN o ViT-based)
3. Entrenar con cross-entropy loss
4. Evaluar precisión y matriz de confusión
5. Guardar mejor modelo

Dataset esperado:
- JSONL con campos: {id, image_path, text, style_label}
- style_label puede ser:
  * Numérico: 0, 1, 2, ...
  * String: "es_mayores", "latam_jovenes", "general"

Uso:
    python scripts/train_selector.py \
        --data_path data/splits/selector_train.jsonl \
        --val_data_path data/splits/selector_val.jsonl \
        --selector_type cnn \
        --num_classes 3 \
        --max_epochs 30
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from ..models.selector_style import (
    StyleSelector,
    StyleSelectorCNN,
    StyleSelectorViT,
    MultiTaskStyleSelector
)
from ..models.encoder_vit import ViTEncoder


# =============================================================================
# DATASET
# =============================================================================

class StyleDataset(Dataset):
    """
    Dataset de imágenes con etiquetas de estilo para entrenar el selector.
    """
    def __init__(
        self,
        jsonl_path: str,
        transform = None,
        label_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            jsonl_path: Archivo JSONL con {id, image_path, text, style_label}
            transform: Transformaciones de imagen
            label_to_idx: Mapeo de etiquetas string a índices numéricos
        """
        self.data = pd.read_json(jsonl_path, lines=True)
        self.transform = transform
        
        # Construir mapeo de labels
        if label_to_idx is None:
            unique_labels = sorted(self.data['style_label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"✅ StyleDataset cargado: {len(self.data)} muestras")
        print(f"   Clases: {list(self.label_to_idx.keys())}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Cargar imagen
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Transformar
        if self.transform:
            image = self.transform(image)
        
        # Convertir label
        label = row['style_label']
        if isinstance(label, str):
            label_idx = self.label_to_idx[label]
        else:
            label_idx = int(label)
        
        return {
            'image': image,
            'label': label_idx,
            'image_path': image_path
        }


class MultiTaskStyleDataset(StyleDataset):
    """
    Dataset con múltiples etiquetas por imagen (age_group, region, formality, etc).
    
    JSONL esperado: {id, image_path, text, age_group, region, formality, quality}
    """
    def __init__(
        self,
        jsonl_path: str,
        transform = None,
        task_configs: Dict[str, Dict[str, int]] = None
    ):
        """
        Args:
            task_configs: {"task_name": {"label": idx}}
        """
        self.data = pd.read_json(jsonl_path, lines=True)
        self.transform = transform
        self.task_configs = task_configs or {}
        
        print(f"✅ MultiTaskStyleDataset cargado: {len(self.data)} muestras")
        print(f"   Tasks: {list(self.task_configs.keys())}")
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Cargar imagen
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extraer labels por task
        labels = {}
        for task_name, label_map in self.task_configs.items():
            label_str = row[task_name]
            labels[task_name] = label_map[label_str]
        
        return {
            'image': image,
            'labels': labels,
            'image_path': image_path
        }


# =============================================================================
# LIGHTNING MODULE
# =============================================================================

class StyleSelectorModule(pl.LightningModule):
    """
    Lightning module para entrenar el selector de estilo.
    """
    def __init__(
        self,
        selector_type: str = "cnn",
        num_classes: int = 3,
        vit_encoder = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        **selector_kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vit_encoder', 'class_weights'])
        
        # Crear selector
        self.selector = StyleSelector(
            selector_type=selector_type,
            num_classes=num_classes,
            vit_encoder=vit_encoder,
            **selector_kwargs
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Métricas
        self.train_acc = []
        self.val_acc = []
        self.val_predictions = []
        self.val_labels = []
    
    def forward(self, x):
        return self.selector(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        # Forward
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        # Forward
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        # Guardar para métricas finales
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calcular métricas globales
        if len(self.val_predictions) > 0:
            y_true = np.array(self.val_labels)
            y_pred = np.array(self.val_predictions)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print("\n📊 Confusion Matrix:")
            print(cm)
            
            # Classification report
            report = classification_report(y_true, y_pred)
            print("\n📈 Classification Report:")
            print(report)
            
            # Resetear
            self.val_predictions = []
            self.val_labels = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


class MultiTaskSelectorModule(pl.LightningModule):
    """
    Lightning module para selector multi-task.
    """
    def __init__(
        self,
        task_configs: Dict[str, int],
        vit_encoder = None,
        learning_rate: float = 1e-4,
        task_weights: Optional[Dict[str, float]] = None,
        **selector_kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vit_encoder'])
        
        # Crear selector multi-task
        self.selector = MultiTaskStyleSelector(
            vit_encoder=vit_encoder,
            task_configs=task_configs,
            **selector_kwargs
        )
        
        # Loss por task
        self.task_weights = task_weights or {task: 1.0 for task in task_configs.keys()}
        self.criterions = {task: nn.CrossEntropyLoss() for task in task_configs.keys()}
    
    def forward(self, x):
        return self.selector(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']  # Dict[str, Tensor]
        
        # Forward
        logits = self(images)  # Dict[str, Tensor]
        
        # Calcular loss por task
        total_loss = 0.0
        for task_name, task_logits in logits.items():
            task_labels = labels[task_name]
            task_loss = self.criterions[task_name](task_logits, task_labels)
            task_weight = self.task_weights[task_name]
            total_loss += task_weight * task_loss
            
            # Accuracy
            preds = torch.argmax(task_logits, dim=-1)
            acc = (preds == task_labels).float().mean()
            
            self.log(f'train/{task_name}_loss', task_loss)
            self.log(f'train/{task_name}_acc', acc)
        
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        
        total_loss = 0.0
        for task_name, task_logits in logits.items():
            task_labels = labels[task_name]
            task_loss = self.criterions[task_name](task_logits, task_labels)
            task_weight = self.task_weights[task_name]
            total_loss += task_weight * task_loss
            
            preds = torch.argmax(task_logits, dim=-1)
            acc = (preds == task_labels).float().mean()
            
            self.log(f'val/{task_name}_loss', task_loss)
            self.log(f'val/{task_name}_acc', acc)
        
        self.log('val/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/total_loss'}
        }


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(image_size=(224, 224), augment=True):
    """Transformaciones de imagen para el selector."""
    if augment:
        return T.Compose([
            T.Resize(image_size),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_selector(
    data_path: str,
    val_data_path: str,
    selector_type: str = "cnn",
    num_classes: int = 3,
    vit_model_path: Optional[str] = None,
    batch_size: int = 32,
    max_epochs: int = 30,
    learning_rate: float = 1e-4,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    output_dir: str = "outputs/selector",
    accelerator: str = "auto"
):
    """
    Entrena el selector de estilo.
    
    Args:
        data_path: JSONL de entrenamiento
        val_data_path: JSONL de validación
        selector_type: "cnn", "vit", "multitask"
        num_classes: Número de clases a predecir
        vit_model_path: Path a checkpoint de ViT encoder (para selector_type="vit")
        batch_size: Batch size
        max_epochs: Máximo de épocas
        learning_rate: Learning rate
        image_size: Tamaño de imagen (H, W)
        num_workers: Workers para DataLoader
        output_dir: Directorio de salida
        accelerator: "auto", "gpu", "cpu"
    """
    print("="*60)
    print("🎯 ENTRENAMIENTO DE SELECTOR DE ESTILO")
    print("="*60)
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cargar encoder ViT si es necesario
    vit_encoder = None
    if selector_type in ["vit", "multitask"]:
        if vit_model_path is None:
            print("⚠️  Creando ViT encoder desde cero...")
            vit_encoder = ViTEncoder(
                model_name="google/vit-base-patch16-224-in21k",
                freeze_encoder=False
            )
        else:
            print(f"📂 Cargando ViT encoder desde {vit_model_path}")
            # Cargar weights del encoder desde checkpoint base
            checkpoint = torch.load(vit_model_path, map_location='cpu')
            vit_encoder = ViTEncoder(
                model_name="google/vit-base-patch16-224-in21k",
                freeze_encoder=True
            )
            # Extraer solo encoder weights
            encoder_state = {k.replace('encoder.', ''): v 
                           for k, v in checkpoint['state_dict'].items() 
                           if k.startswith('encoder.')}
            vit_encoder.load_state_dict(encoder_state, strict=False)
    
    # Crear datasets
    train_transform = get_transforms(image_size=image_size, augment=True)
    val_transform = get_transforms(image_size=image_size, augment=False)
    
    train_dataset = StyleDataset(data_path, transform=train_transform)
    val_dataset = StyleDataset(val_data_path, transform=val_transform, 
                              label_to_idx=train_dataset.label_to_idx)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Crear module
    module = StyleSelectorModule(
        selector_type=selector_type,
        num_classes=num_classes,
        vit_encoder=vit_encoder,
        learning_rate=learning_rate
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='selector-{epoch:02d}-{val/acc:.3f}',
        monitor='val/acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        precision='16-mixed'
    )
    
    # Train
    print(f"\n🚀 Iniciando entrenamiento...")
    print(f"   Tipo de selector: {selector_type}")
    print(f"   Clases: {num_classes}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}\n")
    
    trainer.fit(module, train_loader, val_loader)
    
    print("\n✅ Entrenamiento completado")
    print(f"📂 Modelos guardados en: {output_dir}")
    print(f"   Mejor checkpoint: {checkpoint_callback.best_model_path}")
    
    return trainer, module


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenar selector de estilo")
    
    # Datos
    parser.add_argument('--data_path', type=str, required=True,
                       help='JSONL de entrenamiento')
    parser.add_argument('--val_data_path', type=str, required=True,
                       help='JSONL de validación')
    
    # Modelo
    parser.add_argument('--selector_type', type=str, default='cnn',
                       choices=['cnn', 'vit', 'multitask'],
                       help='Tipo de selector')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Número de clases')
    parser.add_argument('--vit_model_path', type=str, default=None,
                       help='Path a checkpoint de modelo base con ViT encoder')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/selector')
    parser.add_argument('--accelerator', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu'])
    
    args = parser.parse_args()
    
    train_selector(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        selector_type=args.selector_type,
        num_classes=args.num_classes,
        vit_model_path=args.vit_model_path,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        image_size=tuple(args.image_size),
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        accelerator=args.accelerator
    )


if __name__ == '__main__':
    main()

