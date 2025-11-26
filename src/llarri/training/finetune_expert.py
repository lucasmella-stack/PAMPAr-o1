#!/usr/bin/env python3
"""
finetune_expert.py - Fine-tuning de modelos expertos especializados

Este script permite crear expertos especializados a partir del modelo base:
- Carga un modelo base pre-entrenado
- Aplica estrategias de fine-tuning eficiente (Adapter, LoRA, Full)
- Entrena en datos específicos del dominio/estilo
- Guarda el expert head para uso posterior

Estrategias de fine-tuning:
1. Adapter: Añade capas pequeñas, congela el resto (~1% parámetros)
2. LoRA: Matrices de bajo rango en attention (~0.1-1% parámetros)
3. Full: Fine-tuning completo o parcial del decoder

Uso:
    # Fine-tune con configuración de expert
    python -m llarri.training.finetune_expert --config configs/expert_es_mayores.yaml
    
    # Fine-tune con modelo base específico
    python -m llarri.training.finetune_expert \\
        --base-model outputs/experiment_xxx/checkpoints/final_model.ckpt \\
        --config configs/expert_latam_jovenes.yaml \\
        --expert-type adapter
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.models.expert_head import ExpertHead
from llarri.data.datamodule_base import LlarriDataModule


class ExpertModel(pl.LightningModule):
    """
    Modelo Lightning para entrenar expert heads.
    
    Combina:
    - Encoder del modelo base (congelado)
    - Expert head (entrenable)
    """
    def __init__(
        self,
        base_model: LlarriBaseModel,
        expert_type: str = "adapter",
        expert_config: Optional[Dict] = None,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        
        self.learning_rate = learning_rate
        self.expert_type = expert_type
        
        # Usar encoder del modelo base
        self.encoder = base_model.encoder
        
        # Congelar encoder (usualmente queremos esto)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("🔒 Encoder congelado")
        
        # Crear expert head
        expert_config = expert_config or {}
        self.expert_head = ExpertHead(
            base_decoder=base_model.decoder.decoder,
            expert_type=expert_type,
            **expert_config
        )
        
        # Copiar tokenizer del modelo base
        self.tokenizer = base_model.tokenizer
        
        # Estadísticas
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n📊 Estadísticas del modelo experto:")
        print(f"   Tipo: {expert_type}")
        print(f"   Parámetros totales: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        print(f"   Ratio: {100*trainable_params/total_params:.2f}%")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass: encoder + expert head."""
        # Encoder (sin gradientes)
        with torch.no_grad() if not self.encoder.training else torch.enable_grad():
            encoder_hidden = self.encoder(pixel_values)
        
        # Expert head
        output = self.expert_head(encoder_hidden, labels=labels)
        return output
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ):
        """Generate text from images."""
        with torch.no_grad():
            encoder_hidden = self.encoder(pixel_values)
        
        generated_ids = self.expert_head.generate(
            encoder_hidden_states=encoder_hidden,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        output = self(pixel_values, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output["loss"]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        output = self(pixel_values, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output["loss"]
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generar predicciones cada N batches
        if batch_idx % 20 == 0:
            with torch.no_grad():
                predictions = self.generate(pixel_values, max_length=64)
                # Aquí podrías calcular métricas adicionales (CER, WER)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer for expert parameters only."""
        # Solo optimizar parámetros del expert head
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(*config_paths: str) -> Dict[str, Any]:
    """Merge multiple config files."""
    merged = {}
    for path in config_paths:
        if os.path.exists(path):
            config = load_config(path)
            merged = deep_merge(merged, config)
    return merged


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def finetune_expert(
    base_model_path: str,
    config: Dict[str, Any],
    args: argparse.Namespace
) -> None:
    """Main fine-tuning function."""
    
    # Directorio de salida
    expert_config = config.get("expert", {})
    expert_name = expert_config.get("name", "unnamed_expert")
    output_dir = args.output_dir or f"outputs/experts/{expert_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"finetune_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Guardar configuración
    config_save_path = os.path.join(experiment_dir, "expert_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n🎯 Fine-tuning Expert: {expert_name}")
    print(f"📁 Directorio: {experiment_dir}")
    
    # Cargar modelo base
    print(f"\n📦 Cargando modelo base desde: {base_model_path}")
    base_model = LlarriBaseModel.load_from_checkpoint(base_model_path)
    base_model.eval()
    
    # Configuración del expert
    expert_type = args.expert_type or expert_config.get("type", "adapter")
    expert_params = expert_config.get("params", {})
    
    # Configuración específica por tipo
    if expert_type == "adapter":
        expert_params.setdefault("adapter_size", 64)
        expert_params.setdefault("num_adapter_layers", 6)
    elif expert_type == "lora":
        expert_params.setdefault("rank", 8)
        expert_params.setdefault("alpha", 16.0)
    elif expert_type == "full":
        expert_params.setdefault("freeze_embeddings", True)
        expert_params.setdefault("freeze_n_layers", 0)
    
    # Crear expert model
    print(f"\n🔧 Creando expert model (tipo: {expert_type})...")
    expert_model = ExpertModel(
        base_model=base_model,
        expert_type=expert_type,
        expert_config=expert_params,
        learning_rate=config.get("training", {}).get("learning_rate", 1e-4),
        freeze_encoder=config.get("expert", {}).get("freeze_encoder", True),
    )
    
    # Configurar DataModule
    print("\n📊 Configurando DataModule...")
    data_config = config.get("data", {})
    paths_config = config.get("paths", {})
    
    # Usar datos específicos del expert si están definidos
    expert_data_path = expert_config.get("data_path", paths_config.get("splits", "data/splits"))
    
    datamodule = LlarriDataModule(
        train_path=os.path.join(expert_data_path, "train.jsonl"),
        val_path=os.path.join(expert_data_path, "val.jsonl"),
        data_root=paths_config.get("processed_data", "data/processed"),
        batch_size=config.get("training", {}).get("batch_size", 8),
        num_workers=data_config.get("num_workers", 4),
        tokenizer=expert_model.tokenizer,
        max_length=data_config.get("max_length", 128),
        img_height=data_config.get("img_height", 128),
        img_width=data_config.get("img_width", 512),
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(experiment_dir, "checkpoints"),
            filename=f"{expert_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("training", {}).get("patience", 10),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Trainer
    training_config = config.get("training", {})
    trainer = pl.Trainer(
        max_epochs=args.epochs or training_config.get("epochs", 20),
        accelerator=args.accelerator or "auto",
        devices=args.devices or 1,
        precision=training_config.get("precision", "16-mixed"),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    print("\n🚀 Iniciando fine-tuning...")
    print(f"   Épocas: {trainer.max_epochs}")
    print(f"   Batch size: {config.get('training', {}).get('batch_size', 8)}")
    print(f"   Learning rate: {expert_model.learning_rate}")
    
    trainer.fit(expert_model, datamodule)
    
    # Guardar expert final
    final_path = os.path.join(experiment_dir, "checkpoints", f"{expert_name}_final.ckpt")
    trainer.save_checkpoint(final_path)
    
    # También guardar solo el expert head (más pequeño)
    expert_head_path = os.path.join(experiment_dir, f"{expert_name}_head.pt")
    torch.save({
        'expert_type': expert_type,
        'expert_state_dict': expert_model.expert_head.state_dict(),
        'expert_config': expert_params,
        'tokenizer_name': base_model.decoder_cfg.pretrained_model_name,
    }, expert_head_path)
    
    print(f"\n✅ Fine-tuning completado!")
    print(f"   Checkpoint completo: {final_path}")
    print(f"   Expert head: {expert_head_path}")
    
    # Métricas finales
    if trainer.callback_metrics:
        print("\n📈 Métricas finales:")
        for key, value in trainer.callback_metrics.items():
            print(f"   {key}: {value:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune expert models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        required=True,
        help="Path to base model checkpoint"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to expert configuration YAML"
    )
    
    # Expert configuration
    parser.add_argument(
        "--expert-type", "-t",
        type=str,
        choices=["adapter", "lora", "full"],
        default=None,
        help="Type of expert head (overrides config)"
    )
    
    # Training
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of epochs"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Accelerator type"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of devices"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Verificar que existe el modelo base
    if not os.path.exists(args.base_model):
        print(f"❌ Error: Modelo base no encontrado: {args.base_model}")
        sys.exit(1)
    
    # Cargar configuración
    print("📋 Cargando configuración...")
    config = load_config(args.config)
    
    # También cargar configs base si existen
    base_configs = [
        "configs/base_model.yaml",
        "configs/data_paths.yaml",
        "configs/training.yaml",
    ]
    
    for cfg_path in base_configs:
        if os.path.exists(cfg_path):
            base_cfg = load_config(cfg_path)
            config = deep_merge(base_cfg, config)
    
    # Ejecutar fine-tuning
    try:
        finetune_expert(args.base_model, config, args)
        print("\n✅ Proceso completado exitosamente!")
    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

