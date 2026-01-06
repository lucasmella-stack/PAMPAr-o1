#!/usr/bin/env python3
"""
train_base.py - Script de entrenamiento del modelo base LlarriOCR

Este script proporciona:
- Carga de configuración desde archivos YAML
- Interfaz CLI con argumentos
- Entrenamiento con PyTorch Lightning
- Callbacks para checkpoints, early stopping y logging
- Soporte para GPU/CPU/TPU automático
- Logging con WandB (opcional) o TensorBoard

Uso:
    python -m llarri.training.train_base --config configs/training.yaml
    python -m llarri.training.train_base --epochs 20 --batch-size 16
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.models.encoder_vit import ViTEncoderConfig
from llarri.models.decoder_trocr import TrOCRDecoderConfig
from llarri.data.datamodule_base import LlarriDataModule


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(*config_paths: str) -> Dict[str, Any]:
    """Combina múltiples archivos de configuración."""
    merged = {}
    for path in config_paths:
        if os.path.exists(path):
            config = load_config(path)
            merged = deep_merge(merged, config)
    return merged


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Merge profundo de diccionarios."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_callbacks(config: Dict[str, Any], output_dir: str) -> list:
    """Configura callbacks de PyTorch Lightning."""
    callbacks = []
    
    # Model Checkpoint - Guardar mejores modelos
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="llarri-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=config.get("save_top_k", 3),
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    if config.get("early_stopping", {}).get("enabled", True):
        early_stop = EarlyStopping(
            monitor=config.get("early_stopping", {}).get("monitor", "val_loss"),
            patience=config.get("early_stopping", {}).get("patience", 5),
            mode=config.get("early_stopping", {}).get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stop)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Progress Bar (Rich)
    try:
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
    except ImportError:
        pass  # Rich no instalado
    
    # Device Stats (para debugging)
    if config.get("log_device_stats", False):
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)
    
    return callbacks


def setup_logger(config: Dict[str, Any], output_dir: str):
    """Configura logger para entrenamiento."""
    logger_type = config.get("logger", "tensorboard")
    
    if logger_type == "wandb":
        try:
            return WandbLogger(
                project=config.get("wandb_project", "llarri-ocr"),
                name=config.get("experiment_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                save_dir=output_dir,
                log_model=True,
            )
        except ImportError:
            print("WandB no instalado, usando TensorBoard")
            logger_type = "tensorboard"
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard_logs",
            default_hp_metric=False,
        )
    
    return True  # Logger por defecto de Lightning


def create_model(config: Dict[str, Any]) -> LlarriBaseModel:
    """Crea el modelo a partir de la configuración."""
    model_config = config.get("model", {})
    
    # Configurar encoder
    encoder_cfg = ViTEncoderConfig(
        pretrained_model_name=model_config.get(
            "encoder_pretrained", 
            "google/vit-base-patch16-224-in21k"
        ),
        freeze=model_config.get("freeze_encoder", False),
    )
    
    # Configurar decoder
    decoder_cfg = TrOCRDecoderConfig(
        pretrained_model_name=model_config.get(
            "decoder_pretrained",
            "microsoft/trocr-base-handwritten"
        ),
        freeze=model_config.get("freeze_decoder", False),
    )
    
    # Crear modelo
    model = LlarriBaseModel(
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        learning_rate=config.get("training", {}).get("learning_rate", 5e-5),
    )
    
    return model


def create_datamodule(config: Dict[str, Any], tokenizer) -> LlarriDataModule:
    """Crea el DataModule a partir de la configuración."""
    paths_config = config.get("paths", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    splits_dir = paths_config.get("splits", "data/splits")
    
    datamodule = LlarriDataModule(
        train_path=os.path.join(splits_dir, "train.jsonl"),
        val_path=os.path.join(splits_dir, "val.jsonl"),
        data_root=paths_config.get("processed_data", "data/processed"),
        batch_size=training_config.get("batch_size", 8),
        num_workers=data_config.get("num_workers", 4),
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 128),
        img_height=data_config.get("img_height", 128),
        img_width=data_config.get("img_width", 512),
    )
    
    return datamodule


def train(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Función principal de entrenamiento."""
    
    # Directorio de salida
    output_dir = args.output_dir or config.get("output_dir", "outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Guardar configuración usada
    config_save_path = os.path.join(experiment_dir, "config_used.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📁 Directorio de experimento: {experiment_dir}")
    
    # Crear modelo
    print("🔧 Creando modelo...")
    model = create_model(config)
    
    # Crear DataModule con tokenizer del modelo
    print("📊 Configurando DataModule...")
    datamodule = create_datamodule(config, model.tokenizer)
    
    # Configurar callbacks
    callbacks = setup_callbacks(config.get("callbacks", {}), experiment_dir)
    
    # Configurar logger
    logger = setup_logger(config.get("logging", {}), experiment_dir)
    
    # Configurar Trainer
    training_config = config.get("training", {})
    
    trainer = pl.Trainer(
        max_epochs=args.epochs or training_config.get("epochs", 10),
        accelerator=args.accelerator or training_config.get("accelerator", "auto"),
        devices=args.devices or training_config.get("devices", 1),
        precision=training_config.get("precision", "16-mixed"),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        val_check_interval=training_config.get("val_check_interval", 1.0),
        log_every_n_steps=training_config.get("log_every_n_steps", 10),
        callbacks=callbacks,
        logger=logger,
        deterministic=training_config.get("deterministic", False),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    print(f"   Épocas: {trainer.max_epochs}")
    print(f"   Batch size: {config.get('training', {}).get('batch_size', 8)}")
    print(f"   Learning rate: {config.get('training', {}).get('learning_rate', 5e-5)}")
    print(f"   Dispositivo: {trainer.accelerator}")
    
    trainer.fit(model, datamodule)
    
    # Guardar modelo final
    final_checkpoint_path = os.path.join(experiment_dir, "checkpoints", "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"✅ Modelo final guardado en: {final_checkpoint_path}")
    
    # Imprimir métricas finales
    print("\n📈 Métricas finales:")
    if trainer.callback_metrics:
        for key, value in trainer.callback_metrics.items():
            print(f"   {key}: {value:.4f}")
    
    return trainer


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo base LlarriOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    # Entrenar con configuración por defecto
    python -m llarri.training.train_base
    
    # Entrenar con configuración personalizada
    python -m llarri.training.train_base --config configs/training.yaml
    
    # Override de parámetros
    python -m llarri.training.train_base --epochs 50 --batch-size 16 --lr 1e-4
    
    # Entrenar en GPU específica
    python -m llarri.training.train_base --accelerator gpu --devices 1
        """
    )
    
    # Configuración
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/training.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/base_model.yaml",
        help="Ruta a la configuración del modelo"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data_paths.yaml",
        help="Ruta a la configuración de datos"
    )
    
    # Override de hiperparámetros
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Número de épocas de entrenamiento"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Tamaño del batch"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    
    # Hardware
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["auto", "cpu", "gpu", "tpu", "mps"],
        default=None,
        help="Tipo de acelerador"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Número de dispositivos"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directorio de salida"
    )
    
    # Checkpoint
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Ruta a checkpoint para continuar entrenamiento"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug (fast_dev_run)"
    )
    
    return parser.parse_args()


def main():
    """Punto de entrada principal."""
    args = parse_args()
    
    # Cargar configuraciones
    print("📋 Cargando configuración...")
    config = merge_configs(
        args.config,
        args.model_config,
        args.data_config,
    )
    
    # Aplicar overrides desde CLI
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    
    # Modo debug
    if args.debug:
        print("⚠️ Modo DEBUG activado")
        config["training"]["epochs"] = 1
        config["training"]["fast_dev_run"] = True
    
    # Ejecutar entrenamiento
    try:
        trainer = train(config, args)
        print("\n✅ Entrenamiento completado exitosamente!")
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()

