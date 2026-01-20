#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Script de entrenamiento para LLARRI-O1.

Uso:
    # Entrenamiento adaptativo (auto-detecta capacidad)
    python scripts/train.py --epochs 10 --batch-size 32
    
    # Forzar modo full (todos los niveles juntos)
    python scripts/train.py --epochs 10 --mode full
    
    # Forzar modo progresivo (por niveles)
    python scripts/train.py --epochs 10 --mode progressive

Author: Lucas Ricardo Mella Chillemi (Independent Researcher)
"""

import argparse
import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1 import LlarriO1, Config
from llarri_o1.training import Trainer, AdaptiveTrainer
from llarri_o1.utils import get_mnist_loaders, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description='Entrenar LLARRI-O1')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Dimensión oculta')
    parser.add_argument('--no-amp', action='store_true', help='Desactivar mixed precision')
    parser.add_argument('--accumulation', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint a cargar')
    parser.add_argument('--mode', type=str, choices=['auto', 'full', 'progressive'], 
                        default='auto', help='Modo de entrenamiento')
    parser.add_argument('--legacy', action='store_true', help='Usar trainer legacy (no adaptativo)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("LLARRI-O1 v4.0 - Entrenamiento")
    print("="*60)
    
    # Info del dispositivo
    print_device_info()
    
    # Configuración
    config = Config(hidden_dim=args.hidden_dim)
    
    # Modelo
    print("\nInicializando modelo...")
    model = LlarriO1(config)
    
    # Datos
    print(f"Cargando datos (batch_size={args.batch_size})...")
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    
    # Elegir trainer
    if args.legacy:
        # Trainer original
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=args.lr,
            use_amp=not args.no_amp,
            accumulation_steps=args.accumulation,
        )
    else:
        # Trainer adaptativo
        force_mode = None if args.mode == 'auto' else args.mode
        trainer = AdaptiveTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=args.lr,
            use_amp=not args.no_amp,
            force_mode=force_mode,
        )
    
    # Cargar checkpoint si se especificó
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Entrenar
    trainer.train(epochs=args.epochs)
    
    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
    main()
