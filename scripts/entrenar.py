#!/usr/bin/env python
"""
LLARRI-O1 - Script Principal de Entrenamiento
==============================================

Entrena el modelo LLARRI-O1 v3.0 hasta el máximo posible.

Uso:
    python scripts/entrenar.py
    python scripts/entrenar.py --epochs 50 --lr 0.001
    python scripts/entrenar.py --progresivo

Autor: Lucas Mella (Segunda Cabeza)
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Agregar el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1 import (
    crear_modelo_fractal,
    LlarriFractalConfig,
    Entrenador,
    EntrenadorConfig,
    EntrenadorProgresivo,
    ResourceDetector
)
from llarri_o1.utils.datos import cargar_mnist_plano


def main():
    parser = argparse.ArgumentParser(description='Entrenar LLARRI-O1 v3.0')
    
    # Modelo
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Dimensión oculta (default: 256)')
    parser.add_argument('--profundidad', type=int, default=-1,
                        help='Profundidad fractal (-1=auto)')
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=30,
                        help='Número de épocas (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    
    # Modo
    parser.add_argument('--progresivo', action='store_true',
                        help='Usar entrenamiento progresivo por niveles')
    parser.add_argument('--modo-hibrido', type=str, default='auto',
                        choices=['auto', 'gpu', 'hibrido', 'cpu'],
                        help='Modo de ejecución (default: auto)')
    
    # Otros
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directorio para checkpoints')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directorio de datos')
    parser.add_argument('--no-amp', action='store_true',
                        help='Deshabilitar mixed precision')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("  LLARRI-O1 v3.0 - ENTRENAMIENTO")
    print("  Trinity Fractal Recursivo Profundo")
    print("  Autor: Lucas Mella (Segunda Cabeza)")
    print("="*70)
    
    # Mostrar recursos
    detector = ResourceDetector()
    detector.print_info()
    
    # Cargar datos
    print("\n📦 Cargando datos MNIST...")
    train_loader, test_loader = cargar_mnist_plano(
        root=args.data_dir,
        batch_size=args.batch_size
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Configuración del modelo
    config = LlarriFractalConfig(
        hidden_dim=args.hidden_dim,
        profundidad_fractal=args.profundidad,
        modo_hibrido=args.modo_hibrido
    )
    
    if args.progresivo:
        # Entrenamiento progresivo
        print("\n🔄 Modo: ENTRENAMIENTO PROGRESIVO")
        
        entrenador = EntrenadorProgresivo(
            config,
            epochs_por_nivel=max(5, args.epochs // 5),
            epochs_finetune=max(10, args.epochs // 3),
            lr_inicial=args.lr,
            checkpoint_dir=os.path.join(args.checkpoint_dir, 'progresivo')
        )
        
        modelo = entrenador.entrenar_todos_niveles(train_loader, test_loader)
        print(entrenador.get_resumen())
        
    else:
        # Entrenamiento estándar
        print("\n🚀 Modo: ENTRENAMIENTO ESTÁNDAR")
        
        # Crear modelo
        modelo = crear_modelo_fractal(
            hidden_dim=args.hidden_dim,
            profundidad=args.profundidad,
            modo_hibrido=args.modo_hibrido
        )
        
        # Configuración del entrenador
        train_config = EntrenadorConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            usar_amp=not args.no_amp,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Entrenar
        entrenador = Entrenador(modelo, train_config)
        historia = entrenador.entrenar(train_loader, test_loader)
        
        # Mostrar resultados
        print("\n📊 RESULTADOS FINALES")
        print("="*50)
        print(f"Mejor accuracy: {max(historia['val_acc']):.2f}%")
        print(f"Última accuracy: {historia['val_acc'][-1]:.2f}%")
    
    # Mostrar estadísticas de compresión
    stats = modelo.get_compression_stats()
    print("\n📉 ESTADÍSTICAS DE COMPRESIÓN")
    print("="*50)
    print(f"Parámetros reales:    {stats['parametros_reales']:,}")
    print(f"Sin compartir serían: {stats['parametros_sin_compartir']:,}")
    print(f"Compresión:           {stats['compresion_porcentaje']:.1f}%")
    print(f"Factor de reducción:  {stats['factor_reduccion']:.1f}x")
    print(f"Modo de ejecución:    {stats['modo']}")
    
    print("\n" + "="*70)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
