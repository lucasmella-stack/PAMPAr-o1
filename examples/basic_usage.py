# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Ejemplo: Uso Básico de LLARRI-O1 v4.0
=====================================

Este ejemplo muestra cómo usar el modelo LLARRI-O1 v4.0 HyperComprimido
para clasificación de dígitos MNIST.

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
"""

import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from llarri_o1 import LlarriO1, Config
from llarri_o1.utils import get_device, print_device_info


def main():
    print("=" * 50)
    print("LLARRI-O1 v4.0 - Ejemplo Básico")
    print("=" * 50)
    
    # Info del dispositivo
    print_device_info()
    device = get_device()
    
    # Crear modelo con configuración por defecto
    print("\n1. Creando modelo...")
    model = LlarriO1()
    model = model.to(device)
    model.eval()
    
    # Crear entrada dummy (batch de 4 imágenes MNIST)
    print("\n2. Creando entrada de prueba...")
    batch_size = 4
    x = torch.randn(batch_size, 784).to(device)  # MNIST: 28x28 = 784
    
    # Forward pass
    print("\n3. Ejecutando forward pass...")
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Predicciones
    predictions = output.argmax(dim=1)
    print(f"\n4. Predicciones: {predictions.tolist()}")
    
    # Probabilidades
    probs = torch.softmax(output, dim=1)
    max_probs = probs.max(dim=1)
    print(f"   Confianza: {[f'{p:.2%}' for p in max_probs.values.tolist()]}")
    
    print("\n" + "=" * 50)
    print("¡Modelo funcionando correctamente!")
    print("=" * 50)


if __name__ == "__main__":
    main()
