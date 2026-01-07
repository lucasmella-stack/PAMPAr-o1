"""
Ejemplo: Uso Básico de LLARRI-O1 v3.0
======================================

Este ejemplo muestra cómo usar el modelo LLARRI-O1 v3.0 con
el modo híbrido CPU/GPU.

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import sys
sys.path.insert(0, '..')

from llarri_o1 import (
    crear_modelo_fractal,
    LlarriFractalConfig,
    ResourceDetector
)


def main():
    print("\n" + "="*60)
    print("EJEMPLO: LLARRI-O1 v3.0 - Uso Básico")
    print("="*60)
    
    # 1. Detectar recursos
    print("\n1. DETECTANDO RECURSOS...")
    detector = ResourceDetector()
    detector.print_info()
    
    # 2. Crear modelo
    print("\n2. CREANDO MODELO...")
    modelo = crear_modelo_fractal(
        input_dim=784,      # MNIST: 28x28 = 784
        hidden_dim=256,     # Dimensión oculta
        output_dim=10,      # 10 clases (dígitos 0-9)
        modo_hibrido="auto" # Detecta automáticamente
    )
    
    # 3. Ver estructura
    print("\n3. ESTRUCTURA FRACTAL:")
    print(modelo.get_estructura_fractal())
    
    # 4. Estadísticas de compresión
    print("\n4. ESTADÍSTICAS DE COMPRESIÓN:")
    stats = modelo.get_compression_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # 5. Test forward pass
    print("\n5. TEST FORWARD PASS:")
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    print(f"   Input: {x.shape}")
    
    with torch.no_grad():
        output = modelo(x)
    
    print(f"   Output: {output.shape}")
    
    # Predicciones
    predictions = output.argmax(dim=1)
    print(f"   Predicciones: {predictions[:10].tolist()}...")
    
    # 6. Modo de ejecución
    print("\n6. MODO DE EJECUCIÓN:")
    print(f"   Modo: {modelo.rm.modo.upper()}")
    print(f"   Device pesado: {modelo.rm.device_pesado}")
    print(f"   Device ligero: {modelo.rm.device_ligero}")
    
    if modelo.rm.modo == "hibrido":
        print("\n   ✓ Modo HÍBRIDO activo:")
        print("     - Multiplicaciones de matrices → GPU")
        print("     - Dropout, LayerNorm, GELU → CPU")
        print("     - 'No usar topadora para botellas'")
    elif modelo.rm.modo == "gpu":
        print("\n   ✓ Modo GPU completo activo")
        print("     - Toda la computación en GPU")
    else:
        print("\n   ✓ Modo CPU activo")
        print("     - Toda la computación en CPU")
    
    print("\n" + "="*60)
    print("EJEMPLO COMPLETADO")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
