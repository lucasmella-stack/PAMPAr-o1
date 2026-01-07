"""
Ejemplo: Modo Híbrido CPU/GPU
==============================

Demuestra cómo LLARRI-O1 delega operaciones entre CPU y GPU:
- GPU: Operaciones pesadas (multiplicación de matrices)
- CPU: Operaciones ligeras (dropout, normalización)

"No usar topadora para levantar botellas"

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import sys
sys.path.insert(0, '..')

from llarri_o1 import crear_modelo_fractal, LlarriFractalConfig
from llarri_o1.utils.recursos import ResourceDetector, HybridMemoryManager


def demostrar_hibrido():
    print("\n" + "="*60)
    print("DEMOSTRACIÓN: MODO HÍBRIDO CPU/GPU")
    print("="*60)
    
    # Detectar recursos
    detector = ResourceDetector()
    info = detector.get_info()
    
    print(f"\nRecursos detectados:")
    print(f"  GPU: {info.gpu_nombre if info.gpu_disponible else 'No disponible'}")
    print(f"  VRAM: {info.vram_libre_gb:.2f} GB libre")
    print(f"  RAM: {info.ram_libre_gb:.2f} GB libre")
    
    # Crear gestor híbrido
    print("\n--- Gestor de Memoria Híbrida ---")
    manager = HybridMemoryManager(umbral_vram_mb=500)
    print(f"Modo: {manager.get_modo()}")
    print(f"Device pesado: {manager.device_pesado}")
    print(f"Device ligero: {manager.device_ligero}")
    
    # Demostrar movimiento de tensores
    print("\n--- Movimiento de Tensores ---")
    tensor = torch.randn(1000, 1000)
    print(f"Tensor original: device={tensor.device}, shape={tensor.shape}")
    
    tensor_pesado = manager.to_device_pesado(tensor)
    print(f"En device pesado: device={tensor_pesado.device}")
    
    tensor_ligero = manager.to_device_ligero(tensor)
    print(f"En device ligero: device={tensor_ligero.device}")
    
    # Crear modelo con diferentes modos
    print("\n--- Modelos con Diferentes Modos ---")
    
    for modo in ['auto', 'gpu', 'hibrido', 'cpu']:
        try:
            config = LlarriFractalConfig(
                hidden_dim=128,
                modo_hibrido=modo
            )
            modelo = crear_modelo_fractal(
                hidden_dim=128,
                modo_hibrido=modo
            )
            modo_real = modelo.rm.modo
            print(f"\n  Modo '{modo}' → Ejecuta como '{modo_real}'")
            print(f"    Device pesado: {modelo.rm.device_pesado}")
            print(f"    Device ligero: {modelo.rm.device_ligero}")
        except Exception as e:
            print(f"  Modo '{modo}': Error - {e}")
    
    # Explicación del principio
    print("\n" + "="*60)
    print("PRINCIPIO: 'NO USAR TOPADORA PARA BOTELLAS'")
    print("="*60)
    print("""
    El modo híbrido implementa el principio de eficiencia:
    
    🔨 GPU (Topadora) - Para trabajo pesado:
       • Multiplicación de matrices grandes
       • Convoluciones
       • Atención (attention)
    
    ✋ CPU (Mano) - Para trabajo ligero:
       • Dropout (aleatorio simple)
       • LayerNorm (normalización)
       • Activaciones (GELU, ReLU)
       • Conexiones residuales
    
    Resultado:
       • Menor uso de VRAM
       • Mayor throughput
       • Funciona en GPUs con poca memoria
    """)


if __name__ == "__main__":
    demostrar_hibrido()
