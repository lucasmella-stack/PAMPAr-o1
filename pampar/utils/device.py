# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Utilidades para detección de dispositivos (CPU/GPU).
"""

import torch


def get_device() -> torch.device:
    """
    Detecta el mejor dispositivo disponible.
    
    Returns:
        torch.device: cuda si hay GPU, cpu en caso contrario
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def print_device_info():
    """Imprime información del dispositivo disponible."""
    device = get_device()
    print(f"\n{'='*50}")
    print(f"Dispositivo: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM Total: {total_mem / 1e9:.1f} GB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            print(f"VRAM Usado: {allocated / 1e9:.2f} GB")
            print(f"VRAM Reservada: {cached / 1e9:.2f} GB")
    else:
        print("No hay GPU disponible, usando CPU")
    
    print(f"{'='*50}\n")
