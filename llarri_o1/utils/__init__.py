"""
LLARRI-O1 - Módulo de Utilidades
================================

Recursos, datos y helpers.

Autor: Lucas Mella (Segunda Cabeza)
"""

from .recursos import ResourceDetector, HybridMemoryManager
from .datos import cargar_mnist, crear_dataloaders, get_dataset_info

__all__ = [
    'ResourceDetector',
    'HybridMemoryManager',
    'cargar_mnist',
    'crear_dataloaders',
    'get_dataset_info'
]
