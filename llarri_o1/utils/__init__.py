# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

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
