# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Utilidades para PAMPAr-o1 v9.

Módulos:
- device.py: Detección de dispositivos (CPU/GPU)
- data.py: Carga de datos (texto para LM, MNIST legacy)
"""

from pampar.utils.device import get_device, print_device_info
from pampar.utils.data import (
    TextDataset,
    cargar_corpus,
    crear_dataloader,
    get_mnist_loaders,  # Legacy, para compatibilidad
)

__all__ = [
    # Device
    "get_device",
    "print_device_info",
    # Data - Language Model v9
    "TextDataset",
    "cargar_corpus",
    "crear_dataloader",
    # Data - MNIST (legacy)
    "get_mnist_loaders",
]
