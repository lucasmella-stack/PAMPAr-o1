# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 

"""
Utilidades para PampaR.
"""

from pampar.utils.data import get_mnist_loaders
from pampar.utils.device import get_device, print_device_info

__all__ = [
    "get_mnist_loaders",
    "get_device",
    "print_device_info",
]
