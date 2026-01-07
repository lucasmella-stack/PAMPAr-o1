# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 - Módulo de Entrenamiento
====================================

Entrenadores para el modelo fractal.

Autor: Lucas Mella (Segunda Cabeza)
"""

from .entrenador import Entrenador, EntrenadorConfig
from .progresivo import EntrenadorProgresivo

__all__ = [
    'Entrenador',
    'EntrenadorConfig',
    'EntrenadorProgresivo'
]
