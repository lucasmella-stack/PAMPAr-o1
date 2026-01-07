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
