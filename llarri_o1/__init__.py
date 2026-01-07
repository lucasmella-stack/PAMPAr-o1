# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
====================================================

Arquitectura de IA revolucionaria con:
- Recursión fractal hasta el límite matemático
- 99%+ de compresión de parámetros
- Modo híbrido CPU/GPU adaptativo
- Entrenamiento progresivo por cuadrantes

Autor: Lucas Mella (Segunda Cabeza)
Licencia: Propietaria con atribución
"""

__version__ = "3.0.0"
__author__ = "Lucas Mella"
__organization__ = "Segunda Cabeza"
__email__ = "lucas@segundacabeza.com"

# Modelos
from .models.fractal_profundo import (
    LlarriO1_FractalProfundo,
    LlarriFractalConfig,
    CuadranteFractal,
    CajaTrinityFractal,
    LlaveTrinity,
    crear_modelo_fractal,
    explorar_profundidades,
    ResourceManager
)

# Entrenamiento
from .training.entrenador import Entrenador, EntrenadorConfig
from .training.progresivo import EntrenadorProgresivo

# Utilidades
from .utils.recursos import (
    ResourceDetector,
    HybridMemoryManager,
    limpiar_memoria,
    get_memoria_usada
)
from .utils.datos import cargar_mnist, cargar_mnist_plano

__all__ = [
    # Versión
    '__version__',
    '__author__',
    '__organization__',
    
    # Modelos
    'LlarriO1_FractalProfundo',
    'LlarriFractalConfig',
    'CuadranteFractal',
    'CajaTrinityFractal',
    'LlaveTrinity',
    'crear_modelo_fractal',
    'explorar_profundidades',
    'ResourceManager',
    
    # Entrenamiento
    'Entrenador',
    'EntrenadorConfig',
    'EntrenadorProgresivo',
    
    # Utilidades
    'ResourceDetector',
    'HybridMemoryManager',
    'limpiar_memoria',
    'get_memoria_usada',
    'cargar_mnist',
    'cargar_mnist_plano',
]
