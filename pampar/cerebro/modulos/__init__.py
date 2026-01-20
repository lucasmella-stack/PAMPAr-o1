# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Módulos especializados del cerebro PampaR.

6 neuronas con dominios específicos:
- NeuronaLenguaje: gramática, sintaxis, fluidez
- NeuronaLogica: razonamiento, inferencia
- NeuronaMatematicas: números, operaciones
- NeuronaPatrones: secuencias, repeticiones
- NeuronaContexto: referencias, coherencia
- NeuronaCreatividad: generación, variabilidad
"""

from .especializados import (
    NeuronaLenguaje,
    NeuronaLogica,
    NeuronaMatematicas,
    NeuronaPatrones,
    NeuronaContexto,
    NeuronaCreatividad,
)

__all__ = [
    'NeuronaLenguaje',
    'NeuronaLogica',
    'NeuronaMatematicas',
    'NeuronaPatrones',
    'NeuronaContexto',
    'NeuronaCreatividad',
]
