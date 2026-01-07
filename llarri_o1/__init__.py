# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v4.0 HyperComprimido
==============================

Arquitectura fractal con:
- 6 cajas (3 datos + 3 cálculos)
- 8 niveles fractales (2 → 256)
- Flujo completo IDA/VUELTA/BIDI
- Cache binario para nivel 2

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)
"""

from llarri_o1.model import LlarriO1, Config
from llarri_o1.modules import (
    CacheBinario,
    CuadranteProgresivo,
    CajaDatos,
    CajaCalculos,
    SistemaFlujoCompleto,
)

__version__ = "4.0.0"
__author__ = "Lucas Ricardo Mella Chillemi"
__email__ = "lucas@segundacabeza.com"
__license__ = "AGPL-3.0-or-later"

__all__ = [
    "LlarriO1",
    "Config",
    "CacheBinario",
    "CuadranteProgresivo",
    "CajaDatos",
    "CajaCalculos",
    "SistemaFlujoCompleto",
]
