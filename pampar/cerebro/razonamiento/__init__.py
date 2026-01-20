# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Razonamiento - Componentes de lógica deductiva

- axiomas.py: Reglas fundamentales (modus ponens, silogismo, etc.)
- inferencia.py: Motor de inferencia (TODO)
"""

from .axiomas import (
    MotorAxiomas,
    Axioma,
    ModusPonens,
    ModusTollens,
    Silogismo,
    TipoAxioma,
    Proposicion,
)

__all__ = [
    'MotorAxiomas',
    'Axioma',
    'ModusPonens',
    'ModusTollens',
    'Silogismo',
    'TipoAxioma',
    'Proposicion',
]
