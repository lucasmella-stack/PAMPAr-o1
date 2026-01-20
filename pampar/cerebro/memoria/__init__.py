# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Memoria - Componentes de almacenamiento y aprendizaje

- experiencia.py: Memoria de éxitos/fracasos para aprender de resultados
- hipocampo.py: Memoria a largo plazo (TODO)
- reflexion.py: Auto-evaluación (TODO)
"""

from .experiencia import (
    MemoriaExperiencia,
    Experiencia,
)

__all__ = [
    'MemoriaExperiencia',
    'Experiencia',
]
