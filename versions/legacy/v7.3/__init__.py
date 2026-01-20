# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
LLARRI v7.3/7.4 - Arquitectura Cerebral con Liderazgo Dinámico

Módulos inspirados en la organización del cerebro humano:
- Liderazgo: un módulo domina según la tarea
- Acoplamiento: los otros módulos siguen al líder
- Consenso: decisiones coordinadas entre módulos
- Memoria: hipocampo con recuperación O(1)
"""

from .talamo_liderazgo import TalamoConLiderazgo
from .talamo_reglas import TalamoConReglas, DetectorDeContenido
from .modulos_acoplables import (
    ModuloAcoplable,
    ModuloLenguajeAcoplable,
    ModuloLogicaAcoplable, 
    ModuloMatematicasAcoplable,
    ModuloPatronesAcoplable,
    ModuloContextoAcoplable,
    ModuloCreatividadAcoplable,
)
from .integracion_liderazgo import CoordinadorCerebral
from .hipocampo import Hipocampo

__all__ = [
    'TalamoConLiderazgo',
    'TalamoConReglas',
    'DetectorDeContenido',
    'ModuloAcoplable',
    'ModuloLenguajeAcoplable',
    'ModuloLogicaAcoplable',
    'ModuloMatematicasAcoplable', 
    'ModuloPatronesAcoplable',
    'ModuloContextoAcoplable',
    'ModuloCreatividadAcoplable',
    'CoordinadorCerebral',
    'Hipocampo',
]
