# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi <lucas@segundacabeza.com>
# Coordinator: Alvaro <alvaro@segundacabeza.com>
"""
LLARRI v7 - Arquitectura Cerebral

Módulos inspirados en la organización del cerebro humano:
- Especialización: cada módulo tiene una tarea específica
- No interferencia: los módulos no se meten en el dominio de otros  
- Modulación continua: nada se apaga, todo se modula
- Propagación: la activación se propaga a módulos vecinos
- Integración ordenada: comunicación via protocolos claros
"""

from .talamo import Talamo, TalamoConMemoria
from .modulos_especializados import (
    ModuloLenguaje,
    ModuloLogica, 
    ModuloMatematicas,
    ModuloPatrones,
    ModuloContexto,
    ModuloCreatividad,
)
from .integracion import CuerpoCalloso, IntegradorCerebral
from .hipocampo import Hipocampo

__all__ = [
    'Talamo',
    'TalamoConMemoria',
    'ModuloLenguaje',
    'ModuloLogica',
    'ModuloMatematicas', 
    'ModuloPatrones',
    'ModuloContexto',
    'ModuloCreatividad',
    'CuerpoCalloso',
    'IntegradorCerebral',
    'Hipocampo',
]
