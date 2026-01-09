# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 — Cerebral Language Model v7.3
========================================

Arquitectura de Modelo de Lenguaje Cerebral con:
- 6 Módulos Especializados Acoplables (Lenguaje, Lógica, Matemáticas, Patrones, Contexto, Creatividad)
- Tálamo con Liderazgo: Un módulo LIDERA, los otros se ACOPLAN
- Coordinador Cerebral: Consenso real entre módulos
- Hipocampo: Memoria con LSH para recuperación O(1)

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)

Uso rápido:
    from llarri_o1 import LLARRIv73Liderazgo
    
    model = LLARRIv73Liderazgo(vocab_size=8000, dim=128)
    output = model(input_ids)
"""

__version__ = "7.3.0"
__author__ = "Lucas Ricardo Mella Chillemi"

# Language Model v7.3 - Arquitectura Cerebral con Liderazgo
from llarri_o1.models.language_model_v73 import LLARRIv73Liderazgo

# Módulos cerebrales
from llarri_o1.modules.cerebral.talamo_liderazgo import TalamoConLiderazgo
from llarri_o1.modules.cerebral.modulos_acoplables import (
    ModuloAcoplable,
    ModuloLenguajeAcoplable,
    ModuloLogicaAcoplable,
    ModuloMatematicasAcoplable,
    ModuloPatronesAcoplable,
    ModuloContextoAcoplable,
    ModuloCreatividadAcoplable,
)
from llarri_o1.modules.cerebral.integracion_liderazgo import CoordinadorCerebral
from llarri_o1.modules.cerebral.hipocampo import Hipocampo

# Utilidades
from llarri_o1.utils.device import get_device, print_device_info

__all__ = [
    # Modelo principal
    "LLARRIv73Liderazgo",
    # Módulos
    "TalamoConLiderazgo",
    "ModuloAcoplable",
    "ModuloLenguajeAcoplable",
    "ModuloLogicaAcoplable",
    "ModuloMatematicasAcoplable",
    "ModuloPatronesAcoplable",
    "ModuloContextoAcoplable",
    "ModuloCreatividadAcoplable",
    "CoordinadorCerebral",
    "Hipocampo",
    # Utils
    "get_device",
    "print_device_info",
]
