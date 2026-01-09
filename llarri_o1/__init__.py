# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 — Cerebral Language Model v8
======================================

Arquitectura de Modelo de Lenguaje Cerebral con:
- 6 Módulos Especializados (Lenguaje, Lógica, Matemáticas, Patrones, Contexto, Creatividad)
- Tálamo con LLAVES: Reglas explícitas de dominio + atención aprendida
- Sinapsis: Conexiones lógicas entre módulos
- Axiomas: Razonamiento deductivo (modus ponens, silogismo, etc.)
- Memoria de Experiencia: Aprendizaje a partir de éxitos/fracasos

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)

Uso rápido:
    from llarri_o1 import LLARRIv8, ConfigLLARRI
    
    config = ConfigLLARRI(vocab_size=8000, dim=256)
    model = LLARRIv8(config)
    output = model(input_ids)
"""

__version__ = "8.0.0"
__author__ = "Lucas Ricardo Mella Chillemi"

# ============================================
# Modelo Principal v8
# ============================================
from llarri_o1.cerebro import (
    LLARRIv8,
    ConfigLLARRI,
    CerebralBlock,
    Talamo,
    Sinapsis,
    Neurona,
)

# Módulos especializados
from llarri_o1.cerebro.modulos import (
    NeuronaLenguaje,
    NeuronaLogica,
    NeuronaMatematicas,
    NeuronaPatrones,
    NeuronaContexto,
    NeuronaCreatividad,
)

# Razonamiento
from llarri_o1.cerebro.razonamiento import MotorAxiomas

# Memoria
from llarri_o1.cerebro.memoria import MemoriaExperiencia

# Utilidades (si existen)
try:
    from llarri_o1.utils.device import get_device, print_device_info
except ImportError:
    get_device = None
    print_device_info = None

__all__ = [
    # Modelo principal
    "LLARRIv8",
    "ConfigLLARRI",
    "CerebralBlock",
    # Orquestación
    "Talamo",
    "Sinapsis",
    "Neurona",
    # Módulos especializados
    "NeuronaLenguaje",
    "NeuronaLogica",
    "NeuronaMatematicas",
    "NeuronaPatrones",
    "NeuronaContexto",
    "NeuronaCreatividad",
    # Razonamiento
    "MotorAxiomas",
    # Memoria
    "MemoriaExperiencia",
    # Utils
    "get_device",
    "print_device_info",
]
