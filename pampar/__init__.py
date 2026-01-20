# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 

"""
PampaR v9 — Modelo de Lenguaje Cerebral con Territorios
========================================================

Definición en una frase:
"PampaR es un cerebro artificial donde el tálamo orquesta tokens hacia 
territorios especializados (Expresivo, Contextual, Formal, Estructural) 
que colaboran via fronteras bidireccionales, combinando reglas explícitas 
(LLAVES 70%) con atención aprendida (30%) para generar lenguaje."

Arquitectura v9:
- TÁLAMO: Orquestador con LLAVES (reglas + aprendizaje)
- 4 TERRITORIOS:
  · Expresivo: Lenguaje + Creatividad
  · Contextual: Contexto
  · Formal: Lógica
  · Estructural: Patrones + Matemáticas
- 6 FRONTERAS: Conexiones bidireccionales entre territorios
- AXIOMAS: Razonamiento deductivo (opcional)
- MEMORIA: Experiencia acumulada (opcional)

Autor: Lucas Ricardo Mella Chillemi
Contacto: lucas.mella@outlook.com

Uso rápido:
    from pampar import PampaR, ConfigPampaR, LOCAL_4GB
    
    model = PampaR(LOCAL_4GB)
    output = model(input_ids)
"""

__version__ = "9.0.0"
__author__ = "Lucas Ricardo Mella Chillemi"

# ============================================
# Modelo Principal v9
# ============================================
from pampar.cerebro import (
    PampaR,
    ConfigPampaR,
    BloqueTerrritorial,
    Talamo,
    TalamoTerritorial,
    Territorio,
    GestorTerritorios,
    FronteraBidireccional,
    GestorFronteras,
    Neurona,
)

# Configuraciones predefinidas
from pampar.config import (
    LOCAL_4GB,
    LOCAL_4GB_MAX,
    SERVER_8GB,
    SERVER_24GB,
    SERVER_80GB,
)

# Módulos especializados
from pampar.cerebro.modulos import (
    NeuronaLenguaje,
    NeuronaLogica,
    NeuronaMatematicas,
    NeuronaPatrones,
    NeuronaContexto,
    NeuronaCreatividad,
)

# Razonamiento
from pampar.cerebro.razonamiento import MotorAxiomas

# Memoria
from pampar.cerebro.memoria import MemoriaExperiencia

# Utilidades
try:
    from pampar.utils.device import get_device, print_device_info
except ImportError:
    get_device = None
    print_device_info = None

__all__ = [
    # Modelo principal v9
    "PampaR",
    "ConfigPampaR",
    "BloqueTerrritorial",
    # Configuraciones
    "LOCAL_4GB",
    "LOCAL_4GB_MAX",
    "SERVER_8GB",
    "SERVER_24GB",
    "SERVER_80GB",
    # Orquestación
    "Talamo",
    "TalamoTerritorial",
    # Territorios v9
    "Territorio",
    "GestorTerritorios",
    # Fronteras v9
    "FronteraBidireccional",
    "GestorFronteras",
    # Base
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
