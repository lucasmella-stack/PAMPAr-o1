# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
PampaR Cerebro v9 - Arquitectura Territorial

Definición en una frase:
"PampaR es un cerebro artificial donde el tálamo orquesta tokens hacia 
territorios especializados (Expresivo, Contextual, Formal, Estructural) 
que colaboran via fronteras bidireccionales, combinando reglas explícitas 
(LLAVES 70%) con atención aprendida (30%) para generar lenguaje."

Componentes v9:
- model.py: PampaR con BloqueTerrritorial
- talamo.py: Orquestador con LLAVES + TalamoTerritorial
- territorio.py: 4 territorios (Expresivo, Contextual, Formal, Estructural)
- frontera.py: 6 fronteras bidireccionales
- neurona.py: Clase base para módulos

Submódulos:
- modulos/: 6 neuronas especializadas
- razonamiento/: Axiomas y lógica
- memoria/: Experiencia y aprendizaje

Legacy (v8):
- sinapsis.py: Sistema de 18 conexiones (movido a versions/legacy/v8/)
"""

from .model import PampaR, BloqueTerrritorial
from pampar.config import ConfigPampaR
from .talamo import Talamo, TalamoTerritorial, LlaveModulo
from .territorio import Territorio, GestorTerritorios, TipoTerritorio
from .frontera import FronteraBidireccional, GestorFronteras
from .neurona import Neurona

__all__ = [
    # Modelo principal v9
    'PampaR',
    'ConfigPampaR',
    'BloqueTerrritorial',
    # Orquestación
    'Talamo',
    'TalamoTerritorial',
    'LlaveModulo',
    # Territorios v9
    'Territorio',
    'GestorTerritorios',
    'TipoTerritorio',
    # Fronteras v9
    'FronteraBidireccional',
    'GestorFronteras',
    # Base
    'Neurona',
]
