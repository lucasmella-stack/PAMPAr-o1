# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
PampaR Cerebro - Arquitectura neural modular

Componentes:
- model.py: PampaR, el modelo principal
- talamo.py: Orquestador con LLAVES
- sinapsis.py: Conexiones inter-módulo
- neurona.py: Clase base para módulos

Submódulos:
- modulos/: 6 neuronas especializadas
- razonamiento/: Axiomas y lógica
- memoria/: Experiencia y aprendizaje
"""

from .model import PampaR, CerebralBlock
from pampar.config import ConfigPampaR
from .talamo import Talamo, LlaveModulo
from .sinapsis import Sinapsis, TipoSinapsis, ReglaSinaptica
from .neurona import Neurona

__all__ = [
    # Modelo principal
    'PampaR',
    'ConfigPampaR',
    'CerebralBlock',
    # Orquestación
    'Talamo',
    'LlaveModulo',
    # Conexiones
    'Sinapsis',
    'TipoSinapsis',
    'ReglaSinaptica',
    # Base
    'Neurona',
]
