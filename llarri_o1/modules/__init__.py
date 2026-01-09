# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Módulos componentes de LLARRI-O1 v7.3 (Arquitectura Cerebral con Liderazgo).
"""

# Utilidades básicas
from llarri_o1.modules.cache import CacheBinario
from llarri_o1.modules.attention import (
    AttentionFractalProgresivo,
    AttentionConfig,
)
from llarri_o1.modules.lm_head import (
    LMHeadFractal,
    LMHeadConfig
)

# Módulos cerebrales v7.3 con liderazgo
from llarri_o1.modules.cerebral import (
    # Tálamo con liderazgo
    TalamoConLiderazgo,
    # Módulos acoplables
    ModuloAcoplable,
    ModuloLenguajeAcoplable,
    ModuloLogicaAcoplable,
    ModuloMatematicasAcoplable,
    ModuloPatronesAcoplable,
    ModuloContextoAcoplable,
    ModuloCreatividadAcoplable,
    # Integración
    CoordinadorCerebral,
    # Memoria
    Hipocampo,
)

__all__ = [
    # Utils
    "CacheBinario",
    "AttentionFractalProgresivo",
    "AttentionConfig",
    "LMHeadFractal",
    "LMHeadConfig",
    # v7.3 Cerebral con Liderazgo
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
]
