# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Módulos componentes de LLARRI-O1 v7 (Arquitectura Cerebral).
"""

from llarri_o1.modules.cache import CacheBinario
from llarri_o1.modules.tokenizer import (
    TokenizadorFractal, 
    TokenizerConfig, 
    EmbeddingFractal, 
    EmbeddingComposicional,
    EmbeddingPosicionalFractal
)
from llarri_o1.modules.attention import (
    AttentionFractalProgresivo,
    AttentionConfig,
    AttentionNivel,
    crear_causal_mask,
    crear_causal_mask_por_ventana
)
from llarri_o1.modules.lm_head import (
    LMHeadFractal,
    LMHeadConfig
)
from llarri_o1.modules.bloque_fractal import (
    BloqueFractal,
    BloqueFractalConfig,
    BloqueFractalMultinivel,
    CajaMezcla,
    CajaProcesa,
    CajaEvalua,
    CajaOutput
)

# Módulos cerebrales v7
from llarri_o1.modules.cerebral import (
    Talamo,
    TalamoConMemoria,
    ModuloLenguaje,
    ModuloLogica,
    ModuloMatematicas,
    ModuloPatrones,
    ModuloContexto,
    ModuloCreatividad,
    CuerpoCalloso,
    IntegradorCerebral,
    Hipocampo,
)

__all__ = [
    "CacheBinario",
    "TokenizadorFractal",
    "TokenizerConfig", 
    "EmbeddingFractal",
    "EmbeddingComposicional",
    "EmbeddingPosicionalFractal",
    "AttentionFractalProgresivo",
    "AttentionConfig",
    "AttentionNivel",
    "crear_causal_mask",
    "crear_causal_mask_por_ventana",
    "LMHeadFractal",
    "LMHeadConfig",
    "BloqueFractal",
    "BloqueFractalConfig",
    "BloqueFractalMultinivel",
    "CajaMezcla",
    "CajaProcesa",
    "CajaEvalua",
    "CajaOutput",
    # v7 Cerebral
    "Talamo",
    "TalamoConMemoria",
    "ModuloLenguaje",
    "ModuloLogica",
    "ModuloMatematicas",
    "ModuloPatrones",
    "ModuloContexto",
    "ModuloCreatividad",
    "CuerpoCalloso",
    "IntegradorCerebral",
    "Hipocampo",
]
