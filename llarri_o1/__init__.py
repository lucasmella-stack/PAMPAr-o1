# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 — Cerebral Language Model v7
======================================

Arquitectura de Modelo de Lenguaje Cerebral con:
- 6 Módulos Especializados (Lenguaje, Lógica, Matemáticas, Patrones, Contexto, Creatividad)
- Tálamo: Router con modulación continua (nunca apaga, solo modula 15%-100%)
- Spreading Activation: Módulos vecinos se alertan mutuamente
- Hipocampo: Memoria con LSH para recuperación O(1)
- Cuerpo Calloso: Integración de hemisferios
- Detector de Consenso: Early exit cuando módulos están de acuerdo

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)

Uso rápido:
    from llarri_o1 import LLARRIv7Cerebral
    
    model = LLARRIv7Cerebral(vocab_size=50257, d_model=128)
    output = model(input_ids)
"""

# Language Model v7 - Arquitectura Cerebral
from llarri_o1.models.language_model_v7 import (
    LLARRIv7Cerebral,
    LLARRIv7Mini,
    LLARRIv7Base,
    LLARRIv7Large,
)

# Módulos componentes
from llarri_o1.modules import (
    # Tokenización Transmutativa (TT)
    TokenizadorFractal,
    TokenizerConfig,
    # Embeddings Composicionales (ECN)
    EmbeddingComposicional,
    EmbeddingPosicionalFractal,
    # Bloque Fractal 6 Cajas (MPC + FPD + EEM + CGC)
    BloqueFractal,
    BloqueFractalConfig,
    CajaMezcla,
    CajaProcesa,
    CajaEvalua,
    CajaOutput,
    # LM Head
    LMHeadFractal,
    LMHeadConfig,
    # Cache Evolutivo Binario (CEB)
    CacheBinario,
    # v7 Cerebral
    Talamo,
    TalamoConMemoria,
    ModuloLenguaje,
    ModuloLogica,
    ModuloMatematicas,
    ModuloPatrones,
    ModuloContexto,
    ModuloCreatividad,
    IntegradorCerebral,
    Hipocampo,
)

__version__ = "7.0.0"
__author__ = "Lucas Ricardo Mella Chillemi"
__email__ = "lucas@segundacabeza.com"
__license__ = "AGPL-3.0-or-later"

__all__ = [
    # API principal v7
    "LLARRIv7Cerebral",
    "LLARRIv7Mini",
    "LLARRIv7Base", 
    "LLARRIv7Large",
    # Tokenización Transmutativa
    "TokenizadorFractal",
    "TokenizerConfig",
    # Embeddings Composicionales
    "EmbeddingComposicional",
    "EmbeddingPosicionalFractal",
    # Bloque Fractal 6 Cajas
    "BloqueFractal",
    "BloqueFractalConfig",
    "CajaMezcla",
    "CajaProcesa",
    "CajaEvalua",
    "CajaOutput",
    # LM Head
    "LMHeadFractal",
    "LMHeadConfig",
    # Cache
    "CacheBinario",
    # v7 Cerebral
    "Talamo",
    "TalamoConMemoria",
    "ModuloLenguaje",
    "ModuloLogica",
    "ModuloMatematicas",
    "ModuloPatrones",
    "ModuloContexto",
    "ModuloCreatividad",
    "IntegradorCerebral",
    "Hipocampo",
]
