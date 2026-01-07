# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 — Fractal Language Model
==================================

Arquitectura de Modelo de Lenguaje con:
- 6 cajas (Mezcla → Procesa → Evalúa → Output)
- Tokenización Transmutativa (TT)
- Embeddings Composicionales por Nivel (ECN)
- FFN Progresivo por Distancia (FPD)
- Early Exit Multietapa (EEM)
- Cache Evolutivo Binario (CEB)

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)

Uso rápido:
    from llarri_o1 import LLARRILanguageModel, LLARRIConfig
    
    model = LLARRILanguageModel()
    output = model.generate("Hola", max_new_tokens=50)
"""

# Language Model (v2) - API principal
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

# Modelo base (v4 HyperComprimido) - para clasificación
from llarri_o1.model import LlarriO1, Config

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
    # Legacy/internos
    CuadranteProgresivo,
    CajaDatos,
    CajaCalculos,
    SistemaFlujoCompleto,
)

__version__ = "2.0.0"
__author__ = "Lucas Ricardo Mella Chillemi"
__email__ = "lucas@segundacabeza.com"
__license__ = "AGPL-3.0-or-later"

__all__ = [
    # API principal
    "LLARRILanguageModel",
    "LLARRIConfig",
    # Modelo base
    "LlarriO1",
    "Config",
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
    # Legacy
    "CuadranteProgresivo",
    "CajaDatos",
    "CajaCalculos",
    "SistemaFlujoCompleto",
]
