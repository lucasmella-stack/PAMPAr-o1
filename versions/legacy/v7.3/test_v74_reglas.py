# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
#
# ⚠️ LEGACY CODE - For historical reference only
# This test references modules that no longer exist
#
"""Test del Tálamo v7.4 con Reglas (LEGACY)"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm
from llarri_o1.models.language_model_v74 import LLARRIv74Orquestado

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('data/tokenizer/llarri_bpe.model')

# Modelo
print("Creando modelo v7.4...")
model = LLARRIv74Orquestado(
    vocab_size=8000, 
    dim=128, 
    n_heads=4, 
    usar_hipocampo=True, 
    capacidad_memoria=2000
)
model.inicializar_talamo(tokenizer)
model = model.to(device)
model.eval()

# Test
print("\n" + "=" * 60)
print("   TEST DE LIDERAZGO - LLARRI v7.4 (CON REGLAS)")
print("=" * 60)

test_cases = [
    ("The cat sat on the mat", "lenguaje"),      # Artículos, preposiciones
    ("2 + 2 = 4", "matematicas"),                 # Números, operadores
    ("If it rains then stay home", "logica"),     # Condicionales
    ("Once upon a time in a land", "contexto"),   # Narrativa temporal
    ("First step second step third", "patrones"), # Secuencias
    ("Imagine a beautiful dream", "creatividad"), # Creatividad
]

for prompt, esperado in test_cases:
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], device=device)
    
    with torch.no_grad():
        out = model(x)
    
    stats = out['stats']
    lider = stats.get('lider', '?')
    match = "✅" if lider == esperado else "❌"
    
    print(f'\n{match} Prompt: "{prompt}"')
    print(f'   Líder: {lider} (esperado: {esperado})')
    print(f'   Score líder: {stats.get("lider_score", 0):.3f}')
    
    # Mostrar scores de todos
    print('   Scores:')
    for nombre in model.ORDEN_MODULOS:
        score = stats.get(f'score_{nombre}', 0)
        mod = stats.get(f'mod_{nombre}', 0)
        barra = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        marker = " ← LÍDER" if nombre == lider else ""
        print(f'      {nombre:12} {barra} {score:.2f} (mod: {mod:.0%}){marker}')
