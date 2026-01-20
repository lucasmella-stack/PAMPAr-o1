# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
#
# ⚠️ LEGACY CODE - For historical reference only
# This test references modules that no longer exist
#
"""Test de liderazgo (LEGACY)"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1 import LLARRIv73Liderazgo
import sentencepiece as spm

device = torch.device('cuda')
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('data/tokenizer/llarri_bpe.model')

# Cargar modelo entrenado
ckpt = torch.load('checkpoints/llarri_v7.3_liderazgo_best.pt', map_location=device)
model = LLARRIv73Liderazgo(vocab_size=8000, dim=128, n_heads=4, usar_hipocampo=True, capacidad_memoria=2000, max_iteraciones=1)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

# Test con diferentes prompts
prompts = ['The scientist', '2 + 2 =', 'Once upon a time', 'If x equals 5']
print("=" * 60)
print("   TEST DE LIDERAZGO - LLARRI v7.3")
print("=" * 60)

for p in prompts:
    ids = tokenizer.encode(p)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(x)
    stats = out['stats']
    print(f'\nPrompt: "{p}"')
    print(f'  Lider: {stats.get("lider", "N/A")}')
    
    print('  Modulación:')
    for k, v in sorted(stats.items()):
        if k.startswith('mod_'):
            nombre = k.replace('mod_', '')
            barra = '█' * int(v * 20) + '░' * (20 - int(v * 20))
            print(f'    {nombre:12} {barra} {v*100:.0f}%')
