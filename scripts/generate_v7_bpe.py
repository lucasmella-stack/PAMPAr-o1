# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.2 - Generación con BPE Tokenizer
"""

import torch
import tiktoken
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Tokenizer BPE
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Crear modelo
    model = LLARRIv7Cerebral(
        vocab_size=50257,  # GPT-2 vocab
        dim=128,
        n_heads=4,
        usar_hipocampo=False,
        max_iteraciones=1,
        actividad_basal=0.2,
    )
    
    # Cargar pesos
    checkpoint = torch.load('checkpoints/llarri_v7_bpe_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'Modelo cargado - Epoch {checkpoint["epoch"]} - Val Loss: {checkpoint["val_loss"]:.4f}')
    print(f'Val PPL: {checkpoint.get("val_ppl", "N/A")}')
    
    # Función para generar
    def generate_text(prompt, max_tokens=50, temperature=0.8, repetition_penalty=1.2):
        # Tokenizar prompt con BPE
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        # Generar
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.92,
                repetition_penalty=repetition_penalty,
            )
        
        # Decodificar con BPE
        tokens = output_ids[0].tolist()
        text = tokenizer.decode(tokens)
        return text
    
    print()
    print('=' * 60)
    print('GENERACIÓN DE TEXTO - LLARRI v7.2 (BPE)')
    print('=' * 60)
    
    # Probar varios prompts
    prompts = [
        'The quick brown fox',
        'Once upon a time',
        'In a world where',
        'The scientist discovered',
    ]
    
    for prompt in prompts:
        print(f'\n📝 Prompt: "{prompt}"')
        print('-' * 40)
        result = generate_text(prompt, max_tokens=40, temperature=0.9, repetition_penalty=1.3)
        print(result)
        print()


if __name__ == "__main__":
    main()
