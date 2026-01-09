# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.2 - Generación con BPE Custom Tokenizer
"""

import torch
import sentencepiece as spm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Cargar tokenizer BPE custom
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer/llarri_bpe.model')
    print(f'Tokenizer cargado - Vocab: {tokenizer.get_piece_size()}')
    
    # Cargar checkpoint para ver config
    checkpoint = torch.load('checkpoints/llarri_v7.2_bpe_best.pt', map_location=device)
    config = checkpoint['config']
    
    # Crear modelo con la misma config
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        usar_hipocampo=True,
        capacidad_memoria=2000,  # Coincidir con checkpoint
        max_iteraciones=1,
        actividad_basal=0.2,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_ppl = torch.exp(torch.tensor(checkpoint['val_loss'])).item()
    print(f'Modelo cargado - Epoch {checkpoint["epoch"]} - Val Loss: {checkpoint["val_loss"]:.4f} - PPL: {val_ppl:.2f}')
    
    # Función para generar
    def generate_text(prompt, max_tokens=50, temperature=0.8, repetition_penalty=1.2):
        # Tokenizar con BPE custom
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        # Generar
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.92,
                repetition_penalty=repetition_penalty,
            )
        
        # Decodificar
        tokens = output_ids[0].tolist()
        text = tokenizer.decode(tokens)
        return text
    
    print()
    print('=' * 60)
    print('GENERACIÓN DE TEXTO - LLARRI v7.2 (BPE Custom)')
    print('=' * 60)
    
    # Probar varios prompts
    prompts = [
        'The quick brown fox',
        'Once upon a time',
        'In a world where',
        'The scientist discovered',
        'Machine learning is',
    ]
    
    for prompt in prompts:
        print(f'\n📝 Prompt: "{prompt}"')
        print('-' * 40)
        try:
            result = generate_text(prompt, max_tokens=40, temperature=0.9, repetition_penalty=1.3)
            print(result)
        except Exception as e:
            print(f'Error: {e}')
        print()


if __name__ == "__main__":
    main()
