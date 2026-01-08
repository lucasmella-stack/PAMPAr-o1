# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7 - Generación de texto
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


def main():
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Crear modelo con misma config
    model = LLARRIv7Cerebral(
        vocab_size=256,
        dim=64,
        n_heads=2,
        usar_hipocampo=False,
        max_iteraciones=1,
        actividad_basal=0.2,
    )

    # Cargar pesos
    checkpoint = torch.load('checkpoints/llarri_v7_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'Modelo cargado - Epoch {checkpoint["epoch"]} - Val Loss: {checkpoint["val_loss"]:.4f}')

    # Función para generar
    def generate_text(prompt, max_tokens=100, temperature=0.8, repetition_penalty=1.3):
        # Convertir prompt a tokens (byte-level)
        input_ids = torch.tensor([[ord(c) % 256 for c in prompt]], device=device)
        
        # Generar
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.92,
                repetition_penalty=repetition_penalty,
            )
        
        # Decodificar
        tokens = output_ids[0].tolist()
        text = ''.join([chr(t) if 32 <= t < 127 else ' ' for t in tokens])
        return text

    print()
    print('=' * 60)
    print('GENERACIÓN DE TEXTO - LLARRI v7')
    print('=' * 60)

    # Probar varios prompts
    prompts = [
        'The quick brown fox jumps over the lazy dog. ',
        'In a world where technology advances rapidly, the future holds many possibilities. ',
        'Once upon a time in a land far away, there lived a young hero who sought adventure. ',
        'The scientist carefully examined the data before reaching her conclusion about the experiment. ',
    ]

    for prompt in prompts:
        print(f'\n Prompt: "{prompt[:40]}..."')
        print('-' * 40)
        result = generate_text(prompt, max_tokens=60, temperature=0.9, repetition_penalty=1.5)
        print(result)
        print()


if __name__ == "__main__":
    main()
