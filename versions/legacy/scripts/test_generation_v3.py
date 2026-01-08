# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""Test de generación de texto con LLARRI v3 Multiescala."""

import torch
from llarri_o1.models.language_model_v3 import LLARRIv3, LLARRIv3Config


def main():
    # Cargar checkpoint
    print("Cargando modelo...")
    checkpoint = torch.load(
        'checkpoints/llarri_v3_best.pt', 
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        weights_only=False
    )
    config = checkpoint['config']
    
    # Crear modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LLARRIv3(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("=" * 60)
    print("LLARRI v3 Multiescala - Test de Generación")
    print("=" * 60)
    print(f"Parámetros: {model.get_num_params():,}")
    if 'best_val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['best_val_loss']:.4f}")
    elif 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"Device: {device}")
    print()
    
    # Prompts de prueba
    prompts = [
        "Once upon a time",
        "The little girl",
        "A magical forest",
        "One day, the cat",
        "The princess was",
    ]
    
    print("=" * 60)
    print("GENERACIÓN DE TEXTO")
    print("=" * 60)
    
    for prompt in prompts:
        print(f'\n📝 Prompt: "{prompt}"')
        print("-" * 50)
        
        for temp in [0.7, 1.0]:
            with torch.no_grad():
                text = model.generate(
                    prompt, 
                    max_new_tokens=100, 
                    temperatura=temp, 
                    top_p=0.9,
                    top_k=50
                )
            # Limpiar texto
            text = text.replace('\x00', '').strip()
            print(f"  T={temp}: {text}")
        
        print()
    
    # Test interactivo
    print("=" * 60)
    print("MODO INTERACTIVO (escribe 'q' para salir)")
    print("=" * 60)
    
    while True:
        prompt = input("\n🖊️  Tu prompt: ").strip()
        if prompt.lower() == 'q':
            break
        if not prompt:
            continue
            
        with torch.no_grad():
            text = model.generate(
                prompt, 
                max_new_tokens=150, 
                temperatura=0.8, 
                top_p=0.9,
                top_k=50
            )
        text = text.replace('\x00', '').strip()
        print(f"🤖 {text}")


if __name__ == "__main__":
    main()
