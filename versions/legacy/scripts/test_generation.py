#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# SPDX-License-Identifier: Apache-2.0
"""
Test de generación de texto con LLARRI-O1.

Author: Lucas Ricardo Mella Chillemi
Organization: Segunda Cabeza
"""

import torch
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

def main():
    print("Cargando modelo entrenado...")
    
    # Cargar checkpoint con config guardada
    checkpoint = torch.load("checkpoints/best_lm_model.pt", map_location="cuda", weights_only=False)
    
    # Usar la config guardada
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"Usando config del checkpoint: embed_dim={config.embed_dim}, niveles={config.niveles}")
    else:
        config = LLARRIConfig()
    
    model = LLARRILanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()

    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print()

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a boy named",
        "There was a big",
    ]

    print("=" * 60)
    print("GENERACION DE TEXTO - LLARRI-O1 v2")
    print("=" * 60)

    for prompt in prompts:
        print(f'\n📝 Prompt: "{prompt}"')
        print("-" * 40)
        text = model.generate(
            prompt=prompt, 
            max_new_tokens=50, 
            temperatura=0.8, 
            top_k=40
        )
        print(f"   {text}")


if __name__ == "__main__":
    main()
