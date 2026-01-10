#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Test básico de PampaR v9
"""

import torch
from pampar import PampaR, LOCAL_4GB

def main():
    print("="*60)
    print("🧠 PampaR v9 - Test de Arquitectura Territorial")
    print("="*60)
    
    # Crear modelo
    print("\n1. Creando modelo...")
    model = PampaR(LOCAL_4GB)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Modelo creado: {params:,} parámetros")
    
    # Contar parámetros por componente
    print("\n2. Distribución de parámetros:")
    counts = model.contar_parametros()
    for key, val in counts.items():
        print(f"   - {key}: {val:,}")
    
    # Test forward
    print("\n3. Test forward pass...")
    x = torch.randint(0, 8000, (2, 64))
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    logits = out['logits']
    print(f"   ✓ Output logits shape: {logits.shape}")
    
    # Test con labels (loss)
    print("\n4. Test con loss...")
    labels = torch.randint(0, 8000, (2, 64))
    out_with_loss = model(x, labels=labels)
    loss = out_with_loss['loss']
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    # Test generate
    print("\n5. Test generación...")
    prompt = torch.randint(0, 8000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"   ✓ Generated shape: {generated.shape}")
    
    # Estado de fronteras
    print("\n6. Estado de fronteras:")
    estado = model.estado_fronteras()
    for bloque, fronteras in estado.items():
        print(f"   {bloque}:")
        for nombre, peso in fronteras.items():
            print(f"     - {nombre}: {peso:.2f}")
    
    print("\n" + "="*60)
    print("✅ TODOS LOS TESTS PASARON!")
    print("="*60)

if __name__ == "__main__":
    main()
