# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

"""
Tests para PampaR v9 (Arquitectura Territorial).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

# Agregar root al path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from pampar import PampaR, ConfigPampaR, LOCAL_4GB


def test_forward_backward_cpu():
    """Test básico de forward y backward en CPU."""
    config = ConfigPampaR(
        vocab_size=1000,
        dim=64,
        n_heads=2,
        n_capas=2,
        max_seq_len=32,
    )
    
    model = PampaR(config)
    model.eval()

    x = torch.randint(0, 1000, (2, 16))
    output = model(x)

    assert 'logits' in output, "Output debe contener 'logits'"
    assert output['logits'].shape == (2, 16, 1000), f"Shape esperado (2, 16, 1000), obtenido {output['logits'].shape}"

    # Test backward con loss
    model.train()
    labels = torch.randint(0, 1000, (2, 16))
    output_with_loss = model(x, labels=labels)
    loss = output_with_loss['loss']
    loss.backward()


def test_config_defaults():
    """Test configuración por defecto."""
    config = ConfigPampaR()
    
    assert config.vocab_size == 8000
    assert config.dim == 128
    assert config.n_heads == 4
    assert config.n_capas == 3
    assert config.peso_llaves == 0.7


def test_config_presets():
    """Test presets de configuración."""
    from pampar import LOCAL_4GB, LOCAL_4GB_MAX, SERVER_8GB
    
    assert LOCAL_4GB.dim == 128
    assert LOCAL_4GB_MAX.dim == 256
    assert SERVER_8GB.dim == 384


def test_model_parameters():
    """Test que el modelo tiene parámetros entrenables."""
    config = ConfigPampaR(
        vocab_size=1000,
        dim=64,
        n_heads=2,
        n_capas=2,
    )
    
    model = PampaR(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable == total_params  # Todos deben ser entrenables


def test_generate():
    """Test generación de texto."""
    config = ConfigPampaR(
        vocab_size=1000,
        dim=64,
        n_heads=2,
        n_capas=2,
        max_seq_len=32,
    )
    
    model = PampaR(config)
    model.eval()
    
    prompt = torch.randint(0, 1000, (1, 5))
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10)
    
    assert generated.shape[1] > prompt.shape[1], "Debe generar más tokens"
    assert generated.shape[1] <= 15, "No debe exceder prompt + max_new_tokens"


def test_territorial_architecture():
    """Test componentes de arquitectura territorial."""
    config = ConfigPampaR(
        vocab_size=1000,
        dim=64,
        n_heads=2,
        n_capas=2,
    )
    
    model = PampaR(config)
    
    # Verificar que tiene territorios
    assert hasattr(model, 'bloques'), "Modelo debe tener bloques territoriales"
    
    # Verificar estado de fronteras
    estado = model.estado_fronteras()
    assert len(estado) > 0, "Debe tener estado de fronteras"


def test_contar_parametros():
    """Test conteo de parámetros por componente."""
    config = ConfigPampaR(
        vocab_size=1000,
        dim=64,
        n_heads=2,
        n_capas=2,
    )
    
    model = PampaR(config)
    counts = model.contar_parametros()
    
    assert 'total' in counts
    assert 'embedding' in counts
    assert counts['total'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
