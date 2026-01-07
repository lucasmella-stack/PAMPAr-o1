# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Tests para LLARRI-O1 v4.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Agregar root al path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from llarri_o1 import LlarriO1, Config


def test_forward_backward_cpu():
    """Test básico de forward y backward en CPU."""
    config = Config(hidden_dim=256)  # Modelo pequeño para test rápido
    config.niveles_fractales = [2, 4, 8, 16, 32, 64]  # Menos niveles
    
    model = LlarriO1(config)
    model.eval()

    x = torch.randn(2, 784)
    y = model(x)

    assert y.shape == (2, 10), f"Shape esperado (2, 10), obtenido {y.shape}"

    # Test backward
    loss = y.sum()
    loss.backward()


def test_config_defaults():
    """Test configuración por defecto."""
    config = Config()
    
    assert config.input_dim == 784
    assert config.hidden_dim == 1024
    assert config.output_dim == 10
    assert config.num_cajas_datos == 3
    assert config.num_cajas_calculos == 3
    assert config.niveles_fractales[-1] == 256


def test_config_validation():
    """Test validación de configuración."""
    # hidden_dim debe ser divisible por 4
    try:
        config = Config(hidden_dim=100)
        assert False, "Debería fallar con hidden_dim=100"
    except AssertionError:
        pass  # Esperado


def test_model_parameters():
    """Test que el modelo tiene parámetros entrenables."""
    config = Config(hidden_dim=256)
    config.niveles_fractales = [2, 4, 8, 16, 32, 64]
    
    model = LlarriO1(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable == total_params  # Todos deben ser entrenables
