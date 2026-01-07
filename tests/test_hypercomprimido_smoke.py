# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

import torch


def test_hypercomprimido_forward_backward_cpu():
    # Import directly from src/ without importing the src package.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((repo_root / "src").resolve()))

    from llarri_o1_hypercomprimido import LlarriO1_HyperComprimido

    model = LlarriO1_HyperComprimido()
    model.eval()

    x = torch.randn(2, 784)
    y = model(x)

    assert y.shape == (2, 10)

    loss = y.sum()
    loss.backward()


def test_hypercomprimido_levels_include_256():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((repo_root / "src").resolve()))

    from llarri_o1_hypercomprimido import LlarriO1_HyperComprimido

    model = LlarriO1_HyperComprimido()
    assert model.config.niveles_fractales[-1] == 256
