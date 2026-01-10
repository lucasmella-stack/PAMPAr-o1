# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Entrenamiento para PampaR.
"""

from pampar.training.trainer import Trainer
from pampar.training.adaptive_trainer import AdaptiveTrainer

__all__ = ["Trainer", "AdaptiveTrainer"]
