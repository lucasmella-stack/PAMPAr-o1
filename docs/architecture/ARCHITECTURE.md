# LLARRI-O1 Architecture

Spanish version: [ARCHITECTURE.es.md](ARCHITECTURE.es.md)

## Authorship
- Author: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- Coordinator: Alvaro (Segunda Cabeza)
- Date: 2026-01-07

## Overview
LLARRI-O1 explores a parameter-sharing, fractal-like computation pattern:
- Inputs are projected into a hidden space.
- The hidden vector is split into 4 **quadrants**.
- Each quadrant is processed by a **progressive fractal pipeline**.
- Quadrants exchange information via cross-relations.
- Multiple “boxes” (processing stages) are connected via residual “keys”.

## Current implementation focus (v4.0 HyperComprimido)
Key ideas:
- **6 boxes total**: 3 primary processing boxes + 3 secondary processing boxes.
- **8 fractal levels**: `2 → 4 → 8 → 16 → 32 → 64 → 128 → 256` (sequential).
- **Binary cache** at level 2 (`CacheBinario`) for fast lookup of basic operations.
- **Secondary boxes include internal self-calculation** between intermediate values.

## Notes on scalability
- The maximum fractal level is constrained by `quad_dim = hidden_dim // 4`.
- Increasing `hidden_dim` increases `quad_dim`, which enables larger fractal levels (e.g. 256).
- GPU VRAM can still be a bottleneck during training due to optimizer states (AdamW) and activations.

## Where to look
- Package: [llarri_o1/](../../llarri_o1/)
  - Config: [llarri_o1/config.py](../../llarri_o1/config.py)
  - Model: [llarri_o1/model.py](../../llarri_o1/model.py)
  - Modules: [llarri_o1/modules/](../../llarri_o1/modules/) (cache, niveles, relaciones, cajas, flujo)
  - Training: [llarri_o1/training/trainer.py](../../llarri_o1/training/trainer.py)
  - Visualization: [llarri_o1/visualization/diagrams.py](../../llarri_o1/visualization/diagrams.py)
- Scripts: [scripts/train.py](../../scripts/train.py)
- Examples: [examples/basic_usage.py](../../examples/basic_usage.py)
- v4 diagrams: [diagrams/v4-current](../../diagrams/v4-current)
