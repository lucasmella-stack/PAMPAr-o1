---
language:
- en
license: agpl-3.0
library_name: pytorch
tags:
- pytorch
- mnist
- research
- experimental
---

# LLARRI-O1 — Model Card

Spanish version: [MODEL_CARD.es.md](MODEL_CARD.es.md)

## Model Details
- Name: LLARRI-O1 HyperComprimido (v4.0)
- Author: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- Coordinator: Alvaro (Segunda Cabeza)
- License: AGPL-3.0-or-later

## Intended Use
Research and experimentation on fractal computation / parameter sharing patterns.

## Limitations
- Training can be memory-intensive depending on `hidden_dim`, batch size, and optimizer.
- This repository is experimental; interfaces and results may change.

## Training Data
- MNIST (via torchvision) for current experiments.

## Evaluation
- v3.1 Cuadrantes: 98.61% on MNIST (historical result).
- v4.0 HyperComprimido: smoke tests + dev runs; see repo README for current status.

## Risks
- Misuse as a production-grade model without proper evaluation.
- License (AGPL) requires sharing source code modifications when used over a network.

## License
This model/code is released under **AGPL-3.0-or-later**.

## Citation
- See [CITATION.cff](../../CITATION.cff)

## Contact
- lucas@segundacabeza.com
