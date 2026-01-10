---
language:
- en
- es
license: agpl-3.0
library_name: pytorch
tags:
- pytorch
- language-model
- cerebral
- modular
- research
---

<div align="center">

# 🦙 PampaR

### Cerebral Language Model with Modular Architecture

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[Español](README.es.md)** | **[Architecture](docs/architecture/ARCHITECTURE.md)**

</div>

---

## 🎯 What is PampaR?

PampaR is a **modular language model** inspired by how the brain processes information. It uses specialized modules (neurons) coordinated by a central orchestrator (thalamus).

### Key Components

- **🧠 Tálamo (Thalamus)**: Orchestrates which modules process each input using learned rules (LLAVES)
- **🔗 Sinapsis (Synapses)**: Connections between modules for inter-module communication
- **⚡ 6 Specialized Neurons**:
  - Lenguaje (Language)
  - Lógica (Logic)  
  - Matemáticas (Mathematics)
  - Patrones (Patterns)
  - Contexto (Context)
  - Creatividad (Creativity)
- **📐 Axiomas**: Deductive reasoning engine (modus ponens, syllogism, etc.)
- **💾 Memoria**: Experience-based learning from successes/failures

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/lucasmella-stack/llarri-o1.git
cd llarri-o1

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training (5 epochs, 10M tokens)
python scripts/train.py --tokens 10M --epochs 5

# Resume from checkpoint
python scripts/train.py --resume

# Custom configuration
python scripts/train.py --batch-size 32 --lr 1e-4 --epochs 10
```

### Chat (Inference)

```bash
# Interactive chat
python scripts/chat.py

# With custom checkpoint
python scripts/chat.py --checkpoint checkpoints/pampar_best.pt

# Adjust generation parameters
python scripts/chat.py --temperature 0.7 --top_p 0.95
```

---

## 📊 Model Configurations

PampaR scales from 4GB to 80GB+ VRAM:

| Config | VRAM | Params | Dim | Layers | Heads |
|--------|------|--------|-----|--------|-------|
| LOCAL_4GB | 4GB | ~6M | 128 | 3 | 4 |
| SERVER_8GB | 8GB | ~25M | 256 | 4 | 8 |
| SERVER_24GB | 24GB | ~100M | 512 | 6 | 8 |
| SERVER_80GB | 80GB | ~300M | 768 | 8 | 12 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         PampaR                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [Embedding] ──► [CerebralBlock ×N] ──► Output  │
│                                    │                        │
│                                    ▼                        │
│                    ┌──────────────────────────┐            │
│                    │        TÁLAMO            │            │
│                    │   (LLAVES = routing)     │            │
│                    └──────────┬───────────────┘            │
│                               │                             │
│         ┌─────────┬─────────┬─┴─────┬─────────┬─────────┐  │
│         ▼         ▼         ▼       ▼         ▼         ▼  │
│    [Lenguaje] [Lógica] [Matemat] [Patron] [Context] [Creat]│
│         │         │         │       │         │         │  │
│         └─────────┴─────────┴───┬───┴─────────┴─────────┘  │
│                                 │                           │
│                                 ▼                           │
│                          [SINAPSIS]                        │
│                     (inter-module comms)                   │
│                                 │                           │
│                                 ▼                           │
│                          [AXIOMAS]                         │
│                    (deductive reasoning)                   │
│                                 │                           │
│                                 ▼                           │
│                          [LM Head]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
pampar/
├── __init__.py          # Main exports
├── config.py            # ConfigPampaR + presets
└── cerebro/
    ├── model.py         # PampaR main class
    ├── talamo.py        # Orchestrator with LLAVES
    ├── sinapsis.py      # Inter-module connections
    ├── neurona.py       # Base neuron class
    ├── modulos/         # 6 specialized neurons
    ├── razonamiento/    # Axiomas engine
    └── memoria/         # Experience memory

scripts/
├── train.py             # Training script
├── chat.py              # Interactive inference
├── server.py            # API server
└── download_corpus.py   # Download WikiText-103
```

---

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 4GB+ VRAM (minimum)

---

## 📜 License

AGPL-3.0-or-later — See [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Lucas Ricardo Mella Chillemi** (Segunda Cabeza)
- **Álvaro** (Segunda Cabeza) — Coordinator

---

<div align="center">

**Made with ❤️ in Argentina 🇦🇷**

</div>
