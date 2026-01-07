---
language:
- en
- es
license: agpl-3.0
library_name: pytorch
tags:
- pytorch
- language-model
- research
- experimental
- fractal
- parameter-sharing
- early-exit
- efficient-transformers
- adaptive-compute
- hierarchical-tokenization
pipeline_tag: text-generation
---

# 🧠 LLARRI-O1 — Fractal Language Model

<div align="center">

*"Mix first, process with neighbors — from small to large"*

[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[Español](MODEL_CARD.es.md)** | **[GitHub](https://github.com/lucasmella-stack/llarri-o1)** | **[Full Documentation](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.md)**

</div>

---

## What is LLARRI-O1?

LLARRI-O1 is an **experimental language model** that reimagines neural network processing through a **6-box architecture** inspired by CPU cache hierarchies (L1/L2/L3).

Instead of the traditional Transformer pattern (attention → full FFN → repeat), LLARRI:

1. 🔀 **Mixes globally** (attention to see what's relevant)
2. 📍 **Processes nearby first** (small/cheap compute)
3. 📈 **Expands only if needed** (progressive compute)
4. 🚀 **Exits early when confident** (saves resources)

---

## Traditional vs LLARRI

```
┌─────────────────────────────────────────────────────────────┐
│              TRADITIONAL TRANSFORMER                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [Attention] ──► [FFN 4x] ──► Output            │
│                                                             │
│   • Fixed compute per layer                                 │
│   • No early exit                                           │
│   • Same cost for ALL tokens                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              LLARRI-O1 (6 BOXES)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [MIX] ──► [PROCESS] ──► [EVALUATE] ──► Output  │
│                        0.5x→1.0x      early exit?           │
│                                                             │
│   Box 1: MIX (Attention) ─────────────────────────────────► │
│   Box 2: PROCESS nearby (0.5x FFN) ──► EXIT? ──────────────►│
│   Box 3: PROCESS medium (0.75x FFN) ──► EXIT? ─────────────►│
│   Box 4: PROCESS far (1.0x FFN) ───────────────────────────►│
│   Box 6: OUTPUT ◄──────────────────────────────────────────►│
│                                                             │
│   • Adaptive compute (easy inputs exit early)               │
│   • Progressive cost (0.5x → 0.75x → 1.0x)                  │
│   • Like CPU cache: L1 (fast) → L2 → L3 (slow)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8 Key Innovations

| Acronym | Name | What it does |
|---------|------|--------------|
| **TT** | Transmutative Tokenization | Multi-granularity bytes (2→4→8→16) |
| **ECN** | Compositional Level Embeddings | 24x memory reduction |
| **PFH** | Fractal Hybrid Positions | Position + level awareness |
| **MPC** | Mix → Process Nearby | Core architecture philosophy |
| **FPD** | Distance-Progressive FFN | 0.5x → 0.75x → 1.0x expansion |
| **EEM** | Multi-stage Early Exit | Exit at box AND fractal level |
| **CGC** | Gated Box Contributions | Learned contribution control |
| **CEB** | Evolutionary Binary Cache | L1/L2/L3 cache hierarchy |

*Names proposed by founder Lucas Ricardo Mella Chillemi*

📖 **[See full diagrams and comparisons →](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.md)**

---

## Comparison Table

| Aspect | Traditional | LLARRI-O1 |
|--------|-------------|-----------|
| **Tokenization** | BPE (fixed vocab) | Bytes + hierarchical levels |
| **Embeddings** | 50K × 768 = 38M | 256 × 64 + MLPs = ~400K |
| **FFN size** | Always 4x | 0.5x → 0.75x → 1.0x |
| **Early exit** | Rare/none | Built-in per box |
| **Compute** | Fixed | Adaptive |
| **Memory** | ~500 MB | ~3.8 MB |

---

## Model Details

| Property | Value |
|----------|-------|
| **Author** | Lucas Ricardo Mella Chillemi |
| **Organization** | Segunda Cabeza |
| **Coordinator** | Alvaro |
| **License** | AGPL-3.0-or-later |
| **Parameters** | ~544K |
| **Memory** | ~3.8 MB |
| **Version** | 2.0.0 |

---

## Quick Start

```python
from llarri_o1 import LLARRILanguageModel, LLARRIConfig

# Create model
config = LLARRIConfig(
    embed_dim=64,
    niveles=[2, 4, 8, 16],
    num_heads=4,
    ffn_expansion=2.0,
    num_vecinos=3,
    umbral_confianza=0.7
)

model = LLARRILanguageModel(config)

# Generate text
output = model.generate(
    prompt="Hello",
    max_new_tokens=50,
    temperatura=0.8,
    top_k=40
)
print(output)
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Transmutative Tokenizer (TT) | ✅ Complete |
| Compositional Embeddings (ECN) | ✅ Complete |
| Fractal Positions (PFH) | ✅ Complete |
| 6-Box Block (MPC + FPD) | ✅ Complete |
| Early Exit (EEM) | ✅ Complete |
| Gated Contributions (CGC) | ✅ Complete |
| Binary Cache (CEB) | ✅ Complete |
| End-to-end generation | ✅ Working |
| Training | 🔄 In progress |

---

## Intended Use

- ✅ Research on efficient transformer architectures
- ✅ Exploring adaptive compute mechanisms
- ✅ Educational purposes
- ⚠️ Not production-ready (untrained)

---

## Citation

```bibtex
@software{llarri_o1,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {LLARRI-O1: Fractal Language Model with Neighbor-Progressive Processing},
  year = {2026},
  organization = {Segunda Cabeza},
  url = {https://github.com/lucasmella-stack/llarri-o1}
}
```

---

## Links

- 📂 **GitHub:** [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)
- 📖 **Innovations:** [Full documentation with diagrams](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.md)
- 🏗️ **Architecture:** [Technical details](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/architecture/ARCHITECTURE.md)

---

## Contact

- **Lucas Ricardo Mella Chillemi** — lucas@segundacabeza.com
- **Alvaro (Coordinator)** — alvaro@segundacabeza.com
- **Web:** [segundacabeza.com](https://segundacabeza.com)

---

<div align="center">

**Made with 💜 by Segunda Cabeza**

*"Mix first, process with neighbors — from small to large"*

</div>
