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
pipeline_tag: text-generation
---

# LLARRI-O1 — Model Card

**Spanish version:** [MODEL_CARD.es.md](MODEL_CARD.es.md)

## Model Description

LLARRI-O1 is an experimental language model that reimagines neural network processing through a **6-box architecture** inspired by CPU cache hierarchies (L1/L2/L3).

### Core Philosophy: "Mix First, Process with Neighbors"

Instead of the traditional Transformer pattern (attention → full FFN → repeat), LLARRI:
1. **Mixes globally** (attention)
2. **Processes with nearby neighbors first** (small FFN)
3. **Expands only if needed** (progressive FFN)
4. **Exits early when confident** (saves compute)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLARRI-O1 (6 BOXES)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Box 1: MIX (Attention) ─────────────────────────────────► │
│                                                             │
│   Box 2: PROCESS nearby (0.5x FFN) ──► Box 5: EVALUATE ──► │
│          ▲                               │                  │
│          │                               ▼                  │
│   Box 3: PROCESS medium (0.75x FFN) ◄── Continue?          │
│          │                               │                  │
│          ▼                               ▼                  │
│   Box 4: PROCESS far (1.0x FFN) ───────────────────────────►│
│                                                             │
│   Box 6: OUTPUT ◄──────────────────────────────────────────►│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Innovations

| Name | Description |
|------|-------------|
| **Transmutative Tokenization (TT)** | Hierarchical byte groupings (2→4→8→16), multi-granularity at runtime |
| **Compositional Level Embeddings (ECN)** | Base + MLP per level, 24x memory reduction |
| **Fractal Hybrid Positions (PFH)** | Position encoding aware of fractal level |
| **Mix → Process Nearby (MPC)** | Core philosophy: global attention, progressive local processing |
| **Distance-Progressive FFN (FPD)** | 0.5x → 0.75x → 1.0x expansion by "neighbor distance" |
| **Multi-stage Early Exit (EEM)** | Exit at box level AND fractal level |
| **Gated Box Contributions (CGC)** | Each processing box controls its influence |
| **Evolutionary Binary Cache (CEB)** | L1/L2/L3-like cache for frequent operations |

*Names proposed by the founder, Lucas Ricardo Mella Chillemi.*

## Traditional vs LLARRI

| Aspect | Traditional Transformer | LLARRI-O1 |
|--------|------------------------|-----------|
| FFN size | Always 4x | 0.5x → 0.75x → 1.0x |
| Early exit | Rare | Built-in per box |
| Compute | Fixed | Adaptive |
| Embeddings | O(vocab × dim) | O(vocab × dim / 24) |
| Position encoding | Sin/cos only | Sin/cos + level awareness |

## Model Details

- **Author:** Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- **Coordinator:** Alvaro (Segunda Cabeza)
- **License:** AGPL-3.0-or-later
- **Parameters:** ~544K (Language Model v2)
- **Memory:** ~3.8 MB

## Quick Start

```python
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

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
    temperatura=0.8
)
```

## Intended Use

- Research and experimentation on efficient transformer architectures
- Exploring adaptive compute and early exit mechanisms
- Educational purposes for understanding alternative LLM designs

## Limitations

- **Currently untrained** — generates random output
- Experimental architecture, interfaces may change
- Not production-ready

## Current Status

| Component | Status |
|-----------|--------|
| Transmutative Tokenizer | ✅ Complete |
| Compositional Embeddings | ✅ Complete |
| 6-Box Block | ✅ Complete |
| Early Exit | ✅ Complete |
| End-to-end generation | ✅ Working |
| Training | 🔄 In progress |

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

## Contact

- **Lucas Ricardo Mella Chillemi** — lucas@segundacabeza.com
- **Alvaro (Coordinator)** — alvaro@segundacabeza.com
- **Organization:** Segunda Cabeza

## License

AGPL-3.0-or-later — Commercial use allowed, modified network services must share source code.
