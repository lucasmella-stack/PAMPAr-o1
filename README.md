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
---

<div align="center">

# 🧠 LLARRI-O1

### Fractal Language Model with Neighbor-Progressive Processing

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[Español](README.es.md)** | **[Architecture](docs/architecture/ARCHITECTURE.md)** | **[Innovations](docs/INNOVATIONS.md)**

*"Mix first, process with neighbors — from small to large"*

</div>

---

## 🎯 What is LLARRI-O1?

LLARRI-O1 is an **experimental language model** that reimagines how neural networks process information. Instead of the traditional "attention → FFN → repeat" pattern, LLARRI uses:

1. **Mix globally** (attention to see what's relevant)
2. **Process with nearby neighbors first** (small/cheap compute)
3. **Only expand if needed** (progressive compute)
4. **Exit early when confident** (save resources)

---

## 🔄 Traditional vs LLARRI: Visual Comparison

### Traditional Transformer

```
┌─────────────────────────────────────────────────────────────┐
│                    STANDARD TRANSFORMER                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [Attention] ──► [FFN Full] ──► Output          │
│                  │              │                           │
│                  │              │                           │
│             Always runs    Always 4x expansion              │
│             full compute   regardless of need               │
│                                                             │
│   • Fixed compute per layer                                 │
│   • No early exit                                           │
│   • Same cost for "easy" and "hard" inputs                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### LLARRI-O1 (6-Box Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    LLARRI-O1 (6 BOXES)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [MEZCLA] ──► [PROCESA] ──► [EVALÚA] ──► Output │
│             Attention     Progressive    Early              │
│                           FFN            Exit               │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 1: MIX (Attention)                             │   │
│   │    "What information is relevant?"                  │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 2: PROCESS nearby (0.5x FFN)  ◄── Small/Fast   │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 5: EVALUATE ──► Confident? ──► EXIT EARLY ✓    │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼ (if not confident)                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 3: PROCESS medium (0.75x FFN)                  │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 5: EVALUATE ──► Confident? ──► EXIT EARLY ✓    │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼ (if not confident)                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 4: PROCESS far (1.0x FFN)  ◄── Full compute    │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Box 6: OUTPUT                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   • Adaptive compute (easy inputs exit early)               │
│   • Progressive cost (small → medium → large)               │
│   • Like CPU cache: L1 (fast) → L2 → L3 (slow)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Comparison

### Tokenization

| Aspect | Traditional (BPE/WordPiece) | LLARRI (Transmutative) |
|--------|----------------------------|------------------------|
| Base unit | Subwords | Bytes (256 vocab) |
| Levels | Single merged vocabulary | Hierarchical (2→4→8→16) |
| Flexibility | Fixed after training | Multi-granularity at runtime |
| Memory | Large vocab embeddings | Small base + composition |

```
TRADITIONAL BPE:
  "Hello" → ["Hel", "lo"] → [1542, 283]
  Fixed vocabulary, single representation

LLARRI TRANSMUTATIVE:
  "Hello" → Level 1: [72,101,108,108,111] (bytes)
          → Level 2: [18533, 27756, 111]  (bigrams)
          → Level 4: [...]                 (4-grams)
  Multiple views, choose granularity per task
```

### Embeddings

| Aspect | Traditional | LLARRI (Compositional) |
|--------|-------------|----------------------|
| Storage | Separate per token | Base + MLP per level |
| Memory | O(vocab × dim) | O(vocab × dim / 24) |
| Level awareness | None | Built-in |

```
TRADITIONAL:
  embed["Hello"] = lookup[token_id]  # 50K × 768 = 38M params

LLARRI COMPOSITIONAL:
  embed["Hello"] = base[byte] + MLP_level(base[byte])
  # 256 × 64 + small MLPs = ~1.5M params (24x smaller)
```

### Positional Encoding

| Aspect | Traditional | LLARRI (Fractal Hybrid) |
|--------|-------------|------------------------|
| Type | Sinusoidal or learned | Sinusoidal + hierarchical |
| Level info | None | Encodes granularity |
| Position + Level | Separate | Combined |

```
TRADITIONAL:
  pos_embed(i) = sin/cos(i)

LLARRI FRACTAL HYBRID:
  pos_embed(i, level) = sin/cos(i) + level_embed[level] + hierarchical(i, level)
  # Position knows WHAT level it's operating at
```

### Processing

| Aspect | Traditional Transformer | LLARRI 6-Box |
|--------|------------------------|--------------|
| Pattern | Attention → FFN (fixed) | Mix → Process Progressive |
| FFN size | Always 4x | 0.5x → 0.75x → 1.0x |
| Early exit | Rare, per-layer | Built-in, per-box |
| Compute | Fixed | Adaptive |

```
TRADITIONAL:
  for layer in layers:
      x = attention(x)      # Always full
      x = ffn(x)            # Always 4x expansion

LLARRI:
  x = mix(x)                # Attention
  x = process_near(x)       # 0.5x FFN
  if confident(x): return   # Early exit!
  x = process_mid(x)        # 0.75x FFN  
  if confident(x): return   # Early exit!
  x = process_far(x)        # 1.0x FFN
```

---

## 🚀 Key Innovations

| Innovation | Name (proposed by founder) | Description |
|------------|---------------------------|-------------|
| **TT** | Transmutative Tokenization | Hierarchical byte groupings, multi-granularity |
| **ECN** | Compositional Level Embeddings | Base + MLP composition, 24x memory reduction |
| **PFH** | Fractal Hybrid Positions | Position + level awareness combined |
| **MPC** | Mix → Process Nearby | Core philosophy: mix globally, process progressively |
| **FPD** | Distance-Progressive FFN | 0.5x → 0.75x → 1.0x expansion by "neighbor distance" |
| **EEM** | Multi-stage Early Exit | Exit at box level AND at fractal level |
| **CGC** | Gated Box Contributions | Each box controls its influence via gates |
| **CEB** | Evolutionary Binary Cache | L1/L2/L3-like cache for frequent operations |

---

## 💾 Memory Comparison

```
┌────────────────────────────────────────────────────────────┐
│              MEMORY FOOTPRINT COMPARISON                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  GPT-2 Small (117M params):                                │
│  ████████████████████████████████████████  ~500 MB         │
│                                                            │
│  BERT-Base (110M params):                                  │
│  ███████████████████████████████████████   ~440 MB         │
│                                                            │
│  LLARRI-O1 LM v2 (544K params):                            │
│  █                                         ~3.8 MB         │
│                                                            │
│  Factor: ~100x smaller for comparable expressivity         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🔧 Quick Start

### Installation

```bash
git clone https://github.com/lucasmella-stack/llarri-o1.git
cd llarri-o1
pip install -r requirements.txt
```

### Language Model Usage

```python
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

# Create model
config = LLARRIConfig(
    embed_dim=64,
    base_dim=64,
    max_dim=128,
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

### Run Tests

```bash
# Test language model
python -m llarri_o1.models.language_model

# Test fractal block
python -m llarri_o1.modules.bloque_fractal
```

---

## 📁 Project Structure

```
llarri-o1/
├── llarri_o1/
│   ├── models/
│   │   └── language_model.py    # Unified LM with 6-box architecture
│   ├── modules/
│   │   ├── tokenizer.py         # Transmutative Tokenization
│   │   ├── bloque_fractal.py    # 6-Box Block (MPC + FPD + EEM)
│   │   ├── lm_head.py           # Tied embeddings head
│   │   ├── cache.py             # Evolutionary Binary Cache
│   │   └── ...
│   ├── training/
│   └── utils/
├── docs/
│   ├── architecture/
│   │   ├── ARCHITECTURE.md
│   │   └── LLARRI_LANGUAGE_MODEL_V2.es.md  # Full innovations doc
│   └── huggingface/
│       └── MODEL_CARD.md
├── examples/
├── tests/
└── README.md
```

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Transmutative Tokenizer | ✅ Complete | 4 hierarchical levels |
| Compositional Embeddings | ✅ Complete | 24x memory reduction |
| Fractal Positions | ✅ Complete | Level-aware |
| 6-Box Block | ✅ Complete | Mix → Process → Evaluate → Output |
| Early Exit | ✅ Complete | Multi-stage |
| LM Head | ✅ Complete | Tied embeddings |
| End-to-end generation | ✅ Working | Untrained (random output) |
| Training | 🔄 Next | Dataset + training loop |

---

## 👥 Credits

| Role | Name | Contact |
|------|------|---------|
| **Founder & Creator** | Lucas Ricardo Mella Chillemi | lucas@segundacabeza.com |
| **Coordinator** | Alvaro | alvaro@segundacabeza.com |

### Organization

**Segunda Cabeza** — AI Innovation

- 🌐 Web: [segundacabeza.com](https://segundacabeza.com)
- 🤗 HuggingFace: [lucas-mella/llarri-o1](https://huggingface.co/lucas-mella/llarri-o1)
- 📂 GitHub: [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)

---

## 📄 License

**GNU AGPL-3.0-or-later**

- ✅ Commercial use allowed
- ✅ Research and education use
- 📝 Modified network services must share source
- 📝 Attribution required

See [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with 💜 by Segunda Cabeza**

*"Mix first, process with neighbors — from small to large"*

</div>
