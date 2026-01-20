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
- territorial
- brain-inspired
- interpretable
- modular
- research
pipeline_tag: text-generation
model-index:
- name: PampaR-v9
  results:
  - task:
      type: text-generation
    dataset:
      name: WikiText-103
      type: wikitext
    metrics:
    - name: Perplexity
      type: perplexity
      value: 57.1
    - name: Parameters
      type: params
      value: 6834586
---

# 🦙 PampaR v9 — Cerebral Language Model

<div align="center">

*"An artificial brain where territories collaborate through frontiers"*

[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[Español](MODEL_CARD.es.md)** | **[GitHub](https://github.com/lucasmella-stack/llarri-o1)** | **[Architecture Docs](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/architecture/ARCHITECTURE.md)**

</div>

---

## 📄 Documentation PDFs

<table>
<tr>
<td width="50%" align="center">

### 🏗️ Architecture
<a href="https://github.com/lucasmella-stack/llarri-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Arquitectura_Territorial.pdf">
<img src="https://img.shields.io/badge/PDF-Architecture-red?style=for-the-badge&logo=adobe" alt="Architecture PDF"/>
</a>

</td>
<td width="50%" align="center">

### 📊 Benchmarks
<a href="https://github.com/lucasmella-stack/llarri-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Benchmarks_Comparacion.pdf">
<img src="https://img.shields.io/badge/PDF-Benchmarks-blue?style=for-the-badge&logo=adobe" alt="Benchmarks PDF"/>
</a>

</td>
</tr>
</table>

---

## What is PampaR v9?

> **"PampaR is an artificial brain where the thalamus orchestrates tokens toward specialized territories (Expressive, Contextual, Formal, Structural) that collaborate via bidirectional frontiers, combining explicit rules (LLAVES 70%) with learned attention (30%) to generate language."**

PampaR v9 is a **brain-inspired language model** with a unique **territorial architecture**:

- **4 Specialized Territories** (groupings of related neural modules)
- **6 Bidirectional Frontiers** (learned inter-territory connections)
- **1 Tálamo** (orchestrator with hybrid rule-based + neural routing)

---

## Architecture: Territorial Processing

```
Input → Embedding → [BloqueTerrritorial ×N] → LM Head → Output
                              ↓
                  Tálamo (LLAVES 70% + Atención 30%)
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                        ┌───────────────┐
│   EXPRESIVO   │◄────── Frontera ──────►│  CONTEXTUAL   │
│ Lang + Creat  │                        │   Contexto    │
└───────┬───────┘                        └───────┬───────┘
        │                                        │
        │◄─────── Fronteras Bidirec ────────────►│
        │                                        │
┌───────▼───────┐                        ┌───────▼───────┐
│    FORMAL     │◄────── Frontera ──────►│ ESTRUCTURAL   │
│    Lógica     │                        │ Patrón + Mat  │
└───────────────┘                        └───────────────┘
                              ↓
                       Axiomas (reasoning)
```

### 4 Territories

| Territory | Modules | Function |
|-----------|---------|----------|
| **Expresivo** | Lenguaje + Creatividad | Fluent text generation, novel ideas |
| **Contextual** | Contexto | Working memory, coherence |
| **Formal** | Lógica | Logical reasoning, rules |
| **Estructural** | Patrones + Matemáticas | Sequences, numbers, patterns |

### 6 Bidirectional Frontiers

| Connection | Strength | Function |
|------------|----------|----------|
| Expresivo ↔ Contextual | 0.8 | Narrative coherence |
| Expresivo ↔ Formal | 0.5 | Argumentation |
| Expresivo ↔ Estructural | 0.4 | Structured creativity |
| Contextual ↔ Formal | 0.6 | Logical context |
| Contextual ↔ Estructural | 0.5 | Pattern memory |
| Formal ↔ Estructural | 0.7 | Mathematical logic |

---

## Key Innovation: Hybrid Routing (LLAVES)

The **Tálamo** routes tokens using:

- **LLAVES (70%)**: Explicit rules for known patterns
  - "el", "la", "the" → Language territory
  - "0-9", "+", "=" → Mathematics territory
  
- **Learned Attention (30%)**: Neural network for novel patterns

**Why hybrid?** Provides **interpretability** — you can see exactly why a token was routed to a territory.

---

## Performance

| Property | Value |
|----------|-------|
| **Parameters** | 6,834,586 |
| **Validation Loss** | 4.05 |
| **Perplexity** | 57.1 |
| **Training Data** | WikiText-103 |
| **VRAM Usage** | 110 MB |

### Comparison

| Model | Parameters | Perplexity | Efficiency |
|-------|-----------|------------|------------|
| LSTM baseline | ~10M | 100-120 | 0.1M/PPL |
| **PampaR v9** | **6.8M** | **57.1** | **0.12M/PPL** ✨ |
| Transformer (small) | 44M | 65 | 0.68M/PPL |
| GPT-2 Small | 124M | 29-35 | 4M/PPL |

---

## Quick Start

```python
import torch
from pampar.cerebro import PampaR
from pampar.config import LOCAL_4GB
import sentencepiece as sp

# Load
tok = sp.SentencePieceProcessor()
tok.Load('data/tokenizer/llarri_bpe.model')

model = PampaR(LOCAL_4GB).cuda()
ckpt = torch.load('checkpoints/pampar_best.pt', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Generate
prompt = "The history of"
ids = tok.Encode(prompt)
x = torch.tensor([ids]).cuda()

with torch.no_grad():
    for _ in range(50):
        out = model(x)
        next_id = out['logits'][0, -1].argmax().item()
        x = torch.cat([x, torch.tensor([[next_id]]).cuda()], dim=1)

print(tok.Decode(x[0].tolist()))
```

---

## Model Configurations

| Config | VRAM | Params | Dim | Layers |
|--------|------|--------|-----|--------|
| LOCAL_4GB | 4GB | ~6.8M | 128 | 3 |
| SERVER_8GB | 8GB | ~25M | 256 | 4 |
| SERVER_24GB | 24GB | ~100M | 512 | 6 |
| SERVER_80GB | 80GB | ~300M | 768 | 8 |

---

## Intended Use

- ✅ Research on brain-inspired neural architectures
- ✅ Exploring interpretable routing mechanisms
- ✅ Efficient language modeling on limited hardware
- ⚠️ Not production-ready for commercial applications

---

## Citation

```bibtex
@software{pampar_v9,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {PampaR: Cerebral Language Model with Territorial Architecture},
  year = {2026},
  version = {9.0.0},
  organization = {Independent Researcher},
  url = {https://github.com/lucasmella-stack/PAMPAr-o1}
}
```

---

## Links

- 📂 **GitHub:** [lucasmella-stack/PAMPAr-o1](https://github.com/lucasmella-stack/PAMPAr-o1)
- 📄 **Architecture PDF:** [View](https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Arquitectura_Territorial.pdf)
- 📊 **Benchmarks PDF:** [View](https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Benchmarks_Comparacion.pdf)

---

## Contact

- **Lucas Ricardo Mella Chillemi** — lucas.mella@outlook.com
- **GitHub:** [lucasmella-stack/PAMPAr-o1](https://github.com/lucasmella-stack/PAMPAr-o1)

---

<div align="center">

**Made with ❤️ in Argentina 🇦🇷**

</div>
