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
- modular
- research
---

<div align="center">

<img src="PAMPAr-o1-imagen.png" alt="PampaR Logo" width="200"/>

# 🦙 PampaR v9

### Cerebral Language Model with Territorial Architecture

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![DOI](https://img.shields.io/badge/DOI-Pending-green.svg)](#)

**[Español](README.es.md)** | **[Technical Paper](docs/PAPER_TECNICO.md)** | **[Architecture](docs/architecture/ARCHITECTURE.md)** | **[Benchmarks](docs/BENCHMARK_RESULTS.md)**

</div>

---

## 👤 For Recruiters / Employers

> **TL;DR**: This is an original AI architecture developed from scratch by a self-taught developer, achieving results competitive with published research while using minimal resources.

| What This Demonstrates |
|------------------------|
| ✅ **Independent Research** — Novel architecture designed without academic supervision |
| ✅ **Full-Stack ML** — Data processing, model design, training infrastructure, evaluation |
| ✅ **Resource Optimization** — 14M params trained on 4GB VRAM consumer GPU |
| ✅ **Documentation** — Technical papers, diagrams, reproducible code |
| ✅ **Software Engineering** — Clean Python, modular design, tests, CI-ready |

**Key Achievement**: Outperforms LSTM (24M params) and Transformer-XL Small (24M params) with **42% fewer parameters**.

---

## 🏆 Highlights

> ⚠️ **Experimental Research**: This is a work-in-progress exploring brain-inspired architectures. Results are preliminary and require further validation.

<table>
<tr>
<td align="center"><b>14M</b><br/>Parameters</td>
<td align="center"><b>~45</b><br/>Perplexity*</td>
<td align="center"><b>250M+</b><br/>Tokens trained</td>
<td align="center"><b>4GB</b><br/>VRAM (GTX 1650)</td>
</tr>
</table>

*Single run on WikiText-103. See [limitations](#limitations) for caveats.

> **Experimental architecture** that shows promising results compared to LSTM and Transformer-XL (24M params) with 42% fewer parameters.  
> Trained entirely on **consumer hardware** — no cloud, no A100s.

---

## 📄 Documentation

<table>
<tr>
<td width="50%" align="center">

### 🏗️ Architecture Diagram
<a href="diagrams/v9-territorial/PampaR_v9_Arquitectura_Territorial.pdf">
<img src="https://img.shields.io/badge/PDF-Architecture-red?style=for-the-badge&logo=adobe" alt="Architecture PDF"/>
</a>

*Complete territorial architecture with frontiers and tálamo routing*

</td>
<td width="50%" align="center">

### 📊 Benchmarks & Comparison
<a href="diagrams/v9-territorial/PampaR_v9_Benchmarks_Comparacion.pdf">
<img src="https://img.shields.io/badge/PDF-Benchmarks-blue?style=for-the-badge&logo=adobe" alt="Benchmarks PDF"/>
</a>

*Performance comparison vs GPT-2, LSTM, and other models*

</td>
</tr>
</table>

---

## 🎯 What is PampaR?

> **"PampaR is an artificial brain where the thalamus orchestrates tokens toward specialized territories (Expressive, Contextual, Formal, Structural) that collaborate via bidirectional frontiers, combining explicit rules (LLAVES 70%) with learned attention (30%) to generate language."**

PampaR v9 reimagines neural language models through a **brain-inspired territorial architecture**. Instead of uniform transformer layers, it uses **4 specialized territories** connected by **6 bidirectional frontiers**, coordinated by a central **tálamo** (thalamus) that routes tokens using hybrid rule-based + learned attention.

---

## 🧠 Architecture v9 — Territorial

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

All territories connect to each other through learned bidirectional gates:
- Expresivo ↔ Contextual (0.8 strength)
- Expresivo ↔ Formal (0.5 strength)
- Expresivo ↔ Estructural (0.4 strength)
- Contextual ↔ Formal (0.6 strength)
- Contextual ↔ Estructural (0.5 strength)
- Formal ↔ Estructural (0.7 strength)

### Tálamo — The Orchestrator

The tálamo routes tokens using a **hybrid system**:
- **70% LLAVES** (explicit rules): Pattern matching for known token types
- **30% Learned attention**: Neural network for novel patterns

This provides **interpretability** (you can inspect which territory processes each token) while maintaining **flexibility** (the model learns to route unknown patterns).

---

## 📊 Performance

Trained on WikiText-103 with **14M parameters** on a **GTX 1650 4GB**:

| Metric | Value |
|--------|-------|
| **Parameters** | 14,069,410 |
| **Best Loss** | 3.81 |
| **Perplexity** | ~45.3 |
| **Training Tokens** | 250M+ |
| **Training Time** | ~70 hours |
| **Hardware** | GTX 1650 4GB VRAM |

### Comparison with Other Models (WikiText-103)

| Model | Parameters | Perplexity | Notes |
|-------|-----------|------------|-------|
| LSTM (Merity et al.) | 24M | 69.1 | AWD-LSTM, 2018 |
| Transformer-XL (Small) | 24M | 54.5 | Recurrent memory, 2019 |
| **PampaR v9** | **14M** | **~45*** | **Territorial arch., 2026** |
| GPT-2 Small | 125M | 35.1 | Standard Transformer, 2019 |

*Single training run. Comparison has limitations — see [Technical Paper](docs/PAPER_TECNICO.md#52-limitations).

**Preliminary observations**:
- ⚠️ PampaR shows promising efficiency with **42% fewer parameters** than comparable baselines
- ⚠️ Results require validation with multiple runs and additional datasets
- ✅ Trained entirely on **consumer hardware** (4GB VRAM)

---

## ⚠️ Limitations {#limitations}

This is experimental research with important caveats:

- **Single dataset**: Only evaluated on WikiText-103
- **Single run**: No confidence intervals or statistical analysis
- **Limited baselines**: Comparison with 2018-2019 models only
- **No ablations**: Individual component contributions not isolated
- **No downstream tasks**: Only perplexity evaluation, no GLUE/reasoning benchmarks
- **Interpretability claims**: Qualitative only, not formally validated

See the [full limitations section](docs/PAPER_TECNICO.md#52-limitations) in the technical paper.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/lucasmella-stack/PAMPAr-o1.git
cd PAMPAr-o1

# Install dependencies
pip install -r requirements.txt

# Download training data (WikiText-103)
python scripts/download_corpus.py
```

### Training

```bash
# Basic training
python scripts/train.py --tokens 10M --epochs 5

# Full training (50M tokens, ~70 hours on GTX 1650)
python scripts/train.py --tokens 50M --epochs 10 --batch-size 4 --accum 8

# Resume from checkpoint
python scripts/train.py --resume
```

### Inference

```python
import torch
from pampar.cerebro import PampaR
from pampar.config import LOCAL_4GB
import sentencepiece as sp

# Load tokenizer and model
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
        logits = out['logits']
        next_id = logits[0, -1].argmax().item()
        x = torch.cat([x, torch.tensor([[next_id]]).cuda()], dim=1)

print(tok.Decode(x[0].tolist()))
```

---

## 📊 Model Configurations

PampaR scales from 4GB to 80GB+ VRAM:

| Config | VRAM | Params | Dim | Layers | Heads |
|--------|------|--------|-----|--------|-------|
| LOCAL_4GB | 4GB | ~7M | 128 | 3 | 4 |
| **LOCAL_4GB_MAX** | **4GB** | **~14M** | **160** | **4** | **4** | ← *Used for training* |
| SERVER_8GB | 8GB | ~25M | 256 | 4 | 8 |
| SERVER_24GB | 24GB | ~100M | 512 | 6 | 8 |
| SERVER_80GB | 80GB | ~300M | 768 | 8 | 12 |

---

## 🏗️ Architecture v9

```
Input → Embedding → [BloqueTerrritorial ×N] → Axiomas → LM Head → Output
                              ↓
                    TálamoTerritorial
                     (LLAVES 70% + Atención 30%)
                              ↓
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
    ┌─────────────┐                      ┌─────────────┐
    │  EXPRESIVO  │◄──── Frontera ──────►│ CONTEXTUAL  │
    │ Lang+Creat  │                      │  Contexto   │
    └──────┬──────┘                      └──────┬──────┘
           │◄────── Fronteras Bidirec ─────────►│
    ┌──────▼──────┐                      ┌──────▼──────┐
    │   FORMAL    │◄──── Frontera ──────►│ESTRUCTURAL  │
    │   Lógica    │                      │ Patrón+Mat  │
    └─────────────┘                      └─────────────┘
```

---

## 📁 Project Structure

```
pampar/
├── __init__.py              # Main exports
├── config.py                # ConfigPampaR + presets
└── cerebro/
    ├── model.py             # Re-exports from model_v9.py
    ├── model_v9.py          # PampaR main class, BloqueTerrritorial
    ├── talamo.py            # TalamoTerritorial with LLAVES
    ├── territorio.py        # 4 Territories + GestorTerritorios
    ├── frontera.py          # 6 Bidirectional Frontiers
    ├── neurona.py           # Base neuron class
    ├── modulos/             # 6 specialized neurons
    │   └── especializados.py
    ├── razonamiento/        # Axiomas engine
    │   └── axiomas.py
    └── memoria/             # Experience memory

scripts/
├── train.py                 # Training script
├── chat.py                  # Interactive inference
├── test_v9.py               # Test v9 architecture
├── server.py                # API server
└── download_corpus.py       # Download WikiText-103

diagrams/
└── v9-territorial/
    ├── arquitectura_v9.txt
    ├── PampaR_v9_Arquitectura_Territorial.pdf
    └── PampaR_v9_Benchmarks_Comparacion.pdf
```

---

## 🔬 Innovations

### 1. Hybrid Routing (LLAVES + Attention)
Unlike pure neural routers (MoE), PampaR uses 70% explicit rules for known patterns + 30% learned routing for novel inputs. This provides interpretability while maintaining flexibility.

### 2. Territorial Processing
Instead of 18 pairwise synaptic connections (O(n²)), v9 uses 4 territories with 6 bidirectional frontiers. Modules within a territory share a buffer, reducing communication overhead.

### 3. Bidirectional Frontiers
Frontiers are NOT unidirectional. Information flows both ways with learned gates, mimicking biological inter-cortical connections.

### 4. Deductive Reasoning (Axiomas)
Built-in logical reasoning with modus ponens, syllogism, and other deductive rules. The model can explain its reasoning chain.

---

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 4GB+ VRAM (minimum)

---

## 📜 License

AGPL-3.0-or-later — See [LICENSE](LICENSE) for details.

**Note**: This is copyleft. If you modify and distribute PampaR, you must also release your source code under AGPL.

---

## 📚 Citation

```bibtex
@software{pampar_v9,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {PampaR: Cerebral Language Model with Territorial Architecture},
  year = {2026},
  version = {9.0.0},
  organization = {Independent Researcher},
  url = {https://github.com/lucasmella-stack/PAMPAr-o1},
  note = {14M parameters, PPL ~45 on WikiText-103, trained on GTX 1650 4GB}
}
```

---

## 👤 Author

- **Lucas Ricardo Mella Chillemi** — Architecture & Development

---

<div align="center">

**Made with ❤️ in Argentina 🇦🇷**

*"An artificial brain where territories collaborate through frontiers"*

</div>
