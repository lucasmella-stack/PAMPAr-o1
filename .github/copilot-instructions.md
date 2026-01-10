# PampaR Copilot Instructions

## Project Overview
PampaR is a **cerebral language model** with brain-inspired modular architecture. The model uses 6 specialized "neurons" coordinated by a central "thalamus" (Tálamo) that routes tokens to appropriate modules using explicit rules (LLAVES).

## Architecture (Big Picture)

```
Input → Embedding → [CerebralBlock ×N] → LM Head → Output
                          ↓
              Tálamo (LLAVES routing)
                          ↓
    ┌─────────────────────┼─────────────────────┐
    ↓         ↓         ↓         ↓         ↓         ↓
Lenguaje  Lógica  Matemat  Patron  Context  Creat
    └─────────────────────┼─────────────────────┘
                    Sinapsis (inter-module)
                          ↓
                       Axiomas (deductive reasoning)
```

### Key Components in `pampar/cerebro/`
- **model.py** → `PampaR` main class, `CerebralBlock` processing blocks
- **talamo.py** → `Talamo` orchestrator with `LlaveModulo` routing rules
- **sinapsis.py** → `Sinapsis` inter-module connections with `CONEXIONES_NATURALES` matrix
- **modulos/especializados.py** → 6 specialized neurons (`NeuronaLenguaje`, `NeuronaLogica`, etc.)
- **razonamiento/axiomas.py** → `MotorAxiomas` with modus ponens, silogismo, etc.

### Configuration System
- **config.py** → `ConfigPampaR` for language model (v8), `Config` for legacy MNIST (v4)
- Presets: `LOCAL_4GB`, `SERVER_8GB`, `SERVER_24GB`, `SERVER_80GB` for different VRAM targets

## Developer Workflows

### Training
```bash
python scripts/train.py                     # Basic training
python scripts/train.py --tokens 10M        # Limit tokens (10M, 50M)
python scripts/train.py --resume            # Resume from checkpoint
python scripts/train.py --batch-size 32 --lr 1e-4 --epochs 10
```

### Inference/Chat
```bash
python scripts/chat.py
python scripts/chat.py --checkpoint checkpoints/pampar_best.pt
python scripts/chat.py --temperature 0.7 --top_p 0.95
```

### Testing
```bash
pytest tests/                  # Run all tests
pytest tests/test_model.py -q  # Specific test file
```

### Data Requirements
- Tokenizer: `data/tokenizer/llarri_bpe.model` (SentencePiece BPE)
- Corpus: `data/wikitext-103/wikitext-103-raw/wiki.train.raw`
- Run `python scripts/download_corpus.py` to fetch WikiText-103

## Code Conventions

### File Headers
All Python files use SPDX license headers:
```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
```

### Naming Conventions
- Spanish names for domain concepts: `Tálamo`, `Sinapsis`, `Neurona`, `Axiomas`, `LLAVES`
- English for standard ML terms: `forward`, `batch`, `embedding`
- Configuration classes: `ConfigPampaR`, `Config`
- Module neurons: `NeuronaLenguaje`, `NeuronaLogica`, etc.

### Module Pattern
Each specialized neuron in `modulos/especializados.py`:
1. Inherits from `Neurona` base class
2. Defines `LLAVES` dict with domain-specific token patterns
3. Implements `es_mi_dominio()` for relevance scoring
4. Implements `procesar()` with specialized attention + domain-specific layers

### Tálamo LLAVES System
The thalamus uses 70% rule-based (`peso_llaves=0.7`) + 30% learned attention:
```python
# In talamo.py - define token patterns per module
self.llaves = {
    'lenguaje': LlaveModulo(patrones=['el', 'la', 'de', 'que', ...]),
    'matematicas': LlaveModulo(patrones=['0-9', '+', '-', '=', ...]),
    # ...
}
```

### Sinapsis Connections
Inter-module connections defined in `CONEXIONES_NATURALES` dict:
```python
('lenguaje', 'contexto'): (TipoSinapsis.EXCITATORIA, 0.8),
('logica', 'matematicas'): (TipoSinapsis.EXCITATORIA, 0.7),
```

## Key Implementation Details

### Checkpointing
- Checkpoints save: `{'model': state_dict, 'config': dict, 'epoch': int, 'val_loss': float}`
- Resume with: `torch.load(path, map_location=device, weights_only=False)`

### Mixed Precision
- Enabled via `config.use_mixed_precision = True`
- Uses `torch.amp.autocast` and `GradScaler`

### Gradient Checkpointing
- Enabled via `config.use_gradient_checkpointing = True`
- Applied in `CerebralBlock.forward()` using `torch.utils.checkpoint.checkpoint`

### Tokenizer Registration
```python
model.registrar_tokenizer(tokenizer)  # Populates LLAVES token mappings
```

## Important Notes
- **Bilingual docs**: English and Spanish versions exist (`*.es.md`)
- **Legacy code**: `versions/legacy/` contains older v4 MNIST implementation
- **Diagrams**: Architecture diagrams in `diagrams/v8-current/`
- **License**: AGPL-3.0-or-later (copyleft, disclose source for derivatives)
