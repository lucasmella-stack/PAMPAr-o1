# PampaR: A Brain-Inspired Territorial Architecture for Language Modeling

## Technical Report — Work in Progress

**Lucas Ricardo Mella Chillemi**  
Independent Researcher  
January 2026

> ⚠️ **Note**: This is a preliminary technical report describing an experimental architecture under active development. Results are promising but require further validation across multiple datasets and configurations.

---

## Abstract

I present PampaR, an **experimental** language model architecture inspired by the functional organization of the human brain. Unlike standard transformer architectures that treat all computations uniformly, PampaR explores **Territorial Processing** where specialized neural modules are organized into functional territories (Expressive, Contextual, Formal, Structural) coordinated by a central **Thalamus** (Tálamo) that routes tokens using a hybrid approach: 70% explicit rules (LLAVES) and 30% learned attention. 

My **preliminary experiments** on WikiText-103 suggest that this architecture can achieve **perplexity of approximately 45** with only **14M parameters**, comparing favorably to LSTM (69.1 PPL) and Transformer-XL Small (54.5 PPL) baselines that use 24M parameters. The model was trained entirely on consumer hardware (GTX 1650 4GB VRAM), demonstrating potential efficiency gains of the territorial approach.

**Important caveats**: These results are from a single training run on one dataset. Further validation with multiple runs, ablation studies, and evaluation on diverse benchmarks is ongoing. I release this report to document my approach and invite community feedback.

---

## 1. Introduction

Current large language models (LLMs) like GPT, LLaMA, and Claude achieve remarkable performance but suffer from several limitations:

1. **Opacity**: Attention patterns are difficult to interpret
2. **Homogeneity**: All layers perform identical operations
3. **Inefficiency**: All tokens are processed by all parameters

The human brain, in contrast, exhibits clear **functional specialization**:
- Broca's area for language production
- Wernicke's area for language comprehension  
- Prefrontal cortex for logical reasoning
- Hippocampus for memory and context

PampaR (Pampa Reasoning) draws inspiration from this organization, exploring a **territorial architecture** where:
- Specialized **modules** (neurons) handle different aspects of language
- A central **Thalamus** routes information between modules
- **Bidirectional frontiers** allow inter-territory communication
- **Explicit rules** (LLAVES/Keys) provide interpretable routing

**This work is exploratory**: I aim to investigate whether brain-inspired functional specialization can improve efficiency and interpretability in small-scale language models. My current implementation represents one possible instantiation of these ideas.

---

## 2. Architecture

### 2.1 Overview

```
Input → Embedding → [Territorial Block ×N] → LM Head → Output
                          ↓
              Thalamus (LLAVES 70% + Attention 30%)
                          ↓
    ┌─────────────────────┴─────────────────────┐
    │                                           │
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│   EXPRESSIVE  │◄───── Frontier ───►│  CONTEXTUAL   │
│ Language+Crea │                    │   Context     │
└───────┬───────┘                    └───────┬───────┘
        │                                    │
        │◄────── Bidirectional ─────────────►│
        │                                    │
┌───────▼───────┐                    ┌───────▼───────┐
│    FORMAL     │◄───── Frontier ───►│  STRUCTURAL   │
│    Logic      │                    │ Pattern+Math  │
└───────────────┘                    └───────────────┘
```

### 2.2 Territories

PampaR organizes computation into **4 functional territories**, each containing specialized modules:

| Territory | Modules | Function | Analogous Brain Region |
|-----------|---------|----------|----------------------|
| **Expressive** | Language, Creativity | Fluent text generation, novel ideas | Broca's area, Right hemisphere |
| **Contextual** | Context | Working memory, coherence | Hippocampus, PFC |
| **Formal** | Logic | Deductive reasoning, rules | Prefrontal cortex |
| **Structural** | Patterns, Mathematics | Sequences, numbers, structure | Parietal lobe |

### 2.3 The Thalamus (Tálamo)

The Thalamus is the central orchestrator that routes tokens to appropriate territories. It implements a **hybrid routing mechanism**:

```python
routing_weights = peso_llaves * llaves_activation + (1 - peso_llaves) * learned_attention
```

Where:
- `peso_llaves = 0.7` (70% explicit rules)
- `llaves_activation` = rule-based routing from LLAVES system
- `learned_attention` = learned routing via attention mechanism

#### 2.3.1 LLAVES (Keys) System

LLAVES are explicit, interpretable routing rules based on token patterns:

| Module | Pattern Examples |
|--------|-----------------|
| Language | "the", "is", "of", articles, prepositions |
| Mathematics | digits 0-9, "+", "-", "=", "%" |
| Logic | "if", "then", "because", "therefore" |
| Patterns | repetitions, sequences, structures |
| Context | pronouns, references, temporal markers |
| Creativity | adjectives, metaphors, novel combinations |

This provides **interpretability**: one can directly inspect why a token was routed to a specific territory.

### 2.4 Bidirectional Frontiers

Territories communicate via **6 bidirectional frontier connections** with learned gates:

```python
FRONTIERS = [
    ("expressive", "contextual", 0.8),   # High: narrative needs context
    ("expressive", "formal", 0.5),       # Medium: argumentation
    ("expressive", "structural", 0.4),   # Lower: creative structure
    ("contextual", "formal", 0.6),       # Medium-high: logical context
    ("contextual", "structural", 0.5),   # Medium: pattern context
    ("formal", "structural", 0.7),       # High: mathematical logic
]
```

Each frontier implements:
```python
output = gate * transform(territory_a) + (1-gate) * transform(territory_b)
```

### 2.5 Optional Components

#### Axioms Engine (Motor de Axiomas)
Implements deductive reasoning patterns:
- **Modus Ponens**: If P→Q and P, then Q
- **Syllogism**: If A→B and B→C, then A→C
- **Negation**: Logical NOT operations

#### Practical Memory
Stores successful/failed patterns for rapid retrieval during inference.

---

## 3. Implementation Details

### 3.1 Model Configuration

```python
ConfigPampaR(
    vocab_size=8000,        # BPE tokenizer
    dim=160,                # Hidden dimension
    n_heads=4,              # Attention heads per module
    n_capas=4,              # Layers per module
    dropout=0.1,
    max_seq_len=256,
    peso_llaves=0.7,        # 70% rules, 30% learned
    usar_axiomas=True,      # Enable axiom engine
    usar_memoria=True,      # Enable practical memory
)
```

**Total Parameters**: 14,069,410 (~14M)

### 3.2 Training Setup

- **Dataset**: WikiText-103 (100M tokens)
- **Hardware**: NVIDIA GTX 1650 (4GB VRAM)
- **Batch Size**: 4 (effective 32 with gradient accumulation)
- **Sequence Length**: 128 tokens
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.01)
- **Precision**: Mixed FP16

### 3.3 Fragmented Training

Due to hardware constraints, I implement **fragmented training**:

| Fragment | Tokens | Epochs | Cumulative |
|----------|--------|--------|------------|
| 1 | 10M | 3 | 30M |
| 2 | 20M | 3 | 90M |
| 3 | 35M | 3 | 195M |
| 4 | 50M | 3 | 345M |
| 5 | 75M | 2 | 495M |
| 6 | 100M | 2 | 695M |

---

## 4. Results

### 4.1 Training Progress

| Fragment | Final Loss | Final PPL | Tokens Seen | Improvement |
|----------|------------|-----------|-------------|-------------|
| 1 (10M) | 4.85 | 127.5 | 30M | Baseline |
| 2 (20M) | 4.22 | 68.1 | 90M | -46.6% PPL |
| 3 (35M) | 3.93 | 50.7 | 195M | -60.2% PPL |
| **4 (50M)** | **3.81** | **~45** | **250M+** | **-64.7% PPL** |

**Total training time**: ~70 hours on GTX 1650 4GB VRAM.

### 4.2 Comparison with Baselines

| Model | Parameters | PPL (WikiText-103) | Year |
|-------|------------|-------------------|------|
| LSTM (Merity et al.) | 24M | 69.1 | 2018 |
| Transformer-XL Small | 24M | 54.5 | 2019 |
| **PampaR v9** | **14M** | **~45** | **2026** |
| GPT-2 Small | 125M | 35.1 | 2019 |

**Preliminary finding**: My results suggest PampaR may achieve **comparable or better perplexity than Transformer-XL Small** with **42% fewer parameters**, trained entirely on consumer hardware.

> ⚠️ **Caveat**: This comparison has limitations. My evaluation uses a single training run without confidence intervals. The baselines are from published papers with potentially different preprocessing. A rigorous comparison would require multiple runs and standardized evaluation protocols.

### 4.3 Interpretability Analysis (Qualitative)

Unlike black-box transformers, PampaR provides insight into token routing:

```
Token: "mathematics" → LLAVES activation:
  - Mathematics module: 0.85 (high)
  - Language module: 0.15 (low)
  
Token: "therefore" → LLAVES activation:
  - Logic module: 0.90 (high)
  - Context module: 0.10 (low)
```

---

## 5. Discussion

### 5.1 Advantages

1. **Interpretability**: LLAVES provide explicit routing rationale
2. **Parameter Efficiency**: Competitive PPL with fewer parameters
3. **Modularity**: Easy to add/modify specialized territories
4. **Biological Plausibility**: Mirrors brain functional organization

### 5.2 Limitations

I acknowledge several significant limitations of this work:

#### Experimental Design
1. **Single Dataset**: Results reported only on WikiText-103; generalization to other domains (code, multilingual, scientific text) is untested
2. **Single Run**: No statistical analysis with multiple training runs; reported metrics lack confidence intervals
3. **Limited Baselines**: Comparison with older models (2018-2019); newer efficient architectures not evaluated

#### Architecture
4. **LLAVES Design**: Routing rules are hand-crafted based on intuition; optimal rules unknown
5. **Hyperparameter Sensitivity**: The 70/30 split (rules vs learned) was not systematically ablated
6. **Territory Count**: Why 4 territories? This design choice was not empirically validated
7. **Scale Unknown**: Architecture behavior at 100M+ parameters is unexplored

#### Evaluation
8. **Perplexity Only**: No evaluation on downstream tasks (classification, QA, reasoning benchmarks)
9. **No Ablations**: Contribution of individual components (LLAVES, frontiers, axioms) not isolated
10. **Interpretability Claims**: Qualitative only; no user studies or formal interpretability metrics

#### Reproducibility
11. **Hardware Specific**: Trained on specific GPU; reproducibility on other hardware not verified
12. **Tokenizer**: Custom BPE tokenizer; comparison with standard tokenizers not done

### 5.3 Future Work

I plan to address the limitations above through:

#### Short-term (Ongoing)
1. **Multi-dataset Validation**: Evaluate on Penn Treebank, C4, The Pile
2. **Ablation Studies**: Systematically vary LLAVES weight (50%, 70%, 90%), territory count (2, 4, 6), and frontier configurations
3. **Multiple Runs**: Report mean ± std across 3-5 training runs
4. **Modern Baselines**: Compare with MobileLLM, TinyLlama, and other efficient architectures

#### Medium-term
5. **Downstream Tasks**: Evaluate on GLUE, SuperGLUE, reasoning benchmarks
6. **Scaling Laws**: Test at 50M, 100M, 500M parameter scales
7. **Learned LLAVES**: Train routing rules end-to-end instead of hand-crafting
8. **Formal Interpretability**: Conduct user studies comparing LLAVES explanations vs attention visualization

#### Long-term (Speculative)
9. **Neuro-alignment**: Compare model activations with brain imaging data (fMRI, EEG)
10. **Multimodal Territories**: Extend architecture to vision-language models
11. **Mixture of Experts Integration**: Combine territorial routing with sparse MoE

---

## 6. Conclusion

I have presented PampaR, an **experimental** brain-inspired architecture that explores territorial processing for language modeling. My preliminary results suggest that the combination of explicit routing rules (LLAVES) with learned attention may offer a promising direction for building efficient and more interpretable language models.

**What I claim**:
- The territorial architecture is a viable alternative to uniform transformer layers
- Hybrid rule-based + learned routing is implementable and trainable
- Preliminary results on WikiText-103 are encouraging (PPL ~45 with 14M params)
- The approach may offer interpretability advantages worth investigating

**What I do NOT claim**:
- That PampaR is definitively better than existing architectures
- That my results generalize beyond WikiText-103
- That the specific design choices (4 territories, 70% LLAVES) are optimal
- That the interpretability benefits have been rigorously validated

This work represents an early exploration of brain-inspired language modeling. I release all code under AGPL-3.0 license to enable community validation, criticism, and collaboration. I welcome feedback and contributions.

**Acknowledgments**: This work was conducted independently without institutional support or cloud compute resources. The entire model was trained on a consumer GPU (GTX 1650 4GB), demonstrating that meaningful ML research remains accessible to independent researchers.

---

## References

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Merity, S., et al. (2018). Regularizing and Optimizing LSTM Language Models.
3. Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla).
5. Fedorenko, E., & Thompson-Schill, S. L. (2014). Reworking the language network. Trends in Cognitive Sciences.

---

## Appendix A: Reproducibility

### A.1 Code Repository
- GitHub: `https://github.com/lucasmella-stack/PAMPAr-o1`
- License: AGPL-3.0-or-later

### A.2 Training Commands

```bash
# Fragment 1
python scripts/train_fragmentado.py --fragmento 1

# Resume training
python scripts/train_fragmentado.py --fragmento 2

# Full training
python scripts/train_fragmentado.py --max
```

### A.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4GB | 8GB |
| RAM | 16GB | 32GB |
| Storage | 5GB | 20GB |

---

## Appendix B: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TOKENS                            │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOKEN EMBEDDINGS                             │
│                  (vocab_size × dim)                             │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POSITIONAL ENCODING                           │
│                  (max_seq_len × dim)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
          ╔═══════════════════════════════════════════╗
          ║         TERRITORIAL BLOCK (×N)            ║
          ╠═══════════════════════════════════════════╣
          ║                                           ║
          ║  ┌─────────────────────────────────────┐  ║
          ║  │            TÁLAMO                   │  ║
          ║  │   ┌─────────────┬────────────────┐  │  ║
          ║  │   │   LLAVES    │   ATTENTION    │  │  ║
          ║  │   │    (70%)    │     (30%)      │  │  ║
          ║  │   └─────────────┴────────────────┘  │  ║
          ║  └──────────────────┬──────────────────┘  ║
          ║                     │                     ║
          ║     ┌───────────────┼───────────────┐     ║
          ║     ▼               ▼               ▼     ║
          ║  ┌──────┐       ┌──────┐       ┌──────┐  ║
          ║  │EXPRES│◄─────►│CONTEX│◄─────►│FORMAL│  ║
          ║  │ IVO  │       │ TUAL │       │      │  ║
          ║  └──┬───┘       └──┬───┘       └──┬───┘  ║
          ║     │              │              │      ║
          ║     └──────────────┼──────────────┘      ║
          ║                    ▼                     ║
          ║              ┌──────────┐                ║
          ║              │ESTRUCTUR │                ║
          ║              │   AL     │                ║
          ║              └──────────┘                ║
          ║                    │                     ║
          ║  ┌─────────────────┴─────────────────┐   ║
          ║  │        FRONTIER FUSION            │   ║
          ║  └─────────────────┬─────────────────┘   ║
          ║                    │                     ║
          ╚════════════════════╪═════════════════════╝
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LM HEAD                                    │
│                   (dim → vocab_size)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LOGITS                                │
└─────────────────────────────────────────────────────────────────┘
```

---

**License**: AGPL-3.0-or-later  
**Copyright**: © 2024-2026 Lucas Ricardo Mella Chillemi
