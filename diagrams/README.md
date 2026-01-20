# PampaR v9 - Architecture Diagrams

## Quick Overview

```
Input → Embedding → [Territorial Block ×N] → LM Head → Output
                            ↓
                  Tálamo (LLAVES 70% + Attention 30%)
                            ↓
         ┌─────────────────────────────────┐
         │                                 │
    ┌────▼────┐                     ┌──────▼──────┐
    │EXPRESIVO│◄───── Frontier ────►│ CONTEXTUAL  │
    │Lang+Crea│                     │   Context   │
    └────┬────┘                     └──────┬──────┘
         │                                 │
         │◄────── Bidirectional ──────────►│
         │          Frontiers              │
    ┌────▼────┐                     ┌──────▼──────┐
    │  FORMAL │◄───── Frontier ────►│ESTRUCTURAL  │
    │  Logic  │                     │Pattern+Math │
    └─────────┘                     └─────────────┘
```

## Files

| File | Description |
|------|-------------|
| `arquitectura_v9.txt` | Detailed ASCII diagrams |
| `PampaR_v9_Arquitectura_Territorial.pdf` | Visual architecture |
| `PampaR_v9_Benchmarks_Comparacion.pdf` | Benchmark comparisons |
| `PampaR v9 - Resultados de Benchmark - 14M.pdf` | 14M results |

## Key Numbers

| Metric | Value |
|--------|-------|
| Parameters | 14M |
| Perplexity | ~45 |
| Territories | 4 |
| Frontiers | 6 bidirectional |
| LLAVES weight | 70% |
| Attention weight | 30% |

## Territories

| Territory | Modules | Function |
|-----------|---------|----------|
| **Expresivo** | Lenguaje + Creatividad | Fluent text, novel ideas |
| **Contextual** | Contexto | Working memory, coherence |
| **Formal** | Lógica | Rules, reasoning |
| **Estructural** | Patrones + Matemáticas | Sequences, numbers |

## Routing (LLAVES)

```
Token    → LLAVES Classification → Territory Weight
"el"     → [lenguaje: 1.0]       → Expresivo (0.8)
"2+2"    → [matemat: 1.5]        → Estructural (0.9)
"si"     → [lógica: 1.2]         → Formal (0.7)
```

---

*Legacy diagrams (v2-v8) archived in `versions/legacy/diagrams/`*
