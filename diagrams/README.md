# PAMPAr-o1 v9 - Documentation

Technical documentation for the PAMPAr-o1 v9 territorial language model.

## Documents

### Architecture
| Language | File | Description |
|----------|------|-------------|
| English | [PampaR_Architecture.html](PampaR_Architecture.html) | Complete architecture documentation |
| Español | [PampaR_Arquitectura.html](PampaR_Arquitectura.html) | Documentación completa de arquitectura |

### Benchmark Results
| Language | File | Description |
|----------|------|-------------|
| English | [PampaR_Benchmark.html](PampaR_Benchmark.html) | Training progress & comparisons |
| Español | [PampaR_Benchmark_ES.html](PampaR_Benchmark_ES.html) | Progreso de entrenamiento y comparaciones |

### ASCII Diagrams
- [arquitectura_v9.txt](arquitectura_v9.txt) - Text-based architecture diagrams

## How to Generate PDFs

1. Open the `.html` file in your browser
2. Press `Ctrl+P` (or `Cmd+P` on Mac)
3. Select "Save as PDF"
4. Choose A4 paper size
5. Enable "Background graphics" for best results

## Quick Reference

```
Parameters: 14M
Perplexity: ~45
Dataset: WikiText-103
Hardware: GTX 1650 (4GB)
```

## Architecture

```
Input → Embedding → [Territorial Block ×6] → LM Head → Output
                           │
                    ┌──────┴──────┐
                    │   TÁLAMO    │
                    │ LLAVES 70%  │
                    │ + Attn 30%  │
                    └──────┬──────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    ▼          ▼           ▼           ▼          ▼
EXPRESIVO  CONTEXTUAL   FORMAL   ESTRUCTURAL
```

---

*Copyright © 2024-2026 Lucas Ricardo Mella Chillemi*  
*License: AGPL-3.0-or-later*
