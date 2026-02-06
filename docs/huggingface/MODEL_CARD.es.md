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
---

# 🐰 PAMPAr-o1 v9 — Modelo de Lenguaje Cerebral

<div align="center">

*"Un cerebro artificial donde los territorios colaboran a través de fronteras"*

[![Licencia](https://img.shields.io/badge/Licencia-AGPL--3.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[English](MODEL_CARD.md)** | **[GitHub](https://github.com/lucasmella-stack/PAMPAr-o1)** | **[Documentación](https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/docs/architecture/ARCHITECTURE.md)**

</div>

---

## 📄 Documentación en PDF

<table>
<tr>
<td width="50%" align="center">

### 🏗️ Arquitectura
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Arquitectura.html">
<img src="https://img.shields.io/badge/Ver-Arquitectura-1a1a1a?style=for-the-badge&logo=readme" alt="Arquitectura"/>
</a>

</td>
<td width="50%" align="center">

### 📊 Benchmarks
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Benchmark_ES.html">
<img src="https://img.shields.io/badge/Ver-Benchmarks-1a1a1a?style=for-the-badge&logo=readme" alt="Benchmarks"/>
</a>

</td>
</tr>
</table>

---

## ¿Qué es PAMPAr-o1 v9?

> **"PampaR es un cerebro artificial donde el tálamo orquesta tokens hacia territorios especializados (Expresivo, Contextual, Formal, Estructural) que colaboran vía fronteras bidireccionales, combinando reglas explícitas (LLAVES 70%) con atención aprendida (30%) para generar lenguaje."**

PAMPAr-o1 v9 es un **modelo de lenguaje inspirado en el cerebro** con una arquitectura **territorial** única:

- **4 Territorios Especializados** (agrupaciones de módulos neuronales relacionados)
- **6 Fronteras Bidireccionales** (conexiones aprendidas entre territorios)
- **1 Tálamo** (orquestador con routing híbrido reglas + neural)

---

## Arquitectura: Procesamiento Territorial

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
                       Axiomas (razonamiento)
```

### 4 Territorios

| Territorio | Módulos | Función |
|------------|---------|---------|
| **Expresivo** | Lenguaje + Creatividad | Generación fluida, ideas nuevas |
| **Contextual** | Contexto | Memoria de trabajo, coherencia |
| **Formal** | Lógica | Razonamiento lógico, reglas |
| **Estructural** | Patrones + Matemáticas | Secuencias, números, patrones |

### 6 Fronteras Bidireccionales

| Conexión | Fuerza | Función |
|----------|--------|---------|
| Expresivo ↔ Contextual | 0.8 | Coherencia narrativa |
| Expresivo ↔ Formal | 0.5 | Argumentación |
| Expresivo ↔ Estructural | 0.4 | Creatividad estructurada |
| Contextual ↔ Formal | 0.6 | Contexto lógico |
| Contextual ↔ Estructural | 0.5 | Memoria de patrones |
| Formal ↔ Estructural | 0.7 | Lógica matemática |

---

## Innovación Clave: Routing Híbrido (LLAVES)

El **Tálamo** rutea tokens usando:

- **LLAVES (70%)**: Reglas explícitas para patrones conocidos
  - "el", "la", "the" → Territorio de Lenguaje
  - "0-9", "+", "=" → Territorio de Matemáticas
  
- **Atención Aprendida (30%)**: Red neuronal para patrones nuevos

**¿Por qué híbrido?** Proporciona **interpretabilidad** — podés ver exactamente por qué un token fue ruteado a un territorio.

---

## Rendimiento

| Propiedad | Valor |
|-----------|-------|
| **Parámetros** | 6,834,586 |
| **Loss de Validación** | 4.05 |
| **Perplejidad** | 57.1 |
| **Datos de Entrenamiento** | WikiText-103 |
| **Uso de VRAM** | 110 MB |

### Comparación

| Modelo | Parámetros | Perplejidad | Eficiencia |
|--------|-----------|-------------|------------|
| LSTM baseline | ~10M | 100-120 | 0.1M/PPL |
| **PAMPAr-o1 v9** | **6.8M** | **57.1** | **0.12M/PPL** ✨ |
| Transformer (small) | 44M | 65 | 0.68M/PPL |
| GPT-2 Small | 124M | 29-35 | 4M/PPL |

---

## Inicio Rápido

```python
import torch
from pampar.cerebro import PampaR
from pampar.config import LOCAL_4GB
import sentencepiece as sp

# Cargar
tok = sp.SentencePieceProcessor()
tok.Load('data/tokenizer/llarri_bpe.model')

model = PampaR(LOCAL_4GB).cuda()
ckpt = torch.load('checkpoints/pampar_best.pt', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Generar
prompt = "La historia de"
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

## Configuraciones del Modelo

| Config | VRAM | Params | Dim | Capas |
|--------|------|--------|-----|-------|
| LOCAL_4GB | 4GB | ~6.8M | 128 | 3 |
| SERVER_8GB | 8GB | ~25M | 256 | 4 |
| SERVER_24GB | 24GB | ~100M | 512 | 6 |
| SERVER_80GB | 80GB | ~300M | 768 | 8 |

---

## Uso Previsto

- ✅ Investigación en arquitecturas neuronales inspiradas en el cerebro
- ✅ Exploración de mecanismos de routing interpretables
- ✅ Modelado de lenguaje eficiente en hardware limitado
- ⚠️ No listo para producción en aplicaciones comerciales

---

## Cita

```bibtex
@software{pampar_v9,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {PampaR: Modelo de Lenguaje Cerebral con Arquitectura Territorial},
  year = {2026},
  version = {9.0.0},
  organization = {Investigador Independiente},
  url = {https://github.com/lucasmella-stack/PAMPAr-o1}
}
```

---

## Enlaces

- 📂 **GitHub:** [lucasmella-stack/PAMPAr-o1](https://github.com/lucasmella-stack/PAMPAr-o1)
- 📄 **PDF Arquitectura:** [Ver](https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Arquitectura_Territorial.pdf)
- 📊 **PDF Benchmarks:** [Ver](https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/v9-territorial/PampaR_v9_Benchmarks_Comparacion.pdf)

---

## Contacto

- **Lucas Ricardo Mella Chillemi** — lucas.mella@outlook.com
- **GitHub:** [lucasmella-stack/PAMPAr-o1](https://github.com/lucasmella-stack/PAMPAr-o1)

---

<div align="center">

**Hecho con ❤️ en Argentina 🇦🇷**

</div>
