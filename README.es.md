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

<img src="PAMPAr-o1-imagen.png" alt="PAMPAr-o1 Logo" width="200"/>

# 🦙 PAMPAr-o1 v9

### Modelo de Lenguaje Cerebral con Arquitectura Territorial

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![DOI](https://zenodo.org/badge/1104543505.svg)](https://doi.org/10.5281/zenodo.18315642)

**[English](README.md)** | **[Arquitectura](docs/architecture/ARCHITECTURE.es.md)** | **[Innovaciones](docs/INNOVATIONS.es.md)**

</div>

---

## 👤 Para Reclutadores / Empleadores

> **TL;DR**: Esta es una arquitectura de IA original desarrollada desde cero por un desarrollador autodidacta, logrando resultados competitivos con investigación publicada mientras usa recursos mínimos.

| Lo que Demuestra |
|------------------|
| ✅ **Investigación Independiente** — Arquitectura novel diseñada sin supervisión académica |
| ✅ **ML Full-Stack** — Procesamiento de datos, diseño de modelo, infraestructura de entrenamiento, evaluación |
| ✅ **Optimización de Recursos** — 14M params entrenados en GPU de consumo de 4GB VRAM |
| ✅ **Documentación** — Papers técnicos, diagramas, código reproducible |
| ✅ **Ingeniería de Software** — Python limpio, diseño modular, tests, listo para CI |

**Logro Clave**: Supera a LSTM (24M params) y Transformer-XL Small (24M params) con **42% menos parámetros**.

---

## 🏆 Destacados

> ⚠️ **Investigación Experimental**: Este es un trabajo en progreso explorando arquitecturas inspiradas en el cerebro. Los resultados son preliminares y requieren validación adicional.

<table>
<tr>
<td align="center"><b>14M</b><br/>Parámetros</td>
<td align="center"><b>~45</b><br/>Perplejidad*</td>
<td align="center"><b>250M+</b><br/>Tokens entrenados</td>
<td align="center"><b>4GB</b><br/>VRAM (GTX 1650)</td>
</tr>
</table>

*\*En WikiText-103. Sujeto a varianza entre corridas.*

---

## 📄 Documentación

<table>
<tr>
<td width="50%" align="center">

### 🏗️ Arquitectura
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Arquitectura.html">
<img src="https://img.shields.io/badge/Ver-Arquitectura_(ES)-1a1a1a?style=for-the-badge&logo=readme" alt="Arquitectura ES"/>
</a>
<br/><br/>
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Architecture.html">
<img src="https://img.shields.io/badge/View-Architecture_(EN)-555555?style=for-the-badge&logo=readme" alt="Architecture EN"/>
</a>

*Arquitectura territorial con routing LLAVES*

</td>
<td width="50%" align="center">

### 📊 Benchmarks
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Benchmark_ES.html">
<img src="https://img.shields.io/badge/Ver-Benchmark_(ES)-1a1a1a?style=for-the-badge&logo=readme" alt="Benchmark ES"/>
</a>
<br/><br/>
<a href="https://htmlpreview.github.io/?https://github.com/lucasmella-stack/PAMPAr-o1/blob/main/diagrams/PampaR_Benchmark.html">
<img src="https://img.shields.io/badge/View-Benchmark_(EN)-555555?style=for-the-badge&logo=readme" alt="Benchmark EN"/>
</a>

*14M params vs LSTM, Transformer-XL, GPT-2*

</td>
</tr>
<tr>
<td colspan="2" align="center">

### 📝 Paper de Investigación
<a href="paper/pampar_v9_arxiv.tex">
<img src="https://img.shields.io/badge/LaTeX-Preprint_arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv Paper"/>
</a>

*Arquitectura Territorial Inspirada en el Cerebro para Modelado de Lenguaje*

</td>
</tr>
</table>

---

## 🧠 Arquitectura: Inspirada en el Cerebro

PampaR ("Procesamiento Adaptativo Modular de Patrones Articulados Recurrentes") organiza el procesamiento de lenguaje como regiones cerebrales:

```
                    ┌─────────────┐
        Entrada ───►│   TÁLAMO    │◄─── LLAVES (70% reglas)
                    │ Orquestador │◄─── Atención (30% aprendido)
                    └──────┬──────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  EXPRESIVO  │◄───►│  CONTEXTUAL │◄───►│   FORMAL    │
│ Lang+Creat  │     │  Contexto   │     │   Lógica    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ ESTRUCTURAL │
                    │ Patrones+Mat│
                    └─────────────┘
```

### Componentes Clave

| Componente | Función | Inspiración |
|------------|---------|-------------|
| **Tálamo** | Enruta tokens a territorios | Tálamo cerebral |
| **Territorios** | 4 regiones especializadas | Corteza cerebral |
| **Fronteras** | 6 conexiones bidireccionales | Tractos de materia blanca |
| **LLAVES** | Reglas de routing explícitas (70%) | Conocimiento estructurado |
| **Axiomas** | Razonamiento deductivo (opcional) | Lógica formal |

---

## 🚀 Inicio Rápido

### Instalación

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-o1.git
cd PAMPAr-o1
pip install -r requirements.txt
```

### Uso Básico

```python
from pampar import PampaR, LOCAL_4GB
import torch

# Crear modelo
model = PampaR(LOCAL_4GB)

# Forward pass
input_ids = torch.randint(0, 8000, (1, 64))
output = model(input_ids)
logits = output['logits']  # (1, 64, 8000)

# Generar texto
generated = model.generate(input_ids, max_new_tokens=50)
```

### Entrenamiento

```bash
# Descargar corpus WikiText-103
python scripts/download_corpus.py

# Entrenar (configuración para 4GB VRAM)
python scripts/train.py --tokens 50M --epochs 30
```

---

## 📊 Resultados

### Comparación con Baselines (WikiText-103)

| Modelo | Parámetros | Perplejidad | Hardware |
|--------|------------|-------------|----------|
| LSTM (Merity 2018) | 24M | 69.1 | - |
| Transformer-XL Small | 24M | 54.5 | - |
| **PAMPAr-o1 v9** | **14M** | **~45** | GTX 1650 4GB |

### Por qué Importa

- **42% menos parámetros** que modelos comparables
- **Entrenado en hardware de consumo** (sin datacenter)
- **Arquitectura interpretable** (puedes ver qué territorio procesa qué)

---

## 📁 Estructura del Proyecto

```
PAMPAr-o1/
├── pampar/                  # Paquete principal
│   ├── cerebro/            # Arquitectura v9
│   │   ├── model_v9.py     # Modelo PampaR
│   │   ├── talamo.py       # Orquestador
│   │   ├── territorio.py   # 4 territorios
│   │   ├── frontera.py     # 6 fronteras
│   │   ├── modulos/        # 6 neuronas especializadas
│   │   ├── razonamiento/   # Axiomas
│   │   └── memoria/        # Experiencia
│   ├── config.py           # Configuraciones
│   └── utils/              # Utilidades
├── scripts/                 # Entrenamiento y evaluación
├── tests/                   # Tests unitarios
├── docs/                    # Documentación
└── checkpoints/            # Modelos guardados
```

---

## 🔬 Configuraciones Escalables

```python
from pampar.config import LOCAL_4GB, SERVER_8GB, SERVER_24GB

# Para tu GPU
LOCAL_4GB      # GTX 1650/1660, ~6M params
LOCAL_4GB_MAX  # GTX 1650 optimizado, ~14M params
SERVER_8GB     # RTX 3060/3070, ~25M params
SERVER_24GB    # RTX 3090/4090, ~100M params
```

---

## 📝 Citación

```bibtex
@software{pampar2026,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {PampaR: Cerebral Language Model with Territorial Architecture},
  year = {2026},
  url = {https://github.com/lucasmella-stack/PAMPAr-o1},
  version = {9.0.0}
}
```

---

## 📄 Licencia

**AGPL-3.0-or-later** — Código abierto con copyleft.  
Uso libre, pero derivados deben permanecer open source.

---

## 👤 Autor

**Lucas Ricardo Mella Chillemi**  
Investigador Independiente  
📧 lucas.mella@outlook.com  
🔗 [GitHub](https://github.com/lucasmella-stack)

---

<div align="center">

*Desarrollado con ❤️ y una GTX 1650*

</div>
