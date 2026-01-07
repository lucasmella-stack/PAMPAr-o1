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
- adaptive-compute
- hierarchical-tokenization
pipeline_tag: text-generation
---

# 🧠 LLARRI-O1 — Modelo de Lenguaje Fractal

<div align="center">

*"Mezcla primero, procesa con vecinos — de pequeño a grande"*

[![Licencia](https://img.shields.io/badge/Licencia-AGPL--3.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[English](MODEL_CARD.md)** | **[GitHub](https://github.com/lucasmella-stack/llarri-o1)** | **[Documentación Completa](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.es.md)**

</div>

---

## ¿Qué es LLARRI-O1?

LLARRI-O1 es un **modelo de lenguaje experimental** que reimagina el procesamiento de redes neuronales a través de una **arquitectura de 6 cajas** inspirada en jerarquías de caché de CPU (L1/L2/L3).

En lugar del patrón tradicional Transformer (atención → FFN completa → repetir), LLARRI:

1. 🔀 **Mezcla globalmente** (atención para ver qué es relevante)
2. 📍 **Procesa cercanos primero** (cómputo pequeño/barato)
3. 📈 **Expande solo si es necesario** (cómputo progresivo)
4. 🚀 **Sale temprano cuando está seguro** (ahorra recursos)

---

## Tradicional vs LLARRI

```
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER TRADICIONAL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [Atención] ──► [FFN 4x] ──► Output             │
│                                                             │
│   • Cómputo fijo por capa                                   │
│   • Sin early exit                                          │
│   • Mismo costo para TODOS los tokens                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              LLARRI-O1 (6 CAJAS)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [MEZCLA] ──► [PROCESA] ──► [EVALÚA] ──► Output │
│                          0.5x→1.0x      ¿salida?            │
│                                                             │
│   Caja 1: MEZCLA (Atención) ──────────────────────────────►│
│   Caja 2: PROCESA cercano (0.5x FFN) ──► ¿SALIR? ─────────►│
│   Caja 3: PROCESA medio (0.75x FFN) ──► ¿SALIR? ──────────►│
│   Caja 4: PROCESA lejos (1.0x FFN) ───────────────────────►│
│   Caja 6: OUTPUT ◄────────────────────────────────────────►│
│                                                             │
│   • Cómputo adaptativo (inputs fáciles salen temprano)      │
│   • Costo progresivo (0.5x → 0.75x → 1.0x)                  │
│   • Como caché de CPU: L1 (rápido) → L2 → L3 (lento)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8 Innovaciones Clave

| Acrónimo | Nombre | Qué hace |
|----------|--------|----------|
| **TT** | Tokenización Transmutativa | Bytes multi-granularidad (2→4→8→16) |
| **ECN** | Embeddings Composicionales por Nivel | 24x reducción de memoria |
| **PFH** | Posiciones Fractales Híbridas | Consciencia de posición + nivel |
| **MPC** | Mezcla → Procesa Cercanos | Filosofía central de arquitectura |
| **FPD** | FFN Progresivo por Distancia | Expansión 0.5x → 0.75x → 1.0x |
| **EEM** | Early Exit Multietapa | Salida en caja Y nivel fractal |
| **CGC** | Contribuciones Gated por Caja | Control de contribución aprendido |
| **CEB** | Cache Evolutivo Binario | Jerarquía de caché L1/L2/L3 |

*Nombres propuestos por el fundador Lucas Ricardo Mella Chillemi*

📖 **[Ver diagramas y comparaciones completas →](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.es.md)**

---

## Tabla Comparativa

| Aspecto | Tradicional | LLARRI-O1 |
|---------|-------------|-----------|
| **Tokenización** | BPE (vocabulario fijo) | Bytes + niveles jerárquicos |
| **Embeddings** | 50K × 768 = 38M | 256 × 64 + MLPs = ~400K |
| **Tamaño FFN** | Siempre 4x | 0.5x → 0.75x → 1.0x |
| **Early exit** | Raro/ninguno | Integrado por caja |
| **Cómputo** | Fijo | Adaptativo |
| **Memoria** | ~500 MB | ~3.8 MB |

---

## Detalles del Modelo

| Propiedad | Valor |
|-----------|-------|
| **Autor** | Lucas Ricardo Mella Chillemi |
| **Organización** | Segunda Cabeza |
| **Coordinador** | Alvaro |
| **Licencia** | AGPL-3.0-or-later |
| **Parámetros** | ~544K |
| **Memoria** | ~3.8 MB |
| **Versión** | 2.0.0 |

---

## Inicio Rápido

```python
from llarri_o1 import LLARRILanguageModel, LLARRIConfig

# Crear modelo
config = LLARRIConfig(
    embed_dim=64,
    niveles=[2, 4, 8, 16],
    num_heads=4,
    ffn_expansion=2.0,
    num_vecinos=3,
    umbral_confianza=0.7
)

model = LLARRILanguageModel(config)

# Generar texto
output = model.generate(
    prompt="Hola",
    max_new_tokens=50,
    temperatura=0.8,
    top_k=40
)
print(output)
```

---

## Estado Actual

| Componente | Estado |
|------------|--------|
| Tokenizador Transmutativo (TT) | ✅ Completo |
| Embeddings Composicionales (ECN) | ✅ Completo |
| Posiciones Fractales (PFH) | ✅ Completo |
| Bloque 6-Cajas (MPC + FPD) | ✅ Completo |
| Early Exit (EEM) | ✅ Completo |
| Contribuciones Gated (CGC) | ✅ Completo |
| Caché Binario (CEB) | ✅ Completo |
| Generación end-to-end | ✅ Funcionando |
| Entrenamiento | 🔄 En progreso |

---

## Uso Previsto

- ✅ Investigación en arquitecturas transformer eficientes
- ✅ Exploración de mecanismos de cómputo adaptativo
- ✅ Propósitos educativos
- ⚠️ No listo para producción (sin entrenar)

---

## Citación

```bibtex
@software{llarri_o1,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {LLARRI-O1: Fractal Language Model with Neighbor-Progressive Processing},
  year = {2026},
  organization = {Segunda Cabeza},
  url = {https://github.com/lucasmella-stack/llarri-o1}
}
```

---

## Enlaces

- 📂 **GitHub:** [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)
- 📖 **Innovaciones:** [Documentación completa con diagramas](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/INNOVATIONS.es.md)
- 🏗️ **Arquitectura:** [Detalles técnicos](https://github.com/lucasmella-stack/llarri-o1/blob/main/docs/architecture/ARCHITECTURE.es.md)

---

## Contacto

- **Lucas Ricardo Mella Chillemi** — lucas@segundacabeza.com
- **Alvaro (Coordinador)** — alvaro@segundacabeza.com
- **Web:** [segundacabeza.com](https://segundacabeza.com)

---

<div align="center">

**Hecho con 💜 por Segunda Cabeza**

*"Mezcla primero, procesa con vecinos — de pequeño a grande"*

</div>
