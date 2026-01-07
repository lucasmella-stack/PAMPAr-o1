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
pipeline_tag: text-generation
---

# LLARRI-O1 — Model Card

**Versión en inglés:** [MODEL_CARD.md](MODEL_CARD.md)

## Descripción del Modelo

LLARRI-O1 es un modelo de lenguaje experimental que reimagina el procesamiento de redes neuronales a través de una **arquitectura de 6 cajas** inspirada en jerarquías de cache de CPU (L1/L2/L3).

### Filosofía Central: "Mezcla Primero, Procesa con Cercanos"

En lugar del patrón tradicional de Transformer (attention → FFN completo → repetir), LLARRI:
1. **Mezcla globalmente** (attention)
2. **Procesa primero con vecinos cercanos** (FFN pequeño)
3. **Expande solo si hace falta** (FFN progresivo)
4. **Sale temprano cuando hay confianza** (ahorra cómputo)

## Vista General de la Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    LLARRI-O1 (6 CAJAS)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Caja 1: MEZCLA (Attention) ─────────────────────────────► │
│                                                             │
│   Caja 2: PROCESA cercano (0.5x FFN) ──► Caja 5: EVALÚA ──►│
│           ▲                               │                 │
│           │                               ▼                 │
│   Caja 3: PROCESA medio (0.75x FFN) ◄── ¿Continuar?        │
│           │                               │                 │
│           ▼                               ▼                 │
│   Caja 4: PROCESA lejano (1.0x FFN) ───────────────────────►│
│                                                             │
│   Caja 6: OUTPUT ◄─────────────────────────────────────────►│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Innovaciones Clave

| Nombre | Descripción |
|--------|-------------|
| **Tokenización Transmutativa (TT)** | Agrupaciones jerárquicas de bytes (2→4→8→16), multi-granularidad en runtime |
| **Embeddings Composicionales por Nivel (ECN)** | Base + MLP por nivel, 24x reducción de memoria |
| **Posiciones Fractales Híbridas (PFH)** | Codificación posicional consciente del nivel fractal |
| **Mezcla → Procesa Cercanos (MPC)** | Filosofía central: attention global, procesamiento local progresivo |
| **FFN Progresivo por Distancia (FPD)** | Expansión 0.5x → 0.75x → 1.0x por "distancia de vecino" |
| **Early Exit Multietapa (EEM)** | Salida a nivel de caja Y a nivel fractal |
| **Contribuciones Gated por Caja (CGC)** | Cada caja de procesamiento controla su influencia |
| **Cache Evolutivo Binario (CEB)** | Cache tipo L1/L2/L3 para operaciones frecuentes |

*Nombres propuestos por el fundador, Lucas Ricardo Mella Chillemi.*

## Tradicional vs LLARRI

| Aspecto | Transformer Tradicional | LLARRI-O1 |
|---------|------------------------|-----------|
| Tamaño FFN | Siempre 4x | 0.5x → 0.75x → 1.0x |
| Salida temprana | Rara | Incorporada por caja |
| Cómputo | Fijo | Adaptativo |
| Embeddings | O(vocab × dim) | O(vocab × dim / 24) |
| Codificación posicional | Solo sin/cos | Sin/cos + conciencia de nivel |

## Detalles del Modelo

- **Autor:** Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- **Coordinador:** Alvaro (Segunda Cabeza)
- **Licencia:** AGPL-3.0-or-later
- **Parámetros:** ~544K (Language Model v2)
- **Memoria:** ~3.8 MB

## Inicio Rápido

```python
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

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
    temperatura=0.8
)
```

## Uso Previsto

- Investigación y experimentación en arquitecturas de transformers eficientes
- Exploración de mecanismos de cómputo adaptativo y salida temprana
- Propósitos educativos para entender diseños alternativos de LLM

## Limitaciones

- **Actualmente sin entrenar** — genera output aleatorio
- Arquitectura experimental, las interfaces pueden cambiar
- No listo para producción

## Estado Actual

| Componente | Estado |
|------------|--------|
| Tokenizador Transmutativo | ✅ Completo |
| Embeddings Composicionales | ✅ Completo |
| Bloque 6 Cajas | ✅ Completo |
| Early Exit | ✅ Completo |
| Generación end-to-end | ✅ Funcionando |
| Entrenamiento | 🔄 En progreso |

## Citación

```bibtex
@software{llarri_o1,
  author = {Mella Chillemi, Lucas Ricardo},
  title = {LLARRI-O1: Modelo de Lenguaje Fractal con Procesamiento Progresivo por Vecinos},
  year = {2026},
  organization = {Segunda Cabeza},
  url = {https://github.com/lucasmella-stack/llarri-o1}
}
```

## Contacto

- **Lucas Ricardo Mella Chillemi** — lucas@segundacabeza.com
- **Alvaro (Coordinador)** — alvaro@segundacabeza.com
- **Organización:** Segunda Cabeza

## Licencia

AGPL-3.0-or-later — Uso comercial permitido, servicios de red modificados deben compartir el código fuente.
