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
---

<div align="center">

# 🧠 LLARRI-O1

### Modelo de Lenguaje Fractal con Procesamiento Progresivo por Vecinos

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

**[English](README.md)** | **[Arquitectura](docs/architecture/ARCHITECTURE.es.md)** | **[Innovaciones](docs/architecture/LLARRI_LANGUAGE_MODEL_V2.es.md)**

*"Mezcla primero, procesa con cercanos — de menos a más"*

</div>

---

## 🎯 ¿Qué es LLARRI-O1?

LLARRI-O1 es un **modelo de lenguaje experimental** que reimagina cómo las redes neuronales procesan información. En lugar del patrón tradicional "attention → FFN → repetir", LLARRI usa:

1. **Mezclar globalmente** (attention para ver qué es relevante)
2. **Procesar primero con vecinos cercanos** (cómputo pequeño/barato)
3. **Expandir solo si hace falta** (cómputo progresivo)
4. **Salir temprano cuando hay confianza** (ahorrar recursos)

---

## 🔄 Tradicional vs LLARRI: Comparación Visual

### Transformer Tradicional

```
┌─────────────────────────────────────────────────────────────┐
│                   TRANSFORMER ESTÁNDAR                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Entrada ──► [Attention] ──► [FFN Completo] ──► Salida    │
│                    │              │                         │
│                    │              │                         │
│              Siempre corre   Siempre 4x expansión          │
│              cómputo completo sin importar necesidad        │
│                                                             │
│   • Cómputo fijo por capa                                   │
│   • Sin salida temprana                                     │
│   • Mismo costo para entradas "fáciles" y "difíciles"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### LLARRI-O1 (Arquitectura de 6 Cajas)

```
┌─────────────────────────────────────────────────────────────┐
│                    LLARRI-O1 (6 CAJAS)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Entrada ──► [MEZCLA] ──► [PROCESA] ──► [EVALÚA] ──► Salida│
│               Attention     FFN           Salida            │
│                             Progresivo    Temprana          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 1: MEZCLA (Attention)                         │   │
│   │    "¿Qué información es relevante?"                 │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 2: PROCESA cercano (0.5x FFN) ◄── Chico/Rápido│   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 5: EVALÚA ──► ¿Confianza? ──► SALIR TEMPRANO ✓│   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼ (si no hay confianza)              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 3: PROCESA medio (0.75x FFN)                  │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 5: EVALÚA ──► ¿Confianza? ──► SALIR TEMPRANO ✓│   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼ (si no hay confianza)              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 4: PROCESA lejano (1.0x FFN) ◄── Cómputo full │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Caja 6: OUTPUT                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   • Cómputo adaptativo (entradas fáciles salen temprano)    │
│   • Costo progresivo (chico → medio → grande)               │
│   • Como cache de CPU: L1 (rápido) → L2 → L3 (lento)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Comparación de Componentes

### Tokenización

| Aspecto | Tradicional (BPE/WordPiece) | LLARRI (Transmutativa) |
|---------|----------------------------|------------------------|
| Unidad base | Subpalabras | Bytes (256 vocab) |
| Niveles | Vocabulario único fusionado | Jerárquico (2→4→8→16) |
| Flexibilidad | Fijo post-entrenamiento | Multi-granularidad en runtime |
| Memoria | Embeddings de vocab grande | Base pequeña + composición |

```
BPE TRADICIONAL:
  "Hola" → ["Ho", "la"] → [1542, 283]
  Vocabulario fijo, una sola representación

LLARRI TRANSMUTATIVA:
  "Hola" → Nivel 1: [72,111,108,97]  (bytes)
         → Nivel 2: [18543, 27745]   (bigramas)
         → Nivel 4: [...]             (4-gramas)
  Múltiples vistas, elegir granularidad por tarea
```

### Embeddings

| Aspecto | Tradicional | LLARRI (Composicional) |
|---------|-------------|----------------------|
| Almacenamiento | Separado por token | Base + MLP por nivel |
| Memoria | O(vocab × dim) | O(vocab × dim / 24) |
| Conciencia de nivel | Ninguna | Incorporada |

```
TRADICIONAL:
  embed["Hola"] = lookup[token_id]  # 50K × 768 = 38M params

LLARRI COMPOSICIONAL:
  embed["Hola"] = base[byte] + MLP_nivel(base[byte])
  # 256 × 64 + MLPs pequeños = ~1.5M params (24x menor)
```

### Codificación Posicional

| Aspecto | Tradicional | LLARRI (Fractal Híbrida) |
|---------|-------------|------------------------|
| Tipo | Sinusoidal o aprendida | Sinusoidal + jerárquica |
| Info de nivel | Ninguna | Codifica granularidad |
| Posición + Nivel | Separados | Combinados |

```
TRADICIONAL:
  pos_embed(i) = sin/cos(i)

LLARRI FRACTAL HÍBRIDA:
  pos_embed(i, nivel) = sin/cos(i) + nivel_embed[nivel] + jerarquico(i, nivel)
  # La posición sabe EN QUÉ NIVEL está operando
```

### Procesamiento

| Aspecto | Transformer Tradicional | LLARRI 6 Cajas |
|---------|------------------------|----------------|
| Patrón | Attention → FFN (fijo) | Mezcla → Procesa Progresivo |
| Tamaño FFN | Siempre 4x | 0.5x → 0.75x → 1.0x |
| Salida temprana | Rara, por capa | Incorporada, por caja |
| Cómputo | Fijo | Adaptativo |

```
TRADICIONAL:
  for capa in capas:
      x = attention(x)      # Siempre completo
      x = ffn(x)            # Siempre 4x expansión

LLARRI:
  x = mezcla(x)             # Attention
  x = procesa_cercano(x)    # 0.5x FFN
  if confianza(x): return   # ¡Salida temprana!
  x = procesa_medio(x)      # 0.75x FFN  
  if confianza(x): return   # ¡Salida temprana!
  x = procesa_lejano(x)     # 1.0x FFN
```

---

## 🚀 Innovaciones Clave

| Innovación | Nombre (propuesto por el fundador) | Descripción |
|------------|-----------------------------------|-------------|
| **TT** | Tokenización Transmutativa | Agrupaciones jerárquicas de bytes, multi-granularidad |
| **ECN** | Embeddings Composicionales por Nivel | Composición base + MLP, 24x reducción de memoria |
| **PFH** | Posiciones Fractales Híbridas | Posición + conciencia de nivel combinadas |
| **MPC** | Mezcla → Procesa Cercanos | Filosofía central: mezclar global, procesar progresivo |
| **FPD** | FFN Progresivo por Distancia | Expansión 0.5x → 0.75x → 1.0x por "distancia de vecino" |
| **EEM** | Early Exit Multietapa | Salida a nivel de caja Y a nivel fractal |
| **CGC** | Contribuciones Gated por Caja | Cada caja controla su influencia vía gates |
| **CEB** | Cache Evolutivo Binario | Cache tipo L1/L2/L3 para operaciones frecuentes |

---

## 💾 Comparación de Memoria

```
┌────────────────────────────────────────────────────────────┐
│              COMPARACIÓN DE HUELLA DE MEMORIA              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  GPT-2 Small (117M params):                                │
│  ████████████████████████████████████████  ~500 MB         │
│                                                            │
│  BERT-Base (110M params):                                  │
│  ███████████████████████████████████████   ~440 MB         │
│                                                            │
│  LLARRI-O1 LM v2 (544K params):                            │
│  █                                         ~3.8 MB         │
│                                                            │
│  Factor: ~100x menor para expresividad comparable          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🔧 Inicio Rápido

### Instalación

```bash
git clone https://github.com/lucasmella-stack/llarri-o1.git
cd llarri-o1
pip install -r requirements.txt
```

### Uso del Modelo de Lenguaje

```python
from llarri_o1.models.language_model import LLARRILanguageModel, LLARRIConfig

# Crear modelo
config = LLARRIConfig(
    embed_dim=64,
    base_dim=64,
    max_dim=128,
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

### Ejecutar Tests

```bash
# Test modelo de lenguaje
python -m llarri_o1.models.language_model

# Test bloque fractal
python -m llarri_o1.modules.bloque_fractal
```

---

## 📁 Estructura del Proyecto

```
llarri-o1/
├── llarri_o1/
│   ├── models/
│   │   └── language_model.py    # LM unificado con arquitectura 6 cajas
│   ├── modules/
│   │   ├── tokenizer.py         # Tokenización Transmutativa
│   │   ├── bloque_fractal.py    # Bloque 6 Cajas (MPC + FPD + EEM)
│   │   ├── lm_head.py           # Head con tied embeddings
│   │   ├── cache.py             # Cache Evolutivo Binario
│   │   └── ...
│   ├── training/
│   └── utils/
├── docs/
│   ├── architecture/
│   │   ├── ARCHITECTURE.es.md
│   │   └── LLARRI_LANGUAGE_MODEL_V2.es.md  # Doc completo de innovaciones
│   └── huggingface/
│       └── MODEL_CARD.es.md
├── examples/
├── tests/
└── README.es.md
```

---

## 📊 Estado Actual

| Componente | Estado | Notas |
|------------|--------|-------|
| Tokenizador Transmutativo | ✅ Completo | 4 niveles jerárquicos |
| Embeddings Composicionales | ✅ Completo | 24x reducción de memoria |
| Posiciones Fractales | ✅ Completo | Conscientes de nivel |
| Bloque 6 Cajas | ✅ Completo | Mezcla → Procesa → Evalúa → Output |
| Early Exit | ✅ Completo | Multi-etapa |
| LM Head | ✅ Completo | Tied embeddings |
| Generación end-to-end | ✅ Funcionando | Sin entrenar (output aleatorio) |
| Entrenamiento | 🔄 Siguiente | Dataset + loop de entrenamiento |

---

## 👥 Créditos

| Rol | Nombre | Contacto |
|-----|--------|----------|
| **Fundador y Creador** | Lucas Ricardo Mella Chillemi | lucas@segundacabeza.com |
| **Coordinador** | Alvaro | alvaro@segundacabeza.com |

### Organización

**Segunda Cabeza** — Innovación en IA

- 🌐 Web: [segundacabeza.com](https://segundacabeza.com)
- 🤗 HuggingFace: [lucas-mella/llarri-o1](https://huggingface.co/lucas-mella/llarri-o1)
- 📂 GitHub: [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)

---

## 📄 Licencia

**GNU AGPL-3.0-or-later**

- ✅ Uso comercial permitido
- ✅ Uso en investigación y educación
- 📝 Servicios de red modificados deben compartir el código
- 📝 Atribución requerida

Ver [LICENSE](LICENSE) para detalles.

---

<div align="center">

**Hecho con 💜 por Segunda Cabeza**

*"Mezcla primero, procesa con cercanos — de menos a más"*

</div>
