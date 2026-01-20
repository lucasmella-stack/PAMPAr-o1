# PampaR: Una Arquitectura Territorial Inspirada en el Cerebro para Modelado de Lenguaje

**Lucas Ricardo Mella Chillemi**  
Investigador Independiente  
Enero 2026

---

## Resumen

Presentamos PampaR, una arquitectura de modelo de lenguaje novedosa inspirada en la organización funcional del cerebro humano. A diferencia de las arquitecturas transformer estándar que tratan todos los cómputos de manera uniforme, PampaR introduce **Procesamiento Territorial** donde módulos neuronales especializados se organizan en territorios funcionales (Expresivo, Contextual, Formal, Estructural) coordinados por un **Tálamo** central que enruta tokens usando un enfoque híbrido: 70% reglas explícitas (LLAVES) y 30% atención aprendida. Nuestros experimentos en WikiText-103 demuestran que esta arquitectura logra perplejidad competitiva (PPL ~57) con solo 14M parámetros, mientras ofrece ventajas de interpretabilidad a través de reglas de enrutamiento explícitas.

---

## 1. Introducción

Los modelos de lenguaje grandes (LLMs) actuales como GPT, LLaMA y Claude logran un rendimiento notable pero sufren de varias limitaciones:

1. **Opacidad**: Los patrones de atención son difíciles de interpretar
2. **Homogeneidad**: Todas las capas realizan operaciones idénticas
3. **Ineficiencia**: Todos los tokens son procesados por todos los parámetros

El cerebro humano, en contraste, exhibe clara **especialización funcional**:
- Área de Broca para producción del lenguaje
- Área de Wernicke para comprensión del lenguaje
- Corteza prefrontal para razonamiento lógico
- Hipocampo para memoria y contexto

PampaR (Pampa Reasoning) se inspira en esta organización, implementando una **arquitectura territorial** donde:
- **Módulos** especializados (neuronas) manejan diferentes aspectos del lenguaje
- Un **Tálamo** central enruta información entre módulos
- **Fronteras bidireccionales** permiten comunicación inter-territorial
- **Reglas explícitas** (LLAVES) proporcionan enrutamiento interpretable

---

## 2. Arquitectura

### 2.1 Vista General

```
Entrada → Embedding → [Bloque Territorial ×N] → LM Head → Salida
                          ↓
              Tálamo (LLAVES 70% + Atención 30%)
                          ↓
    ┌─────────────────────┴─────────────────────┐
    │                                           │
    ▼                                           ▼
┌───────────────┐                    ┌───────────────┐
│   EXPRESIVO   │◄───── Frontera ───►│  CONTEXTUAL   │
│ Lenguaje+Crea │                    │   Contexto    │
└───────┬───────┘                    └───────┬───────┘
        │                                    │
        │◄────── Bidireccional ─────────────►│
        │                                    │
┌───────▼───────┐                    ┌───────▼───────┐
│    FORMAL     │◄───── Frontera ───►│ ESTRUCTURAL   │
│    Lógica     │                    │ Patrón+Mate   │
└───────────────┘                    └───────────────┘
```

### 2.2 Territorios

PampaR organiza el cómputo en **4 territorios funcionales**, cada uno conteniendo módulos especializados:

| Territorio | Módulos | Función | Región Cerebral Análoga |
|-----------|---------|----------|------------------------|
| **Expresivo** | Lenguaje, Creatividad | Generación de texto fluido, ideas nuevas | Área de Broca, Hemisferio derecho |
| **Contextual** | Contexto | Memoria de trabajo, coherencia | Hipocampo, CPF |
| **Formal** | Lógica | Razonamiento deductivo, reglas | Corteza prefrontal |
| **Estructural** | Patrones, Matemáticas | Secuencias, números, estructura | Lóbulo parietal |

### 2.3 El Tálamo

El Tálamo es el orquestador central que enruta tokens a los territorios apropiados. Implementa un **mecanismo de enrutamiento híbrido**:

```python
pesos_enrutamiento = peso_llaves * activacion_llaves + (1 - peso_llaves) * atencion_aprendida
```

Donde:
- `peso_llaves = 0.7` (70% reglas explícitas)
- `activacion_llaves` = enrutamiento basado en reglas del sistema LLAVES
- `atencion_aprendida` = enrutamiento aprendido vía mecanismo de atención

#### 2.3.1 Sistema de LLAVES

Las LLAVES son reglas de enrutamiento explícitas e interpretables basadas en patrones de tokens:

| Módulo | Ejemplos de Patrones |
|--------|---------------------|
| Lenguaje | "el", "la", "de", artículos, preposiciones |
| Matemáticas | dígitos 0-9, "+", "-", "=", "%" |
| Lógica | "si", "entonces", "porque", "por lo tanto" |
| Patrones | repeticiones, secuencias, estructuras |
| Contexto | pronombres, referencias, marcadores temporales |
| Creatividad | adjetivos, metáforas, combinaciones novedosas |

Esto proporciona **interpretabilidad**: podemos inspeccionar directamente por qué un token fue enrutado a un territorio específico.

### 2.4 Fronteras Bidireccionales

Los territorios se comunican vía **6 conexiones de frontera bidireccionales** con compuertas aprendidas:

```python
FRONTERAS = [
    ("expresivo", "contextual", 0.8),    # Alto: narrativa necesita contexto
    ("expresivo", "formal", 0.5),        # Medio: argumentación
    ("expresivo", "estructural", 0.4),   # Bajo: estructura creativa
    ("contextual", "formal", 0.6),       # Medio-alto: contexto lógico
    ("contextual", "estructural", 0.5),  # Medio: contexto de patrones
    ("formal", "estructural", 0.7),      # Alto: lógica matemática
]
```

Cada frontera implementa:
```python
salida = compuerta * transformar(territorio_a) + (1-compuerta) * transformar(territorio_b)
```

### 2.5 Componentes Opcionales

#### Motor de Axiomas
Implementa patrones de razonamiento deductivo:
- **Modus Ponens**: Si P→Q y P, entonces Q
- **Silogismo**: Si A→B y B→C, entonces A→C
- **Negación**: Operaciones lógicas NOT

#### Memoria Práctica
Almacena patrones exitosos/fallidos para recuperación rápida durante inferencia.

---

## 3. Detalles de Implementación

### 3.1 Configuración del Modelo

```python
ConfigPampaR(
    vocab_size=8000,        # Tokenizador BPE
    dim=160,                # Dimensión oculta
    n_heads=4,              # Cabezas de atención por módulo
    n_capas=4,              # Capas por módulo
    dropout=0.1,
    max_seq_len=256,
    peso_llaves=0.7,        # 70% reglas, 30% aprendido
    usar_axiomas=True,      # Habilitar motor de axiomas
    usar_memoria=True,      # Habilitar memoria práctica
)
```

**Parámetros Totales**: 14,069,410 (~14M)

### 3.2 Configuración de Entrenamiento

- **Dataset**: WikiText-103 (100M tokens)
- **Hardware**: NVIDIA GTX 1650 (4GB VRAM)
- **Tamaño de Batch**: 4 (efectivo 32 con acumulación de gradientes)
- **Longitud de Secuencia**: 128 tokens
- **Optimizador**: AdamW (lr=2e-4, weight_decay=0.01)
- **Precisión**: FP16 mixta

### 3.3 Entrenamiento Fragmentado

Debido a limitaciones de hardware, implementamos **entrenamiento fragmentado**:

| Fragmento | Tokens | Epochs | Acumulado |
|-----------|--------|--------|-----------|
| 1 | 10M | 3 | 30M |
| 2 | 20M | 3 | 90M |
| 3 | 35M | 3 | 195M |
| 4 | 50M | 3 | 345M |
| 5 | 75M | 2 | 495M |
| 6 | 100M | 2 | 695M |

---

## 4. Resultados

### 4.1 Progreso de Entrenamiento

| Fragmento | Loss Final | PPL Final | Mejora |
|-----------|------------|-----------|--------|
| 1 (10M) | 4.85 | 127.5 | Línea base |
| 2 (20M) | 4.22 | 68.1 | -46.6% PPL |
| 3 (35M) | ~4.05 | ~57.1 | -55.2% PPL |

### 4.2 Comparación con Líneas Base

| Modelo | Parámetros | PPL (WikiText-103) |
|--------|------------|-------------------|
| LSTM (Merity et al.) | 24M | 69.1 |
| Transformer-XL | 24M | 54.5 |
| **PAMPAr-o1 v9** | **14M** | **~57** |
| GPT-2 Small | 125M | 35.1 |

PampaR logra perplejidad competitiva con **40% menos parámetros** que modelos LSTM comparables.

### 4.3 Análisis de Interpretabilidad

A diferencia de transformers de caja negra, PampaR proporciona insight sobre el enrutamiento de tokens:

```
Token: "matemáticas" → Activación LLAVES:
  - Módulo Matemáticas: 0.85 (alto)
  - Módulo Lenguaje: 0.15 (bajo)
  
Token: "por lo tanto" → Activación LLAVES:
  - Módulo Lógica: 0.90 (alto)
  - Módulo Contexto: 0.10 (bajo)
```

---

## 5. Discusión

### 5.1 Ventajas

1. **Interpretabilidad**: LLAVES proporcionan fundamento explícito de enrutamiento
2. **Eficiencia de Parámetros**: PPL competitivo con menos parámetros
3. **Modularidad**: Fácil agregar/modificar territorios especializados
4. **Plausibilidad Biológica**: Refleja organización funcional del cerebro

### 5.2 Limitaciones

1. **Escala**: Aún no probado a escala de 1B+ parámetros
2. **Tareas**: Evaluado solo en modelado de lenguaje (PPL)
3. **Diseño de LLAVES**: Actualmente manual, podría ser aprendido

### 5.3 Trabajo Futuro

1. **Escalamiento**: Probar arquitectura a escalas de 1B, 7B parámetros
2. **Multi-tarea**: Evaluar en razonamiento, QA, generación de código
3. **LLAVES Aprendidas**: Descubrir automáticamente reglas de enrutamiento
4. **Alineación Neuro**: Comparar activaciones con datos de neuroimagen

---

## 6. Conclusión

PampaR demuestra que arquitecturas territoriales inspiradas en el cerebro pueden lograr rendimiento competitivo en modelado de lenguaje mientras proporcionan ventajas de interpretabilidad. La combinación de reglas explícitas (LLAVES) con atención aprendida ofrece una dirección prometedora para construir sistemas de IA más transparentes.

La arquitectura es código abierto bajo licencia AGPL-3.0, habilitando colaboración comunitaria e investigación adicional.

---

## Referencias

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Merity, S., et al. (2018). Regularizing and Optimizing LSTM Language Models.
3. Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla).
5. Fedorenko, E., & Thompson-Schill, S. L. (2014). Reworking the language network. Trends in Cognitive Sciences.

---

## Apéndice A: Reproducibilidad

### A.1 Repositorio de Código
- GitHub: `https://github.com/lucasmella-stack/llarri-o1`
- HuggingFace: `https://huggingface.co/lucas-mella/PAMPAr-o1`

### A.2 Comandos de Entrenamiento

```bash
# Fragmento 1
python scripts/train_fragmentado.py --fragmento 1

# Continuar entrenamiento
python scripts/train_fragmentado.py --fragmento 2

# Entrenamiento completo
python scripts/train_fragmentado.py --max
```

### A.3 Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU VRAM | 4GB | 8GB |
| RAM | 16GB | 32GB |
| Almacenamiento | 5GB | 20GB |

---

**Licencia**: AGPL-3.0-or-later  
**Copyright**: © 2024-2026 Lucas Ricardo Mella Chillemi
