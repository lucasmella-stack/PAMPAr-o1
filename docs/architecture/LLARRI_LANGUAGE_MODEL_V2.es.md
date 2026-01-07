# LLARRI Language Model v2 — Técnicas e Innovaciones

## Autoría y nombres
- Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- Coordinador: Alvaro (Segunda Cabeza)
- Fecha: 2026-01-07

**Nota sobre nomenclatura**: los nombres marcados como **“nombre propuesto por el fundador”** fueron propuestos por **Lucas Ricardo Mella Chillemi (fundador)** para estandarizar y comunicar el diseño.

---

## 1) Visión general
LLARRI Language Model v2 es un modelo de lenguaje experimental que combina:
- Tokenización por bytes con niveles jerárquicos.
- Embeddings de memoria eficiente.
- Mezcla global (attention) seguida de procesamiento progresivo “por cercanía”.
- Salidas tempranas (early exit) para reducir cómputo cuando “alcanza”.

El corazón del modelo es el **BloqueFractal (6 cajas)**: primero mezcla, luego procesa en etapas crecientes, evalúa si puede frenar, y finalmente entrega el resultado.

---

## 2) Flujo completo (pipeline)

```mermaid
flowchart TD
  A[Texto] --> B[Tokenizador (bytes + niveles)]
  B --> C[Embedding Composicional]
  C --> D[Embedding Posicional Fractal]
  D --> E[BloqueFractal (6 cajas)]
  E --> F[LMHead (tied embeddings)]
  F --> G[Logits / Siguiente token]
```

**Implementación (código):**
- Modelo unificado: `llarri_o1/models/language_model.py`
- Tokenizador: `llarri_o1/modules/tokenizer.py`
- Embedding composicional: `llarri_o1/modules/` (integrado por el modelo)
- Posicional: `llarri_o1/modules/` (integrado por el modelo)
- BloqueFractal: `llarri_o1/modules/bloque_fractal.py`
- LMHead: `llarri_o1/modules/lm_head.py`

---

## 3) Técnicas existentes (estado del arte) usadas
Estas piezas son estándares (o variantes bien conocidas) en Transformers y modelos modernos:

- **Self-Attention**: mezcla/selección de información relevante entre tokens.
- **FFN / MLP por token**: procesamiento no lineal por posición.
- **Residual connections**: estabilidad y facilidad de optimización.
- **LayerNorm / normalización**: estabilidad del entrenamiento.
- **Tied embeddings**: reutiliza la matriz de embeddings como proyección de salida (reduce parámetros).
- **Byte-level tokens** (en el sentido de vocab de 256): robustez ante cualquier texto/bytes.
- **Early Exit (concepto general)**: detener el cómputo antes de completar todas las capas si el modelo ya “está seguro”.

---

## 4) Innovaciones / decisiones originales del diseño LLARRI
A continuación se documentan componentes y patrones del diseño que, en esta forma específica, constituyen el enfoque LLARRI.

### 4.1 Tokenización Transmutativa (TT) — nombre propuesto por el fundador
(antes: “Tokenización Fractal Jerárquica”)

**Idea**
- Se parte de **bytes** (vocab=256) y se construyen niveles jerárquicos de agrupación (p.ej. 2, 4, 8, 16).
- La tokenización es “transmutativa” en el sentido de que **el mismo texto puede representarse en múltiples granularidades** para luego decidir a qué nivel operar.

**Qué habilita**
- Un “zoom” dinámico: tokens chicos para detalle, tokens grandes para contexto.

**Diagrama (niveles por agrupación)**

```mermaid
flowchart LR
  A[Texto] --> B[Bytes (nivel 1)]
  B --> C[Grupos 2 (nivel 2)]
  C --> D[Grupos 4 (nivel 4)]
  D --> E[Grupos 8 (nivel 8)]
  E --> F[Grupos 16 (nivel 16)]
```

**Dónde está**
- Implementación principal: `llarri_o1/modules/tokenizer.py`

---

### 4.2 Embeddings Composicionales por Nivel (ECN)
**Idea**
- Se evita duplicar embeddings gigantes por nivel.
- Se usa un embedding base + una composición (MLP/transform) dependiente del nivel.

**Intuición (forma típica)**
- $e = e_{base}(token) + f_{nivel}(e_{base}(token))$

**Beneficio**
- Memoria mucho menor que “un embedding distinto por nivel”.

**Dónde está**
- Integrado en el modelo unificado: `llarri_o1/models/language_model.py`

---

### 4.3 Posiciones Fractales Híbridas (PFH)
**Idea**
- La posición incorpora una parte “sinusoidal/continua” y una parte “jerárquica” asociada al nivel.

**Beneficio**
- El modelo distingue no solo *dónde* está un token, sino *en qué granularidad* está trabajando.

**Dónde está**
- Integrado en el modelo unificado: `llarri_o1/models/language_model.py`

---

### 4.4 “Mezcla → Procesa cercanos” (MPC) — nombre propuesto por el fundador
**Idea central**
1) Primero se hace **mezcla global** (attention) para alinear relevancias.
2) Luego se hace **procesamiento progresivo por etapas** (cercano → medio → lejano) con costos crecientes.

**Lectura práctica**
- En vez de “todo el compute siempre”, se paga compute **solo si hace falta**.

**Dónde está**
- Bloque principal: `llarri_o1/modules/bloque_fractal.py`

---

### 4.5 FFN Progresivo por Distancia (FPD)
**Idea**
- Se definen varias cajas de procesamiento (Cajas 2–4) con **expansiones crecientes**.
- Ejemplo conceptual: 0.5× → 0.75× → 1.0×.

**Beneficio**
- Compute escalonado: barato primero, caro al final.

---

### 4.6 Contribuciones “gated” por caja (CGC)
**Idea**
- Cada caja de proceso aporta una contribución controlada por una compuerta (gate).

**Beneficio**
- El modelo puede aprender a “escuchar más o menos” cada etapa.

---

### 4.7 Early Exit Multietapa (EEM)
**Idea**
- Se evalúa tras cada etapa si la señal es suficientemente confiable.
- Si lo es, se corta el resto de las cajas.

**Diagrama (early exit dentro del bloque)**

```mermaid
flowchart TD
  M[ Caja 1: MEZCLA ] --> P1[ Caja 2: PROCESA (cercano) ]
  P1 --> E1{Caja 5: EVALÚA}
  E1 -- sale --> O[ Caja 6: OUTPUT ]
  E1 -- sigue --> P2[ Caja 3: PROCESA (medio) ]
  P2 --> E2{Caja 5: EVALÚA}
  E2 -- sale --> O
  E2 -- sigue --> P3[ Caja 4: PROCESA (lejano) ]
  P3 --> O
```

**Dónde está**
- Bloque principal: `llarri_o1/modules/bloque_fractal.py`

---

### 4.8 Cache Evolutivo Binario (CEB) — nombre propuesto por el fundador
(antes: “Cache Fractal Binario”)

**Idea**
- Cache “binario” porque opera con estructuras compactas y lookup rápido.
- “Evolutivo” porque se ajusta con el uso (la política/estructura puede adaptarse al patrón de acceso).

**Objetivo**
- Reducir recomputación y acelerar operaciones frecuentes, especialmente en niveles pequeños.

**Dónde está (en el repo)**
- Base de cache histórica del proyecto: `llarri_o1/modules/cache.py`

---

## 5) BloqueFractal (6 cajas): especificación concreta

**Cajas**
1. **Caja 1 — MEZCLA**: attention (mezcla contextual).
2. **Caja 2 — PROCESA cercano**: FFN liviano (expansión baja).
3. **Caja 3 — PROCESA medio**: FFN intermedio.
4. **Caja 4 — PROCESA lejano**: FFN completo.
5. **Caja 5 — EVALÚA**: decide early exit (umbral de confianza).
6. **Caja 6 — OUTPUT**: entrega la representación final al head.

**Diagrama (bloque completo)**

```mermaid
flowchart LR
  X[Input hidden] --> C1[1) MEZCLA]
  C1 --> C2[2) PROCESA cercano]
  C2 --> C5a{5) EVALÚA}
  C5a -- salir --> C6[6) OUTPUT]
  C5a -- seguir --> C3[3) PROCESA medio]
  C3 --> C5b{5) EVALÚA}
  C5b -- salir --> C6
  C5b -- seguir --> C4[4) PROCESA lejano]
  C4 --> C6
  C6 --> Y[Hidden final]
```

**Parámetros principales (config)**
- `ffn_expansion`: controla el tamaño relativo de los FFN.
- `num_vecinos`: controla cuántas etapas/vecinos se modelan.
- `umbral_confianza`: umbral para early exit.

**Dónde está**
- `llarri_o1/modules/bloque_fractal.py`

---

## 6) Glosario corto (nombres y siglas)
- **TT**: Tokenización Transmutativa (nombre propuesto por el fundador)
- **ECN**: Embeddings Composicionales por Nivel
- **PFH**: Posiciones Fractales Híbridas
- **MPC**: Mezcla → Procesa cercanos (nombre propuesto por el fundador)
- **FPD**: FFN Progresivo por Distancia
- **CGC**: Contribuciones Gated por Caja
- **EEM**: Early Exit Multietapa
- **CEB**: Cache Evolutivo Binario (nombre propuesto por el fundador)

---

## 7) Estado y notas prácticas
- El modelo funciona end-to-end (smoke test local) y genera texto (aún no entrenado).
- La tasa de early exit depende de `umbral_confianza` y del estado de entrenamiento.

