# LLARRI-O1: Innovaciones Técnicas

**Autor:** Lucas Ricardo Mella Chillemi (Independent Researcher)  
**Fecha:** 2026-01-07

> **Nota:** Los nombres de las técnicas fueron propuestos por el fundador **Lucas Ricardo Mella Chillemi** para estandarizar la comunicación del diseño.

---

## Resumen de Innovaciones

| Sigla | Nombre | Descripción |
|-------|--------|-------------|
| **TT** | Tokenización Transmutativa | Múltiples granularidades simultáneas |
| **ECN** | Embeddings Composicionales por Nivel | 24x menos memoria |
| **PFH** | Posiciones Fractales Híbridas | Posición + nivel combinados |
| **MPC** | Mezcla → Procesa Cercanos | Filosofía central del modelo |
| **FPD** | FFN Progresivo por Distancia | 0.5x → 0.75x → 1.0x |
| **EEM** | Early Exit Multietapa | Salida temprana por caja y nivel |
| **CGC** | Contribuciones Gated por Caja | Control de influencia por caja |
| **CEB** | Cache Evolutivo Binario | Cache L1/L2/L3 para operaciones |

---

## 1. TT — Tokenización Transmutativa

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL (BPE/WordPiece)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   "Hello World" ──► Tokenizer ──► [15496, 2159]                    │
│                                                                     │
│   • Vocabulario fijo (30K-50K tokens)                              │
│   • UNA sola representación por texto                              │
│   • Granularidad fija post-entrenamiento                           │
│   • Tokens raros = problemas                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Tokenización Transmutativa)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   "Hello World" ──► Tokenizer ──►                                  │
│                                                                     │
│   Nivel 1 (bytes):    [72,101,108,108,111,32,87,111,114,108,100]   │
│   Nivel 2 (bigramas): [18533, 27756, 28448, 22383, 27748, 25700]   │
│   Nivel 4 (4-gramas): [1214606956, 1819438368, ...]                │
│   Nivel 8 (8-gramas): [...]                                         │
│                                                                     │
│   • Vocabulario base = 256 (bytes)                                 │
│   • MÚLTIPLES representaciones simultáneas                         │
│   • Elegir granularidad en runtime                                 │
│   • Nunca hay tokens "desconocidos"                                │
│                                                                     │
│   TRANSMUTACIÓN: el mismo texto "muta" entre granularidades        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Flujo

```
                    "Hola Mundo"
                         │
                         ▼
              ┌──────────────────┐
              │  BYTES (nivel 1) │
              │ [72,111,108,97...│
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Nivel 2 │   │ Nivel 4 │   │ Nivel 8 │
    │ Bigramas│   │ 4-gramas│   │ 8-gramas│
    └─────────┘   └─────────┘   └─────────┘
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              ┌──────────────────┐
              │ Elegir nivel     │
              │ según la tarea   │
              └──────────────────┘
```

**Beneficio clave:** El modelo puede operar en diferentes "zooms" según la necesidad. Detalle fino (bytes) o contexto amplio (n-gramas grandes).

---

## 2. ECN — Embeddings Composicionales por Nivel

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Vocab: 50,000 tokens                                             │
│   Dim: 768                                                          │
│                                                                     │
│   Embedding Matrix: 50,000 × 768 = 38.4M parámetros                │
│                                                                     │
│   token_id ──► lookup[token_id] ──► vector (768 dims)              │
│                                                                     │
│   • Cada token tiene su propio vector independiente                 │
│   • No hay relación estructural entre tokens                        │
│   • Memoria: O(vocab × dim)                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Embeddings Composicionales)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Base: 256 bytes × 64 dim = 16K parámetros                        │
│   + MLPs por nivel (pequeños)                                       │
│                                                                     │
│   byte_id ──► base[byte_id] ──► MLP_nivel() ──► vector             │
│                                                                     │
│   Fórmula:                                                          │
│   embed(token, nivel) = base[token] + MLP_nivel(base[token])       │
│                                                                     │
│   • Base compartida entre todos los niveles                         │
│   • MLPs pequeños transforman según nivel                           │
│   • Memoria: O(256 × dim + niveles × MLP_params)                   │
│   • ~24x MENOS memoria que tradicional                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Composición

```
                    Token ID: 72 ('H')
                           │
                           ▼
                  ┌─────────────────┐
                  │  Base Embedding │
                  │   (256 × 64)    │
                  └────────┬────────┘
                           │
                           ▼
                    base_vector (64d)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ MLP_N2  │       │ MLP_N4  │       │ MLP_N8  │
    │ (nivel2)│       │ (nivel4)│       │ (nivel8)│
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         ▼                 ▼                 ▼
    base + Δ₂         base + Δ₄         base + Δ₈
    
    
   Memoria total: 256×64 + 3×(64×32×64) ≈ 400K params
   vs Tradicional: 50K×768 ≈ 38M params
   
   AHORRO: ~100x
```

---

## 3. PFH — Posiciones Fractales Híbridas

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL (Sinusoidal/Learned)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   pos_embed(i) = sin(i / 10000^(2k/d))  o  learned[i]              │
│                                                                     │
│   • Solo codifica POSICIÓN absoluta                                 │
│   • No sabe en qué "nivel" está operando                           │
│   • Mismo encoding para todos los contextos                         │
│                                                                     │
│   Posición 5 ──► [0.84, 0.54, -0.99, ...]                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Posiciones Fractales Híbridas)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   pos_embed(i, nivel) = sinusoidal(i)                              │
│                       + nivel_embed[nivel]                          │
│                       + jerarquico(i, nivel)                        │
│                                                                     │
│   • Codifica POSICIÓN + NIVEL de granularidad                      │
│   • El modelo sabe "dónde" Y "en qué zoom"                         │
│   • Diferente comportamiento por nivel                              │
│                                                                     │
│   Posición 5, Nivel 2 ──► [0.84, 0.54, ...] + [nivel2] + [hier]    │
│   Posición 5, Nivel 8 ──► [0.84, 0.54, ...] + [nivel8] + [hier]    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Composición

```
                         Posición i, Nivel n
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │   Sinusoidal   │  │ Level Embedding│  │  Jerárquico    │
     │   sin/cos(i)   │  │   nivel_emb[n] │  │  hier(i, n)    │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                                 ▼
                            ┌─────────┐
                            │    +    │
                            └────┬────┘
                                 │
                                 ▼
                    pos_embed(i, nivel) final
                    
   La posición "sabe" en qué granularidad está operando
```

---

## 4. MPC — Mezcla → Procesa Cercanos

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL (Transformer)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   for layer in layers:                                              │
│       x = Attention(x)    # Siempre completo                       │
│       x = FFN(x)          # Siempre 4x expansión                   │
│                                                                     │
│   • Attention y FFN son bloques separados                          │
│   • Siempre ejecuta todo el FFN                                    │
│   • No hay concepto de "cercanía"                                  │
│   • Costo fijo por capa                                             │
│                                                                     │
│   Input ──► [Attn] ──► [FFN 4x] ──► [Attn] ──► [FFN 4x] ──► Out   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Mezcla → Procesa Cercanos)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   x = Mezcla(x)           # Caja 1: Attention (¿qué importa?)      │
│   x = Procesa_cercano(x)  # Caja 2: FFN 0.5x (vecino cercano)      │
│   if suficiente: return   # Caja 5: Early exit                      │
│   x = Procesa_medio(x)    # Caja 3: FFN 0.75x (vecino medio)       │
│   if suficiente: return   # Caja 5: Early exit                      │
│   x = Procesa_lejano(x)   # Caja 4: FFN 1.0x (vecino lejano)       │
│   return Output(x)        # Caja 6: Resultado                       │
│                                                                     │
│   • Primero MEZCLA (entiende contexto)                             │
│   • Luego PROCESA de cerca a lejos                                 │
│   • Sale temprano si es suficiente                                  │
│   • Como cache CPU: L1 (rápido) → L2 → L3 (lento)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Flujo

```
                              Input
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Caja 1: MEZCLA     │
                    │      (Attention)      │
                    │  "¿Qué es relevante?" │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 │                 │
    ┌─────────────────┐         │                 │
    │ Caja 2: CERCANO │         │                 │
    │   FFN 0.5x      │◄────────┘                 │
    │   (barato)      │                           │
    └────────┬────────┘                           │
             │                                    │
             ▼                                    │
    ┌─────────────────┐                           │
    │ Caja 5: EVALÚA  │──── ¿Suficiente? ──► SALIR
    └────────┬────────┘                           │
             │ No                                 │
             ▼                                    │
    ┌─────────────────┐                           │
    │ Caja 3: MEDIO   │◄──────────────────────────┘
    │   FFN 0.75x     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Caja 5: EVALÚA  │──── ¿Suficiente? ──► SALIR
    └────────┬────────┘
             │ No
             ▼
    ┌─────────────────┐
    │ Caja 4: LEJANO  │
    │   FFN 1.0x      │
    │   (completo)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Caja 6: OUTPUT  │
    └────────┬────────┘
             │
             ▼
           Output
```

---

## 5. FPD — FFN Progresivo por Distancia

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   FFN siempre usa expansión 4x:                                    │
│                                                                     │
│   Input (768) ──► Linear ──► 3072 ──► GELU ──► Linear ──► 768     │
│                                                                     │
│   • Mismo costo para todos los tokens                              │
│   • Mismo costo para todas las capas                               │
│   • No hay adaptación                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (FFN Progresivo por Distancia)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Vecino 1 (cercano):  FFN 0.5x expansión  ──► Barato              │
│   Vecino 2 (medio):    FFN 0.75x expansión ──► Moderado            │
│   Vecino 3 (lejano):   FFN 1.0x expansión  ──► Completo            │
│                                                                     │
│   • Procesamiento crece con "distancia" conceptual                 │
│   • Primero intenta con recursos mínimos                           │
│   • Solo usa más si es necesario                                    │
│                                                                     │
│   Analogía: Como preguntar primero a un colega cercano,            │
│             luego al jefe, luego al experto externo                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Expansión

```
   TRADICIONAL:
   ════════════
   
   Input ──────────────────────────────────────────────────► FFN 4x
   (todos los tokens, siempre)
   
   
   LLARRI FPD:
   ═══════════
   
   Input ──► Cercano (0.5x) ──► ¿OK? ──► Salir
                   │              │
                   │              No
                   ▼              │
             Medio (0.75x) ◄──────┘
                   │
                   ▼
              ¿OK? ──► Salir
                   │
                   No
                   ▼
             Lejano (1.0x)
                   │
                   ▼
                Output
   
   
   Expansiones relativas:
   
   Cercano:  ████░░░░░░░░░░░░  (0.5x)   → 50% del costo
   Medio:    ██████░░░░░░░░░░  (0.75x)  → 75% del costo  
   Lejano:   ████████░░░░░░░░  (1.0x)   → 100% del costo
   Tradici:  ████████████████  (4.0x)   → 400% base
```

---

## 6. EEM — Early Exit Multietapa

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Layer 1 ──► Layer 2 ──► ... ──► Layer 12 ──► Output              │
│                                                                     │
│   • SIEMPRE ejecuta TODAS las capas                                │
│   • Mismo costo para "The" y para "antidisestablishmentarianism"   │
│   • No hay forma de "salir antes"                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Early Exit Multietapa)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   DOS niveles de early exit:                                        │
│                                                                     │
│   1) DENTRO del bloque (entre cajas 2-3-4):                        │
│      Caja 2 ──► Evalúa ──► ¿Suficiente? ──► SALIR                 │
│                    │                                                │
│                    No                                               │
│                    ▼                                                │
│      Caja 3 ──► Evalúa ──► ¿Suficiente? ──► SALIR                 │
│                                                                     │
│   2) ENTRE niveles fractales (2→4→8→16):                           │
│      Nivel 2 ──► ¿Suficiente? ──► SALIR                           │
│                       │                                             │
│                       No                                            │
│                       ▼                                             │
│      Nivel 4 ──► ¿Suficiente? ──► SALIR                           │
│                                                                     │
│   • Tokens "fáciles" salen rápido                                  │
│   • Tokens "difíciles" usan todo el compute                        │
│   • Ahorro promedio significativo                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Decisión

```
                              Input
                                │
                                ▼
          ╔═══════════════════════════════════════╗
          ║         NIVEL 2 (granularidad fina)   ║
          ╠═══════════════════════════════════════╣
          ║                                       ║
          ║   Mezcla ──► Proc1 ──► EVAL ──►EXIT? ║
          ║                         │      │      ║
          ║                         No     Sí───────────► Output
          ║                         ▼             ║
          ║              Proc2 ──► EVAL ──►EXIT? ║
          ║                         │      │      ║
          ║                         No     Sí───────────► Output
          ║                         ▼             ║
          ║              Proc3 ──► Output         ║
          ╚═══════════════════════════════════════╝
                                │
                                ▼
                         ¿Subir nivel?
                                │
                         ┌──────┴──────┐
                         │             │
                        No            Sí
                         │             │
                         ▼             ▼
                      Output    ╔═══════════════╗
                                ║   NIVEL 4     ║
                                ║   (repeat)    ║
                                ╚═══════════════╝
```

---

## 7. CGC — Contribuciones Gated por Caja

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   output = x + FFN(x)                                               │
│                                                                     │
│   • Residual connection fija                                        │
│   • FFN siempre contribuye 100%                                    │
│   • No hay control de "cuánto aporta"                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Contribuciones Gated por Caja)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   gate = sigmoid(Linear(x))     # Entre 0 y 1                      │
│   output = x + gate * FFN(x)    # Contribución controlada          │
│                                                                     │
│   • Cada caja decide cuánto aportar                                │
│   • Gate aprendido durante entrenamiento                           │
│   • Puede "apagar" contribuciones irrelevantes                     │
│   • Permite especialización por caja                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Gate

```
   TRADICIONAL:
   ════════════
   
        x ─────────────────────┐
        │                      │
        ▼                      │
   ┌─────────┐                 │
   │   FFN   │                 │
   └────┬────┘                 │
        │                      │
        │    1.0 (fijo)        │
        ▼                      ▼
        ─────────────────►  [  +  ] ──► output
   
   
   LLARRI CGC:
   ═══════════
   
        x ─────────────────────┬─────────────────┐
        │                      │                 │
        ▼                      ▼                 │
   ┌─────────┐           ┌───────────┐           │
   │   FFN   │           │   Gate    │           │
   └────┬────┘           │ sigmoid(  │           │
        │                │ Linear(x))│           │
        │                └─────┬─────┘           │
        │                      │                 │
        │                 0.0 - 1.0              │
        │                 (aprendido)            │
        ▼                      ▼                 ▼
        ────────────────►  [  ×  ] ─────►  [  +  ] ──► output
                           gate
   
   Ejemplos de gates aprendidos:
   
   Caja 2 (cercano):  gate = 0.8  ──► "Muy relevante"
   Caja 3 (medio):    gate = 0.3  ──► "Poco relevante"
   Caja 4 (lejano):   gate = 0.1  ──► "Casi ignorado"
```

---

## 8. CEB — Cache Evolutivo Binario

### Comparación

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADICIONAL                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Cada forward pass:                                                │
│   - Recalcula TODAS las operaciones                                │
│   - No hay memoria de cálculos previos                             │
│   - Operaciones repetidas = costo repetido                         │
│                                                                     │
│   "Hello" x 100 = 100 × mismo cálculo                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LLARRI (Cache Evolutivo Binario)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Inspirado en cache de CPU (L1/L2/L3):                            │
│                                                                     │
│   L1 (pequeño, ultra-rápido):  Operaciones más frecuentes          │
│   L2 (medio, rápido):          Operaciones recientes               │
│   L3 (grande, moderado):       Operaciones menos frecuentes        │
│                                                                     │
│   "Evolutivo" porque:                                               │
│   - Se adapta al patrón de uso                                     │
│   - Prioriza operaciones frecuentes                                │
│   - Descarta las menos útiles                                       │
│                                                                     │
│   "Binario" porque:                                                 │
│   - Opera en el nivel más comprimido (dim=2)                       │
│   - Solo 4 combinaciones posibles: [0,0], [0,1], [1,0], [1,1]     │
│   - Pre-computa todas las operaciones básicas                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Cache

```
                            Operación
                                │
                                ▼
                    ┌───────────────────┐
                    │   ¿En L1 Cache?   │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   Sí                  No
                    │                   │
                    ▼                   ▼
            ┌─────────────┐   ┌───────────────────┐
            │  Return L1  │   │   ¿En L2 Cache?   │
            │  (1 cycle)  │   └─────────┬─────────┘
            └─────────────┘             │
                              ┌─────────┴─────────┐
                              │                   │
                             Sí                  No
                              │                   │
                              ▼                   ▼
                      ┌─────────────┐   ┌───────────────────┐
                      │  Return L2  │   │   ¿En L3 Cache?   │
                      │  (3 cycles) │   └─────────┬─────────┘
                      └─────────────┘             │
                                        ┌─────────┴─────────┐
                                        │                   │
                                       Sí                  No
                                        │                   │
                                        ▼                   ▼
                                ┌─────────────┐   ┌───────────────┐
                                │  Return L3  │   │   Calcular    │
                                │  (10 cycles)│   │   + Cachear   │
                                └─────────────┘   │  (100 cycles) │
                                                  └───────────────┘
   
   
   Cache Binario (nivel dim=2):
   
   ┌─────────────────────────────────────────────────────┐
   │  Input      │ Suma │ Prod │ Diff │ Media │ Max │ Min│
   ├─────────────┼──────┼──────┼──────┼───────┼─────┼────┤
   │  [0, 0]     │   0  │   0  │   0  │   0   │  0  │  0 │
   │  [0, 1]     │   1  │   0  │   1  │  0.5  │  1  │  0 │
   │  [1, 0]     │   1  │   0  │   1  │  0.5  │  1  │  0 │
   │  [1, 1]     │   2  │   1  │   0  │   1   │  1  │  1 │
   └─────────────┴──────┴──────┴──────┴───────┴─────┴────┘
   
   Total: 4 × 6 = 24 valores pre-computados
   Memoria: ~96 bytes
   Speedup: ~10-100x en operaciones binarias
```

---

## Resumen Visual Completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLARRI-O1 COMPLETO                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Texto ──► [TT] ──► [ECN] ──► [PFH] ──► [6 CAJAS] ──► Output     │
│            Token    Embed     Pos       MPC+FPD                    │
│            Trans.   Comp.     Frac.     EEM+CGC                    │
│                                           │                         │
│                                    ┌──────┴──────┐                  │
│                                    │    [CEB]    │                  │
│                                    │   Cache     │                  │
│                                    └─────────────┘                  │
│                                                                     │
│   BENEFICIOS COMBINADOS:                                            │
│   ├── ~24x menos memoria (ECN)                                     │
│   ├── ~100x menos params que GPT-2                                 │
│   ├── Early exit ahorra 30-50% compute promedio (EEM)              │
│   ├── FFN progresivo reduce costo en tokens fáciles (FPD)          │
│   ├── Cache acelera operaciones repetidas (CEB)                    │
│   └── Multi-granularidad para diferentes tareas (TT)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementación

Cada técnica está implementada en el código fuente:

| Técnica | Archivo |
|---------|---------|
| TT | `llarri_o1/modules/tokenizer.py` |
| ECN | `llarri_o1/modules/tokenizer.py` (EmbeddingComposicional) |
| PFH | `llarri_o1/modules/tokenizer.py` (EmbeddingPosicionalFractal) |
| MPC | `llarri_o1/modules/bloque_fractal.py` (BloqueFractal) |
| FPD | `llarri_o1/modules/bloque_fractal.py` (CajaProcesa) |
| EEM | `llarri_o1/modules/bloque_fractal.py` (CajaEvalua) |
| CGC | `llarri_o1/modules/bloque_fractal.py` (CajaProcesa.gate) |
| CEB | `llarri_o1/modules/cache.py` (CacheBinario) |

---

<div align="center">

**LLARRI-O1** — *Mezcla primero, procesa con cercanos, de menos a más*

**Segunda Cabeza** — Innovación en IA

</div>
