# Arquitectura LLARRI-O1

English version: [ARCHITECTURE.md](ARCHITECTURE.md)

## Autoría
- Autor: Lucas Ricardo Mella Chillemi (Independent Researcher)
- Fecha: 2026-01-07

## Visión general
LLARRI-O1 explora un patrón de cómputo “fractal” con reutilización de parámetros:
- La entrada se proyecta a un espacio oculto.
- El vector oculto se divide en 4 **cuadrantes**.
- Cada cuadrante se procesa con un **pipeline fractal progresivo**.
- Los cuadrantes intercambian información mediante relaciones cruzadas.
- Múltiples “cajas” (etapas) se conectan mediante “llaves” residuales.

## Foco de implementación actual (v4.0 HyperComprimido)
Ideas clave:
- **6 cajas**: 3 cajas de procesamiento primario + 3 cajas de procesamiento secundario.
- **8 niveles fractales**: `2 → 4 → 8 → 16 → 32 → 64 → 128 → 256` (secuencial).
- **Cache binario** en nivel 2 (`CacheBinario`) para lookup rápido de operaciones básicas.
- **Las cajas secundarias incluyen auto-cálculos internos** entre valores intermedios.

## Notas de escalabilidad
- El nivel fractal máximo está restringido por `quad_dim = hidden_dim // 4`.
- Aumentar `hidden_dim` aumenta `quad_dim`, habilitando niveles mayores (p.ej. 256).
- La VRAM puede seguir siendo un límite durante entrenamiento por estados del optimizador (AdamW) y activaciones.

## Dónde mirar
- Paquete: [llarri_o1/](../../llarri_o1/)
  - Config: [llarri_o1/config.py](../../llarri_o1/config.py)
  - Modelo: [llarri_o1/model.py](../../llarri_o1/model.py)
  - Language Model (v2): [LLARRI_LANGUAGE_MODEL_V2.es.md](LLARRI_LANGUAGE_MODEL_V2.es.md)
  - Módulos: [llarri_o1/modules/](../../llarri_o1/modules/) (cache, niveles, relaciones, cajas, flujo)
  - Entrenamiento: [llarri_o1/training/trainer.py](../../llarri_o1/training/trainer.py)
  - Visualización: [llarri_o1/visualization/diagrams.py](../../llarri_o1/visualization/diagrams.py)
- Scripts: [scripts/train.py](../../scripts/train.py)
- Ejemplos: [examples/basic_usage.py](../../examples/basic_usage.py)
- Diagramas v4: [diagrams/v4-current](../../diagrams/v4-current)
