# Arquitectura LLARRI-O1

English version: [ARCHITECTURE.md](ARCHITECTURE.md)

## Autoría
- Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- Coordinador: Alvaro (Segunda Cabeza)
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
- Modelo: [src/llarri_o1_hypercomprimido.py](../../src/llarri_o1_hypercomprimido.py)
- Generador de diagramas: [src/generar_diagramas_v4_visuales.py](../../src/generar_diagramas_v4_visuales.py)
- Diagramas v4: [diagrams/v4-current](../../diagrams/v4-current)
