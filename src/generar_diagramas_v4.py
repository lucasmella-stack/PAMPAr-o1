"""
Generador de Diagramas LLARRI-O1 v4.0
=====================================

Genera diagramas ASCII y visualizaciones para la documentación.

Autor: Lucas Mella (Segunda Cabeza)
"""

import os


def crear_diagrama_arquitectura_v4():
    """Diagrama principal de arquitectura v4.0"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                       LLARRI-O1 v4.0 - HYPERCOMPRIMIDO                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                         CAPA DE DATOS                                  │  ║
║  │                                                                        │  ║
║  │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │  ║
║  │    │   CAJA A     │◄──►│   CAJA B     │◄──►│   CAJA C     │           │  ║
║  │    │   (datos)    │    │   (datos)    │    │   (datos)    │           │  ║
║  │    │ ┌──┬──┐      │    │ ┌──┬──┐      │    │ ┌──┬──┐      │           │  ║
║  │    │ │A1│A2│      │    │ │B1│B2│      │    │ │C1│C2│      │           │  ║
║  │    │ ├──┼──┤      │    │ ├──┼──┤      │    │ ├──┼──┤      │           │  ║
║  │    │ │A3│A4│      │    │ │B3│B4│      │    │ │C3│C4│      │           │  ║
║  │    │ └──┴──┘      │    │ └──┴──┘      │    │ └──┴──┘      │           │  ║
║  │    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │  ║
║  │           │                   │                   │                    │  ║
║  └───────────┼───────────────────┼───────────────────┼────────────────────┘  ║
║              │                   │                   │                       ║
║              ▼                   ▼                   ▼                       ║
║        ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐                 ║
║        │  LLAVE    │       │  LLAVE    │       │  LLAVE    │                 ║
║        │  A↔A'     │       │  B↔B'     │       │  C↔C'     │                 ║
║        └─────┬─────┘       └─────┬─────┘       └─────┬─────┘                 ║
║              │                   │                   │                       ║
║              ▼                   ▼                   ▼                       ║
║  ┌───────────┼───────────────────┼───────────────────┼────────────────────┐  ║
║  │           │                   │                   │                    │  ║
║  │    ┌──────┴───────┐    ┌──────┴───────┐    ┌──────┴───────┐           │  ║
║  │    │   CAJA A'    │◄──►│   CAJA B'    │◄──►│   CAJA C'    │           │  ║
║  │    │  (cálculo)   │    │  (cálculo)   │    │  (cálculo)   │           │  ║
║  │    │              │    │              │    │              │           │  ║
║  │    │  suma(A,B)   │    │  suma(B,C)   │    │  suma(C,A)   │           │  ║
║  │    │  mult(A,B)   │───►│  mult(B,C)   │───►│  mult(C,A)   │           │  ║
║  │    │  diff(A,B)   │    │  +calc_A'    │    │  +calc_B'    │           │  ║
║  │    │              │    │              │    │              │           │  ║
║  │    └──────────────┘    └──────────────┘    └──────────────┘           │  ║
║  │                                                                        │  ║
║  │                         CAPA DE CÁLCULOS                               │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║                                    │                                         ║
║                                    ▼                                         ║
║                          ┌─────────────────┐                                 ║
║                          │     FUSIÓN      │                                 ║
║                          │  datos+cálculos │                                 ║
║                          └────────┬────────┘                                 ║
║                                   │                                          ║
║                                   ▼                                          ║
║                          ┌─────────────────┐                                 ║
║                          │     SALIDA      │                                 ║
║                          │   10 clases     │                                 ║
║                          └─────────────────┘                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def crear_diagrama_niveles_fractales():
    """Diagrama de los 8 niveles fractales"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         8 NIVELES FRACTALES                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NIVEL 0 (dim=256)                                                           ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │████████████████████████████████████████████████████████████████████████│  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 1 (dim=128)                                                           ║
║  ┌────────────────────────────────────┐                                      ║
║  │████████████████████████████████████│                                      ║
║  └────────────────────────────────────┘                                      ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 2 (dim=64)                                                            ║
║  ┌──────────────────┐                                                        ║
║  │██████████████████│                                                        ║
║  └──────────────────┘                                                        ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 3 (dim=32)                                                            ║
║  ┌─────────┐                                                                 ║
║  │█████████│                                                                 ║
║  └─────────┘                                                                 ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 4 (dim=16)                                                            ║
║  ┌────┐                                                                      ║
║  │████│                                                                      ║
║  └────┘                                                                      ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 5 (dim=8)                                                             ║
║  ┌──┐                                                                        ║
║  │██│                                                                        ║
║  └──┘                                                                        ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 6 (dim=4)                                                             ║
║  ┌┐                                                                          ║
║  ││                                                                          ║
║  └┘                                                                          ║
║                                      │                                       ║
║                                      ▼                                       ║
║  NIVEL 7 (dim=2) ← BINARIO                                                   ║
║  []  ← [0,1] o [1,0] o [0,0] o [1,1]                                        ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║                                                                              ║
║  COMPRESIÓN:  256 → 2 = 128x reducción por nivel                            ║
║  TOTAL:       8 niveles con skip connections bidireccionales                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def crear_diagrama_cache_binario():
    """Diagrama del cache RAM binario"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           CACHE RAM BINARIO                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  El nivel más profundo (dim=2) tiene solo 4 estados posibles:               ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     COMBINACIONES BINARIAS                              │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   Estado 0: [0, 0]    Estado 1: [0, 1]                                  │ ║
║  │   Estado 2: [1, 0]    Estado 3: [1, 1]                                  │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  Para cada estado, PRE-COMPUTAMOS en RAM:                                    ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │  OPERACIONES PRE-CALCULADAS (por estado)                                │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   • Suma:      sum([a, b])                                              │ ║
║  │   • Producto:  prod([a, b])                                             │ ║
║  │   • Diff:      |a - b|                                                  │ ║
║  │   • Media:     (a + b) / 2                                              │ ║
║  │   • Máximo:    max(a, b)                                                │ ║
║  │   • Mínimo:    min(a, b)                                                │ ║
║  │   • Cross:     a*b' + a'*b                                              │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  Matriz de INTERACCIONES (4×4×7):                                            ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │           │ Estado 0  │ Estado 1  │ Estado 2  │ Estado 3  │             │ ║
║  ├───────────┼───────────┼───────────┼───────────┼───────────┤             │ ║
║  │ Estado 0  │  [7 ops]  │  [7 ops]  │  [7 ops]  │  [7 ops]  │             │ ║
║  │ Estado 1  │  [7 ops]  │  [7 ops]  │  [7 ops]  │  [7 ops]  │             │ ║
║  │ Estado 2  │  [7 ops]  │  [7 ops]  │  [7 ops]  │  [7 ops]  │             │ ║
║  │ Estado 3  │  [7 ops]  │  [7 ops]  │  [7 ops]  │  [7 ops]  │             │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  MEMORIA TOTAL: 4×4×7×4 bytes = 448 bytes + 4×7×4 = 112 bytes               ║
║                 TOTAL: ~560 bytes (¡menos de 1KB!)                          ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║                                                                              ║
║  BENEFICIO:                                                                  ║
║                                                                              ║
║    SIN CACHE:  forward() → calcular operaciones → O(batch × ops)            ║
║    CON CACHE:  forward() → lookup en tabla     → O(1)                       ║
║                                                                              ║
║    SPEEDUP: ~10-100x en el nivel binario                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def crear_diagrama_6_cajas():
    """Diagrama detallado de las 6 cajas"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                             6 CAJAS TRINITY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ═══════════════════════════ CAPA DE DATOS ══════════════════════════════   ║
║                                                                              ║
║       CAJA A (datos)          CAJA B (datos)          CAJA C (datos)         ║
║    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       ║
║    │ ┌─────┬─────┐   │     │ ┌─────┬─────┐   │     │ ┌─────┬─────┐   │       ║
║    │ │ Q1  │ Q2  │   │     │ │ Q1  │ Q2  │   │     │ │ Q1  │ Q2  │   │       ║
║    │ │256→2│256→2│   │◄───►│ │256→2│256→2│   │◄───►│ │256→2│256→2│   │       ║
║    │ ├─────┼─────┤   │     │ ├─────┼─────┤   │     │ ├─────┼─────┤   │       ║
║    │ │ Q3  │ Q4  │   │     │ │ Q3  │ Q4  │   │     │ │ Q3  │ Q4  │   │       ║
║    │ │256→2│256→2│   │     │ │256→2│256→2│   │     │ │256→2│256→2│   │       ║
║    │ └─────┴─────┘   │     │ └─────┴─────┘   │     │ └─────┴─────┘   │       ║
║    │   RELACIONES    │     │   RELACIONES    │     │   RELACIONES    │       ║
║    │  (h, v, diag)   │     │  (h, v, diag)   │     │  (h, v, diag)   │       ║
║    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       ║
║             │                       │                       │                ║
║             ▼                       ▼                       ▼                ║
║         ┌───────┐               ┌───────┐               ┌───────┐            ║
║         │LLAVE  │               │LLAVE  │               │LLAVE  │            ║
║         │A ↔ A' │               │B ↔ B' │               │C ↔ C' │            ║
║         └───┬───┘               └───┬───┘               └───┬───┘            ║
║             │                       │                       │                ║
║             ▼                       ▼                       ▼                ║
║                                                                              ║
║  ═══════════════════════ CAPA DE CÁLCULOS ═══════════════════════════════   ║
║                                                                              ║
║      CAJA A' (calc)          CAJA B' (calc)          CAJA C' (calc)         ║
║    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       ║
║    │                 │     │                 │     │                 │       ║
║    │  OP(A, B)       │────►│  OP(B, C)       │────►│  OP(C, A)       │       ║
║    │                 │     │  + calc_A'      │     │  + calc_B'      │       ║
║    │  • suma(A,B)    │     │                 │     │                 │       ║
║    │  • mult(A,B)    │     │  • suma(B,C)    │     │  • suma(C,A)    │       ║
║    │  • diff(A,B)    │     │  • mult(B,C)    │     │  • mult(C,A)    │       ║
║    │                 │     │  • diff(B,C)    │     │  • diff(C,A)    │       ║
║    │  META-CÁLCULO   │     │  META-CÁLCULO   │     │  META-CÁLCULO   │       ║
║    │                 │     │                 │     │                 │       ║
║    └─────────────────┘◄───►└─────────────────┘◄───►└─────────────────┘       ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║                                                                              ║
║  CONEXIONES:                                                                 ║
║    • Intra-capa datos:    A ↔ B ↔ C ↔ A (ciclo)                             ║
║    • Intra-capa cálculos: A' ↔ B' ↔ C' ↔ A' (ciclo)                         ║
║    • Inter-capa:          A ↔ A', B ↔ B', C ↔ C' (bidireccional)            ║
║    • Cascada cálculos:    A' → B' → C' (acumulativo)                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def crear_diagrama_compresion():
    """Diagrama de compresión y factor"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          COMPRESIÓN ~920,000x                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  "COMO SI ENTRARA 1TB EN 1GB"                                               ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      PARÁMETROS vs RELACIONES                           │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   PARÁMETROS REALES:           ~3.3 Millones                            │ ║
║  │   PARÁMETROS SIN COMPARTIR:    ~13 Millones                             │ ║
║  │   RELACIONES REPRESENTADAS:    ~3.2 BILLONES                            │ ║
║  │                                                                         │ ║
║  │   Factor pesos compartidos:    4x                                       │ ║
║  │   Factor relaciones:           920,000x                                 │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                         TAMAÑO EN DISCO                                 │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   Modelo LLARRI-O1 v4.0:       ~12.5 MB                                 │ ║
║  │                                                                         │ ║
║  │   Si guardara todas las                                                 │ ║
║  │   relaciones explícitamente:   ~11.5 TB                                 │ ║
║  │                                                                         │ ║
║  │   COMPRESIÓN:  11,500 GB → 0.0125 GB = 920,000x                        │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     COMPARACIÓN CON OTROS MODELOS                       │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   Modelo          │ Parámetros │ Relaciones  │ Factor                   │ ║
║  │   ────────────────┼────────────┼─────────────┼───────                   │ ║
║  │   GPT-2 Small     │    117M    │    ~117M    │   1x                     │ ║
║  │   BERT-Base       │    110M    │    ~110M    │   1x                     │ ║
║  │   GPT-3           │    175B    │    ~175B    │   1x                     │ ║
║  │   LLARRI-O1 v4.0  │    3.3M    │    ~3.2B    │ 920,000x                 │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║                                                                              ║
║  ¿CÓMO ES POSIBLE?                                                          ║
║                                                                              ║
║    1. Pesos compartidos: Un cuadrante sirve para todos los niveles          ║
║    2. Relaciones implícitas: Las conexiones generan combinaciones           ║
║    3. Fractales: Cada nivel amplifica las relaciones del anterior           ║
║    4. Cache binario: Las operaciones básicas no ocupan parámetros           ║
║    5. Composición: relaciones × relaciones = relaciones²                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def crear_diagrama_flujo():
    """Diagrama de flujo de datos"""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           FLUJO DE INFORMACIÓN                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                              ┌─────────────┐                                 ║
║                              │   ENTRADA   │                                 ║
║                              │   784 dims  │                                 ║
║                              │   (28×28)   │                                 ║
║                              └──────┬──────┘                                 ║
║                                     │                                        ║
║                    ┌────────────────┼────────────────┐                       ║
║                    │                │                │                       ║
║                    ▼                ▼                ▼                       ║
║              ┌──────────┐    ┌──────────┐    ┌──────────┐                    ║
║              │  CAJA A  │◄──►│  CAJA B  │◄──►│  CAJA C  │                    ║
║              │  (datos) │    │  (datos) │    │  (datos) │                    ║
║              └────┬─────┘    └────┬─────┘    └────┬─────┘                    ║
║                   │               │               │                          ║
║                   │    ┌──────────┼──────────┐    │                          ║
║                   │    │          │          │    │                          ║
║                   ▼    ▼          ▼          ▼    ▼                          ║
║              ┌──────────┐    ┌──────────┐    ┌──────────┐                    ║
║              │ CAJA A'  │───►│ CAJA B'  │───►│ CAJA C'  │                    ║
║              │ (cálculo)│    │ (cálculo)│    │ (cálculo)│                    ║
║              │          │    │ +calc_A' │    │ +calc_B' │                    ║
║              └────┬─────┘    └────┬─────┘    └────┬─────┘                    ║
║                   │               │               │                          ║
║                   │    REFINAN    │   DATOS      │                           ║
║                   ▼    (backward) ▼              ▼                           ║
║              ┌──────────┐    ┌──────────┐    ┌──────────┐                    ║
║              │  CAJA A  │    │  CAJA B  │    │  CAJA C  │                    ║
║              │(refinada)│    │(refinada)│    │(refinada)│                    ║
║              └────┬─────┘    └────┬─────┘    └────┬─────┘                    ║
║                   │               │               │                          ║
║                   └───────────────┼───────────────┘                          ║
║                                   │                                          ║
║                                   ▼                                          ║
║                          ┌────────────────┐                                  ║
║                          │     FUSIÓN     │                                  ║
║                          │ (A+B+C, A'+B'+C')                                │ ║
║                          │   datos+cálculos │                                ║
║                          └────────┬───────┘                                  ║
║                                   │                                          ║
║                                   ▼                                          ║
║                          ┌────────────────┐                                  ║
║                          │     OUTPUT     │                                  ║
║                          │   10 clases    │                                  ║
║                          │ [0-9] digits   │                                  ║
║                          └────────────────┘                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def guardar_diagramas():
    """Guarda todos los diagramas en archivos"""
    os.makedirs('diagrams', exist_ok=True)
    
    diagramas = {
        'arquitectura_v4.txt': crear_diagrama_arquitectura_v4(),
        '8_niveles_fractales.txt': crear_diagrama_niveles_fractales(),
        'cache_binario.txt': crear_diagrama_cache_binario(),
        '6_cajas.txt': crear_diagrama_6_cajas(),
        'compresion.txt': crear_diagrama_compresion(),
        'flujo.txt': crear_diagrama_flujo(),
    }
    
    for nombre, contenido in diagramas.items():
        path = os.path.join('diagrams', nombre)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contenido)
        print(f"✓ Guardado: {path}")
    
    # También guardar un archivo combinado
    with open('diagrams/todos_los_diagramas.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLARRI-O1 v4.0 - TODOS LOS DIAGRAMAS\n")
        f.write("=" * 80 + "\n\n")
        for nombre, contenido in diagramas.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"DIAGRAMA: {nombre}\n")
            f.write(f"{'='*80}\n")
            f.write(contenido)
            f.write("\n")
    
    print(f"\n✅ Todos los diagramas guardados en diagrams/")


if __name__ == "__main__":
    # Imprimir todos los diagramas
    print(crear_diagrama_arquitectura_v4())
    print(crear_diagrama_6_cajas())
    print(crear_diagrama_niveles_fractales())
    print(crear_diagrama_cache_binario())
    print(crear_diagrama_compresion())
    print(crear_diagrama_flujo())
    
    # Guardar en archivos
    guardar_diagramas()
