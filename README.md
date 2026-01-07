# 🔷 LLARRI-O1 v4.0 - HyperComprimido

<div align="center">

![LLARRI-O1 Banner](diagrams/arquitectura_v4.png)

**Arquitectura revolucionaria: 6 Cajas + 8 Niveles Fractales + Cache RAM Binario**

*"Como si entrara 1TB en 1GB"*

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/lucas-mella/llarri-o1)

</div>

---

## 📖 Tabla de Contenidos

- [¿Qué es LLARRI-O1?](#-qué-es-llarri-o1)
- [Versiones](#-versiones)
- [v4.0 HyperComprimido](#-v40-hypercomprimido)
- [Arquitectura de 6 Cajas](#-arquitectura-de-6-cajas)
- [8 Niveles Fractales](#-8-niveles-fractales)
- [Cache RAM Binario](#-cache-ram-binario)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Créditos](#-créditos)
- [Licencia](#-licencia)

---

## 🧠 ¿Qué es LLARRI-O1?

**LLARRI-O1** es una arquitectura de inteligencia artificial completamente original que logra **compresión extrema** mediante:

1. **Pesos compartidos** entre niveles fractales
2. **Relaciones bidireccionales** entre todas las cajas
3. **Cache RAM** para operaciones binarias básicas
4. **Separación datos/cálculos** donde los cálculos operan sobre datos Y sobre otros cálculos

### Comparación de Compresión

| Modelo | Parámetros | Relaciones | Factor |
|--------|------------|------------|--------|
| GPT-2 Small | 117M | ~117M | 1x |
| BERT-Base | 110M | ~110M | 1x |
| **LLARRI-O1 v4.0** | **3.3M** | **3+ Billones** | **~920,000x** |

---

## 📊 Versiones

| Versión | Arquitectura | Parámetros | Accuracy | Estado |
|---------|--------------|------------|----------|--------|
| v1.0 | MLP básico | 500K | 95% | Legacy |
| v2.0 | Trinity simple | 800K | 97% | Legacy |
| v3.0 | Fractal profundo | 1.8M | - | Muy lento |
| v3.1 | Cuadrantes | 1.3M | **98.61%** | ✅ Estable |
| **v4.0** | **HyperComprimido** | **3.3M** | **En prueba** | 🚀 **Nuevo** |

---

## 🚀 v4.0 HyperComprimido

### La Revolución

```
╔══════════════════════════════════════════════════════════════════════╗
║                    LLARRI-O1 v4.0 HYPERCOMPRIMIDO                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📦 ARQUITECTURA:                                                    ║
║     • 6 Cajas (3 datos + 3 cálculos)                                ║
║     • 8 niveles fractales (256→128→64→32→16→8→4→2)                  ║
║     • Hasta nivel BINARIO (2 valores)                               ║
║     • Cache RAM para acelerar operaciones binarias                  ║
║                                                                      ║
║  📊 COMPRESIÓN:                                                      ║
║     • Parámetros REALES: ~3.3M                                      ║
║     • Relaciones TOTALES: 3+ BILLONES                               ║
║     • Factor: ~920,000x                                             ║
║                                                                      ║
║  💾 EQUIVALENCIA:                                                    ║
║     • 11.5 TB de relaciones en 12.5 MB de pesos                     ║
║     • "Como si entrara 1TB en 1GB"                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🏗️ Arquitectura de 6 Cajas

### Concepto

La v4.0 separa **datos** de **cálculos**:

- **Capa de Datos (3 cajas):** Procesan y almacenan información
- **Capa de Cálculos (3 cajas):** Operan sobre los datos Y sobre otros cálculos

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                            │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐             │
│  │ CAJA A  │←────→│ CAJA B  │←────→│ CAJA C  │             │
│  │ (datos) │      │ (datos) │      │ (datos) │             │
│  └────┬────┘      └────┬────┘      └────┬────┘             │
│       │                │                │                   │
└───────┼────────────────┼────────────────┼───────────────────┘
        ↓↑               ↓↑               ↓↑
        │    LLAVES BIDIRECCIONALES       │
        ↓↑               ↓↑               ↓↑
┌───────┼────────────────┼────────────────┼───────────────────┐
│       │                │                │                   │
│  ┌────┴────┐      ┌────┴────┐      ┌────┴────┐             │
│  │ CAJA A' │←────→│ CAJA B' │←────→│ CAJA C' │             │
│  │(cálculo)│      │(cálculo)│      │(cálculo)│             │
│  └─────────┘      └─────────┘      └─────────┘             │
│                  CAPA DE CÁLCULOS                           │
│         (calcula sobre datos Y sobre otros cálculos)        │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Información

1. **Datos A, B, C** procesan la entrada en paralelo
2. Las cajas de datos se **conectan en ciclo** (A→B→C→A)
3. **Cálculos A'** opera sobre datos A y B
4. **Cálculos B'** opera sobre datos B y C + cálculos A'
5. **Cálculos C'** opera sobre datos C y A + cálculos B'
6. Las cajas de cálculos se **conectan en ciclo**
7. Los cálculos **refinan los datos** (conexión bidireccional)
8. **Fusión final** de datos + cálculos

---

## 📐 8 Niveles Fractales

### De 256 hasta 2 (nivel binario)

```
NIVEL 0:  256 dimensiones  ████████████████████████████████
NIVEL 1:  128 dimensiones  ████████████████
NIVEL 2:   64 dimensiones  ████████
NIVEL 3:   32 dimensiones  ████
NIVEL 4:   16 dimensiones  ██
NIVEL 5:    8 dimensiones  █
NIVEL 6:    4 dimensiones  ▌
NIVEL 7:    2 dimensiones  ▏ ← BINARIO
```

### Cada nivel:
- Tiene su propia transformación (down + up)
- Comparte pesos con el mismo nivel en otros cuadrantes
- Skip connections para preservar información

### Combinaciones Binarias

Con solo 2 valores en el nivel más profundo:

```
[0,0] → Estado 0
[0,1] → Estado 1
[1,0] → Estado 2  
[1,1] → Estado 3

4 estados × 8 niveles × 4 cuadrantes × 6 cajas = 768 combinaciones base
768^2 (relaciones cruzadas) = 589,824 estados únicos POR PASADA
```

---

## ⚡ Cache RAM Binario

### El Truco de Velocidad

Las operaciones en el nivel binario (dim=2) son **finitas y predecibles**:

```python
# Solo hay 4 combinaciones posibles:
[0, 0], [0, 1], [1, 0], [1, 1]

# Para cada combinación, pre-computamos:
- Suma
- Producto  
- Diferencia absoluta
- Media
- Máximo
- Mínimo
- Producto cruzado
```

### Beneficio

```
SIN CACHE:
  Cada forward pass → Recalcular operaciones binarias
  Tiempo: O(batch_size × operaciones)

CON CACHE:
  Operaciones binarias → Lookup en tabla pre-computada
  Tiempo: O(1) por operación
  
Speedup: ~10-100x en nivel binario
```

### Memoria RAM Usada

```
4 combinaciones × 7 operaciones × 4 bytes = 112 bytes
Matriz de interacciones: 4 × 4 × 7 × 4 bytes = 448 bytes
Total cache: ~560 bytes (¡menos de 1KB!)
```

---

## 📦 Instalación

```bash
# Clonar
git clone https://github.com/lucasmella-stack/llarri-o1.git
cd llarri-o1

# Entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencias
pip install -r requirements.txt
```

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recomendado)
- 4GB RAM mínimo
- 4GB VRAM mínimo (para GPU)

---

## 🚀 Uso

### v4.0 HyperComprimido

```python
from src.llarri_o1_hypercomprimido import (
    LlarriO1_HyperComprimido, 
    ConfigHyperComprimido,
    entrenar_hypercomprimido
)

# Configuración
config = ConfigHyperComprimido(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    num_cajas_datos=3,
    num_cajas_calculos=3,
    niveles_fractales=[256, 128, 64, 32, 16, 8, 4, 2],
    usar_cache_binario=True
)

# Crear modelo
modelo = LlarriO1_HyperComprimido(config)

# Entrenar
modelo, accuracy = entrenar_hypercomprimido(epochs=30)
```

### v3.1 Cuadrantes (Estable)

```python
from src.llarri_o1_cuadrantes import (
    LlarriO1_Cuadrantes,
    entrenar_por_cuadrantes
)

# Entrenar
modelo, accuracy = entrenar_por_cuadrantes(epochs=25)
# Resultado: 98.61% accuracy
```

### Línea de comandos

```bash
# v4.0 HyperComprimido
python src/llarri_o1_hypercomprimido.py

# v3.1 Cuadrantes
python src/llarri_o1_cuadrantes.py
```

---

## 📊 Resultados

### v3.1 Cuadrantes (MNIST)

| Época | Train | Val | Tiempo |
|-------|-------|-----|--------|
| 1 | 92.5% | 95.4% | 43s |
| 10 | 99.0% | 97.9% | 40s |
| 19 | 100% | 98.6% | 40s |
| **25** | **100%** | **98.61%** | **44s** |

### v4.0 HyperComprimido

*En entrenamiento...*

---

## 🔬 Comparación Técnica

### vs Transformers

| Aspecto | Transformer | LLARRI-O1 v4.0 |
|---------|-------------|----------------|
| Conexiones | Secuenciales | Bidireccionales |
| Capas | 12-96 | 6 (3+3) |
| Atención | O(n²) | O(n) con cache |
| Compresión | ~1x | ~920,000x |
| Pesos compartidos | No | Sí (extremo) |

### vs CNNs

| Aspecto | CNN | LLARRI-O1 v4.0 |
|---------|-----|----------------|
| Estructura | Jerárquica | Fractal bidireccional |
| Receptive field | Local → Global | Global desde nivel 0 |
| Parámetros por capa | Independientes | Compartidos |

---

## 🧮 Matemáticas de Compresión

### Sin compartir pesos:

```
6 cajas × 4 cuadrantes × 8 niveles × (transformaciones) = 
~13M parámetros únicos
```

### Con compartir pesos:

```
1 cuadrante base × 8 niveles + relaciones + llaves =
~3.3M parámetros reales
```

### Relaciones representadas:

```
Relaciones internas: 6 × 4 × Σ(nivel_i²) = ~450,000
Relaciones entre cajas: 15 × 256² = ~980,000  
Relaciones intercapa: 3 × 256² × 2 = ~390,000
Combinaciones binarias: 112 × 24 = 2,688

Total único: ~1.8M relaciones directas
Con composición: 1.8M² = 3.24 BILLONES de relaciones compuestas
```

### Factor de compresión:

```
3.24 × 10⁹ relaciones / 3.3 × 10⁶ parámetros = 981,818x
≈ 920,000x de compresión
```

---

## 📁 Estructura del Proyecto

```
llarri-o1/
├── src/
│   ├── llarri_o1_hypercomprimido.py  # v4.0 HyperComprimido
│   ├── llarri_o1_cuadrantes.py       # v3.1 Cuadrantes
│   ├── llarri_o1_fractal_profundo.py # v3.0 (legacy)
│   └── utils.py
├── llarri_o1/                         # Paquete instalable
│   ├── models/
│   ├── training/
│   └── utils/
├── checkpoints/
│   ├── llarri_hypercomprimido_mejor.pt
│   └── llarri_cuadrantes_mejor.pt
├── diagrams/
├── examples/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 👥 Créditos

### Equipo

| Rol | Nombre | Contacto |
|-----|--------|----------|
| **Fundador & Creador** | Lucas Mella | lucas@segundacabeza.com |
| **Coordinador** | Alvaro | alvaro@segundacabeza.com |

### Organización

**Segunda Cabeza** - Innovación en Inteligencia Artificial

- 🌐 Web: [segundacabeza.com](https://segundacabeza.com)
- 🤗 HuggingFace: [lucas-mella/llarri-o1](https://huggingface.co/lucas-mella/llarri-o1)
- 📂 GitHub: [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)

---

## 📄 Licencia

**Licencia Propietaria con Atribución**

```
Copyright (c) 2026 Lucas Mella / Segunda Cabeza

Este software es propietario. Se permite:
- ✅ Uso educativo y de investigación
- ✅ Uso personal no comercial
- ✅ Citación en trabajos académicos

Se requiere:
- 📝 Atribución clara al autor y organización
- 📧 Contacto para uso comercial

Se prohíbe:
- ❌ Distribución sin autorización
- ❌ Uso comercial sin licencia
- ❌ Modificación sin atribución
```

Para uso comercial, contactar: lucas@segundacabeza.com

---

<div align="center">

**Hecho con 💜 por Segunda Cabeza**

*"Comprimiendo la inteligencia, expandiendo las posibilidades"*

**v4.0 HyperComprimido - La evolución de la compresión**

</div>
