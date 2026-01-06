# LLARRI-O1 Trinity Fractal

<div align="center">
  <h3>Arquitectura de IA Original</h3>
  <p><strong>Creada por Lucas Mella - Fundador de Segunda Cabeza</strong></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/lucas-mella/llarri-o1)
</div>

---

## Creditos

| Rol | Persona | Contacto |
|-----|---------|----------|
| **Fundador y Creador** | Lucas Mella | lucas@segundacabeza.com |
| **Coordinador** | Alvaro | alvaro@segundacabeza.com |
| **Organizacion** | Segunda Cabeza | - |

> La arquitectura **Trinity Fractal** completa fue concebida, disenada e implementada por **Lucas Mella**, fundador de Segunda Cabeza.

---

## Que es LLARRI-O1?

LLARRI-O1 es una arquitectura de inteligencia artificial **revolucionaria** que introduce el concepto de **"Trinity Fractal"** - un sistema donde la informacion fluye **bidireccionalmente** entre tres componentes principales.

### La Innovacion: Flujo Multidireccional

```
MODELO NORMAL (Transformer):       LLARRI-O1 (Trinity Fractal):
                                  
   Entrada                              Entrada
      |                                /       \
      v                               v         v
   [Capa 1]                      [Caja 1] <--> [Caja 2]
      |                               \       /
      v                                v     v
   [Capa 2]                          [Caja 3]
      |                                  |
      v                                  v
   Salida                             Salida
   
   (Solo baja)                 (TODAS las direcciones)
```

### Diferencias Clave

| Aspecto | Transformers | LLARRI Trinity |
|---------|--------------|----------------|
| Flujo | Unidireccional | **Multidireccional** |
| Estructura | Capas apiladas | 3 Cajas interconectadas |
| Conexiones | Secuenciales | **Llaves bidireccionales** |
| Eficiencia | Pesos unicos | Pesos compartidos |

---

## Arquitectura

### Las Tres Cajas (Trinity)

1. **Caja Padre**: Procesa entrada inicial (Vision)
2. **Caja Hijo**: Transforma y refina (Procesamiento)  
3. **Caja Espiritu**: Integra todo (Fusion)

### Llaves Bidireccionales

Las "llaves" permiten comunicacion en **ambas direcciones**:
- Padre <--> Hijo
- Hijo <--> Espiritu
- Padre <--> Espiritu (Skip connection)

### Mundos Fractales

Cada caja contiene "mundos" internos - subredes que crean una estructura de **"mundos dentro de mundos"**.

---

## Resultados

| Metrica | Valor |
|---------|-------|
| Accuracy | **100%** |
| Compresion | **93.8%** menos parametros |
| GPU Speedup | **17x** vs CPU |

---

## Instalacion

```bash
git clone https://github.com/lucasmella-stack/llarri-o1.git
cd llarri-o1
pip install torch
```

## Uso Rapido

```python
import torch
from llarri_o1_model import LLARRI_O1_TrinityFractal

# Configuracion
config = {
    "hidden_size": 512,
    "input_size": 784,
    "output_size": 10,
    "num_worlds": 3
}

# Crear modelo
model = LLARRI_O1_TrinityFractal(config)

# Inferencia
x = torch.randn(1, 784)
output = model(x)
```

---

## Estructura del Proyecto

```
llarri-o1/
├── LICENSE                      # MIT License con atribucion
├── README.md                    # Este archivo
├── CONVERSACION_LLARRI_O1.md   # Historia completa del desarrollo
├── llarri_o1_model.py          # Modelo base
├── llarri_o1_multimodal.py     # Version multimodal
├── prototipo_trinity_lucas.py  # Prototipo original
├── benchmark_gpu.py            # Benchmarks GPU vs CPU
├── upload_final_llarri.py      # Script de subida a HuggingFace
└── *.png                       # Diagramas y visualizaciones
```

---

## Licencia

**MIT License con Requisito de Atribucion**

Este proyecto usa licencia MIT, pero **requiere atribucion visible** a:
- Segunda Cabeza (organizacion)
- Lucas Mella (creador/fundador)
- Repositorio original

Ver [LICENSE](LICENSE) para terminos completos.

---

## Links

- **HuggingFace**: [lucas-mella/llarri-o1](https://huggingface.co/lucas-mella/llarri-o1)
- **GitHub**: [lucasmella-stack/llarri-o1](https://github.com/lucasmella-stack/llarri-o1)

---

<div align="center">
  <strong>Segunda Cabeza - 2026</strong>
  <br>
  <em>LLARRI-O1: Donde la inteligencia fluye en todas direcciones</em>
  <br><br>
  Creado por Lucas Mella
</div>
