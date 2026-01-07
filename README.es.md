# LLARRI-O1 (EN below) — Arquitectura fractal de compresión extrema

English: see [README.md](README.md)

## Resumen
LLARRI-O1 es una arquitectura experimental de redes neuronales basada en **procesamiento por cuadrantes** + **niveles fractales** + **conexiones entre cajas** para lograr alta expresividad con reutilización de parámetros.

Este repositorio contiene múltiples versiones; la versión actual en desarrollo es **v4.0 HyperComprimido** (6 cajas, 8 niveles fractales, cache binario y auto-cálculos internos).

## Estado del proyecto
- v3.1 Cuadrantes (MNIST): 98.61% (histórico)
- v4.0 HyperComprimido (MNIST): smoke-tests OK; entrenamiento “grande” requiere ajustes de memoria/VRAM.

## Instalación
```bash
pip install -r requirements.txt
```

## Uso Rápido

Ejemplo básico:
```bash
python examples/basic_usage.py
```

O en Python:
```python
from llarri_o1 import LlarriO1, Config
import torch

model = LlarriO1()  # Config auto-detecta niveles óptimos
x = torch.randn(2, 784)  # Entrada MNIST
output = model(x)  # Shape: (2, 10)
```

Entrenar v4.0:
```bash
python scripts/train.py --epochs 10 --batch-size 32
```

Generar diagramas:
```bash
python -m llarri_o1.visualization.diagrams
```

## Diagramas
- v4.0 (actual): [diagrams/v4-current](diagrams/v4-current)
- versiones anteriores: [diagrams/v3-legacy](diagrams/v3-legacy)

## Atribución / Citación
**Atribución preferida (texto corto):**
> “LLARRI-O1 — Lucas Ricardo Mella Chillemi (Segunda Cabeza).”

**Cómo citar:**
- Ver [CITATION.cff](CITATION.cff)
- Nombre de arquitectura recomendado: **LLARRI-O1 HyperComprimido (v4.0)**

## Licencia (AGPL)
Este proyecto se publica bajo **GNU AGPLv3 o posterior (AGPL-3.0-or-later)**.

### ¿Por qué AGPL?
- Permite uso comercial.
- Requiere que si ofrecés el software como servicio (network use), también compartas el código fuente modificado.

## Contacto
- Lucas Ricardo Mella Chillemi — lucas@segundacabeza.com
- Coordinación: Alvaro — alvaro@segundacabeza.com
