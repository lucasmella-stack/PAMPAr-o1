---
language:
- es
license: agpl-3.0
library_name: pytorch
tags:
- pytorch
- mnist
- investigacion
- experimental
---

# LLARRI-O1 — Model Card

Versión en inglés: [MODEL_CARD.md](MODEL_CARD.md)

## Detalles del modelo
- Nombre: LLARRI-O1 HyperComprimido (v4.0)
- Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
- Coordinador: Alvaro (Segunda Cabeza)
- Licencia: AGPL-3.0-or-later

## Uso previsto
Investigación y experimentación sobre cómputo fractal y compartición de parámetros.

## Limitaciones
- El entrenamiento puede ser intensivo en memoria según `hidden_dim`, batch size y optimizador.
- Repositorio experimental; interfaces y resultados pueden cambiar.

## Datos de entrenamiento
- MNIST (vía torchvision) para los experimentos actuales.

## Uso Rápido
```python
from llarri_o1 import LlarriO1, Config
import torch

model = LlarriO1()  # Auto-detecta niveles fractales óptimos
x = torch.randn(2, 784)
output = model(x)  # Shape: (2, 10)
```

Entrenamiento:
```bash
python scripts/train.py --epochs 10 --batch-size 32
```

## Evaluación
- v3.1 Cuadrantes: 98.61% en MNIST (resultado histórico).
- v4.0 HyperComprimido: smoke tests + corridas de desarrollo; ver README.

## Riesgos
- Uso como modelo productivo sin evaluación adecuada.
- La licencia (AGPL) exige compartir el código modificado si se ofrece como servicio (red).

## Licencia
Este modelo/código se publica bajo **AGPL-3.0-or-later**.

## Citación
- Ver [CITATION.cff](../../CITATION.cff)

## Contacto
- lucas@segundacabeza.com
