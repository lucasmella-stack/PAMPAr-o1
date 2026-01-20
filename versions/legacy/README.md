# LLARRI-O1 Versiones Legacy

Este directorio contiene versiones anteriores del modelo que ya no están en uso activo.

## Estructura

```
legacy/
├── models/           # Modelos de lenguaje v1-v6
│   ├── language_model.py      # v1 original
│   ├── language_model_v3.py   # v3 con cajas fractales
│   ├── language_model_v4.py   # v4 con compresión
│   ├── language_model_v5.py   # v5 con mejoras
│   └── language_model_v6.py   # v6 pre-bloque neural
├── scripts/          # Scripts de entrenamiento legacy
│   ├── train.py
│   ├── train_language_model.py
│   ├── train_v3.py - train_v6.py
│   └── test_generation*.py
└── modules/          # Módulos no usados en versiones actuales
    ├── compositor.py / compositor_v2.py
    ├── reflexion.py
    ├── niveles.py
    └── multiescala.py
```

## Versiones Activas (en carpeta principal)

- **v6b**: Versión actual en entrenamiento con BloqueNeuralV6
- **v7**: Nueva arquitectura cerebral con módulos especializados

## Notas

- Estos archivos se conservan para referencia histórica
- No se deben modificar a menos que sea para investigación
- Los checkpoints correspondientes están en `checkpoints/`

---
*Organizado: 8 Enero 2026*
*Autor: Lucas Ricardo Mella Chillemi*
