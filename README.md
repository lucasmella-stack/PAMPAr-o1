# llarri-01

## Proyecto de Reconocimiento de Texto Manuscrito (OCR)

Sistema avanzado de OCR especializado en texto manuscrito utilizando arquitectura ViT (Vision Transformer) + TrOCR.

---

## 📋 Estructura del Proyecto

```
llarri-01/
├── configs/              # Archivos de configuración YAML
├── data/                 # Datasets y datos procesados
│   ├── external/        # Datos originales
│   ├── processed/       # Datos preprocesados
│   ├── splits/          # Train/val/test splits (JSONL)
│   └── user_samples/    # Muestras de usuario
├── src/llarri/          # Código fuente principal
│   ├── data/           # DataModules y datasets
│   ├── models/         # Arquitecturas de modelos
│   ├── training/       # Scripts de entrenamiento
│   ├── inference/      # Scripts de inferencia
│   ├── active_learning/ # Active learning
│   └── api/            # API REST
├── scripts/            # Scripts de utilidad
└── deploy/             # Docker y deployment
```

---

## 🚀 Instalación

### Requisitos previos
- Python 3.10+
- CUDA (opcional, para GPU)

### Instalación con Poetry

```bash
# Clonar repositorio
git clone <repo-url>
cd llarri-01

# Instalar dependencias
poetry install

# Activar entorno
poetry shell
```

### Instalación con pip

```bash
pip install -e .
```

---

## ✅ Estado de Implementación

### FASE 0-4: COMPLETADAS Y CORREGIDAS ✅

| Fase | Componente | Estado |
|------|-----------|--------|
| 0 | Estructura del proyecto | ✅ Completa |
| 1 | pyproject.toml | ✅ Completa |
| 2 | DataModule | ✅ Completa + Mejorada |
| 3 | Transforms con augmentations | ✅ **CORREGIDA** |
| 4.1 | Encoder ViT | ✅ Completa |
| 4.2 | Decoder TrOCR | ✅ Completa |
| 4.3 | LlarriBaseModel | ✅ **CORREGIDA** |

### ✨ Mejoras Implementadas

#### 1. **transforms.py** - Augmentations Agresivos
- ✅ Conversión a escala de grises (manteniendo 3 canales para ViT)
- ✅ Resize configurable (default: 128x512)
- ✅ Rotación aleatoria ±10°
- ✅ Affine aleatorio (shear, translate, scale)
- ✅ Ruido gaussiano aleatorio
- ✅ Cambios de brillo/contraste
- ✅ Normalización ImageNet

#### 2. **llarri_base_model.py** - Correcciones Críticas
- ✅ Integración de `TrOCRProcessor` y tokenizer de HuggingFace
- ✅ Corrección de `training_step` y `validation_step` para diccionarios
- ✅ Soporte para `pixel_values` y `labels` correctos
- ✅ Método `generate()` con beam search
- ✅ Optimizer AdamW con scheduler ReduceLROnPlateau
- ✅ Logging mejorado con métricas

#### 3. **datamodule_base.py** - Collate Function
- ✅ Función `collate_fn_with_tokenizer()` personalizada
- ✅ Tokenización automática de textos
- ✅ Padding y truncation correctos
- ✅ Máscaras de atención para labels (-100 para padding)
- ✅ Integración completa con DataLoaders

---

## 📖 Uso Básico

### 1. Preparar Datos

Los datos deben estar en formato JSONL:

```json
{"id": "sample_001", "image_path": "lines/sample_001.png", "text": "Texto manuscrito"}
{"id": "sample_002", "image_path": "lines/sample_002.png", "text": "Otro ejemplo"}
```

Guardar en:
- `data/splits/train.jsonl`
- `data/splits/val.jsonl`

### 2. Entrenar Modelo

```python
from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.data.datamodule_base import LlarriDataModule
import pytorch_lightning as pl

# Inicializar modelo
model = LlarriBaseModel(learning_rate=5e-5)

# Configurar DataModule
datamodule = LlarriDataModule(
    train_path="data/splits/train.jsonl",
    val_path="data/splits/val.jsonl",
    data_root="data/processed",
    batch_size=8,
    tokenizer=model.tokenizer,  # Importante: pasar tokenizer
    img_height=128,
    img_width=512,
)

# Entrenar
trainer = pl.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, datamodule)
```

### 3. Inferencia

```python
from PIL import Image
from llarri.data.transforms import val_transforms

# Cargar modelo entrenado
model = LlarriBaseModel.load_from_checkpoint("checkpoints/model.ckpt")
model.eval()

# Cargar y procesar imagen
transform = val_transforms(img_height=128, img_width=512)
image = Image.open("sample.png").convert("RGB")
pixel_values = transform(image).unsqueeze(0)

# Predecir
predictions = model.generate(pixel_values, max_length=128, num_beams=4)
print(predictions[0])
```

Ver `scripts/example_usage.py` para ejemplos completos.

---

## 🔧 Configuración

### Augmentations
Personalizar en `src/llarri/data/transforms.py`:

```python
transforms = get_transforms(
    stage="train",
    img_height=128,  # Altura de imagen
    img_width=512,   # Ancho de imagen
    normalize=True   # Normalización ImageNet
)
```

### Modelo
Configurar en el código:

```python
from llarri.models.encoder_vit import ViTEncoderConfig
from llarri.models.decoder_trocr import TrOCRDecoderConfig

encoder_config = ViTEncoderConfig(
    pretrained_model_name="google/vit-base-patch16-224-in21k",
    freeze=False
)

decoder_config = TrOCRDecoderConfig(
    pretrained_model_name="microsoft/trocr-base-handwritten",
    freeze=False
)

model = LlarriBaseModel(
    encoder_cfg=encoder_config,
    decoder_cfg=decoder_config,
    learning_rate=5e-5
)
```

---

## 🎯 Próximas Fases

Ahora estás listo para continuar con:
- **FASE 5**: Scripts de entrenamiento base (`train_base.py`)
- **FASE 6**: Fine-tuning de expertos (`finetune_expert.py`)
- **FASE 7**: Selector de estilo (`train_selector.py`)
- **FASE 8**: Active learning loop
- **FASE 9**: API REST y deployment

---

## 📝 Notas Importantes

### Errores de Importación en IDE
Los errores de importación mostrados por el IDE son normales si las dependencias no están instaladas. Para resolverlos:

```bash
# Instalar dependencias
poetry install

# O con pip
pip install torch torchvision transformers pytorch-lightning
```

### Arquitectura Corregida

El modelo ahora funciona correctamente:
1. **ViT Encoder** procesa imágenes → embeddings
2. **TrOCR Decoder** decodifica embeddings → texto
3. **Tokenizer** de HuggingFace maneja vocabulario
4. **DataModule** prepara batches con `collate_fn`
5. **Lightning** orquesta entrenamiento

---

## 📚 Recursos

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [HuggingFace TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)

---

## 🤝 Contribuciones

Este es un proyecto en desarrollo activo. Las fases 0-4 están completadas y corregidas.

---

## 📄 Licencia

Pendiente de definir.
