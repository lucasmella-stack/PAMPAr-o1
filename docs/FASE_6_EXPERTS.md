# 📚 FASE 6 - Fine-tuning de Expertos Especializados

## Estado: ✅ COMPLETADA

---

## 🎯 Objetivo

Crear modelos expertos especializados en dominios específicos mediante fine-tuning eficiente del modelo base, utilizando técnicas como **Adapter Layers**, **LoRA (Low-Rank Adaptation)** y **Fine-tuning Completo**.

---

## 📋 Componentes Implementados

### 1. **`src/llarri/models/expert_head.py`** - Módulos de Expert Heads

Implementa tres estrategias de especialización:

#### A. **AdapterLayer**
- Capas pequeñas insertadas en el modelo
- Arquitectura: down_project → activation → up_project + residual
- ~1% de parámetros entrenables
- Ideal para: Fine-tuning rápido con pocos datos

```python
class AdapterLayer(nn.Module):
    """Adapter según Houlsby et al. (2019)"""
    def __init__(self, hidden_size: int, adapter_size: int):
        # down_project: hidden_size → adapter_size
        # up_project: adapter_size → hidden_size
```

**Características:**
- ✅ Eficiente en parámetros (~64 dim adapter vs 768 hidden)
- ✅ Rápido de entrenar
- ✅ Múltiples adapters por modelo
- ✅ Inicialización cerca de identidad

#### B. **LoRALayer**
- Low-Rank Adaptation según Hu et al. (2021)
- Descompone W en W + BA (matrices de bajo rango)
- ~0.1-1% de parámetros entrenables
- Ideal para: Múltiples expertos, recursos limitados

```python
class LoRALayer(nn.Module):
    """LoRA con rank y alpha configurables"""
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        # Matrices A (in → rank) y B (rank → out)
        # Scaling: alpha / rank
```

**Características:**
- ✅ Muy eficiente (rank 8 = ~8x reducción)
- ✅ Múltiples LoRAs pueden combinarse
- ✅ No modifica arquitectura original
- ✅ Aplicable a attention layers

#### C. **ExpertDecoder**
- Decoder TrOCR completo especializado
- Clonado desde modelo base
- Estrategias de congelamiento configurables
- Ideal para: Máxima especialización con datos suficientes

```python
class ExpertDecoder(nn.Module):
    """Decoder completo con congelamiento parcial"""
    def __init__(
        self,
        base_decoder,
        freeze_embeddings=True,
        freeze_n_layers=0
    ):
```

**Características:**
- ✅ Máxima capacidad de adaptación
- ✅ Congelamiento flexible (embeddings, N capas)
- ✅ Fine-tuning completo o parcial
- ✅ Requiere más datos y tiempo

#### D. **ExpertHead (Wrapper)**
Factory class que unifica los tres tipos:

```python
expert = ExpertHead(
    base_decoder=decoder,
    expert_type="adapter",  # or "lora" or "full"
    adapter_size=64,  # para adapter
    rank=8, alpha=16.0  # para lora
)
```

---

### 2. **`src/llarri/training/finetune_expert.py`** - Script de Fine-tuning

Script completo para entrenar expertos especializados.

#### Clase `ExpertModel`

Lightning Module que combina:
- Encoder del modelo base (congelado)
- Expert head (entrenable)
- Tokenizer compartido

```python
class ExpertModel(pl.LightningModule):
    def __init__(
        self,
        base_model: LlarriBaseModel,
        expert_type: str = "adapter",
        expert_config: Optional[Dict] = None,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
    ):
```

**Características:**
- ✅ Solo entrena parámetros del expert head
- ✅ Encoder congelado por defecto
- ✅ Estadísticas de parámetros entrenables
- ✅ Generate() para inferencia
- ✅ Optimizer solo para expert parameters

#### CLI Completa

```bash
# Entrenar experto con configuración
python -m llarri.training.finetune_expert \
    --base-model outputs/experiment_xxx/checkpoints/final_model.ckpt \
    --config configs/expert_es_mayores.yaml

# Override de parámetros
python -m llarri.training.finetune_expert \
    --base-model model.ckpt \
    --config expert.yaml \
    --expert-type lora \
    --epochs 30
```

**Funcionalidades:**
- ✅ Carga modelo base desde checkpoint
- ✅ Crea expert head según configuración
- ✅ DataModule con datos específicos del experto
- ✅ Callbacks (ModelCheckpoint, EarlyStopping)
- ✅ Guarda checkpoint completo + solo expert head

---

### 3. **`configs/expert_es_mayores.yaml`** - Experto Español Mayores

Configuración para escritura de personas mayores.

**Características del dominio:**
- Letra temblorosa o irregular
- Tamaño variable
- Vocabulario formal/tradicional
- Acentuación correcta

**Configuración:**
```yaml
expert:
  name: "expert_es_mayores"
  specialization: "elderly_handwriting"
  language: "es"
  age_group: "60+"
  type: "adapter"
  
  params:
    adapter_size: 64
    num_adapter_layers: 6

training:
  epochs: 30
  batch_size: 8
  learning_rate: 0.0001
```

**Augmentations específicas:**
- Más rotación (15°)
- Más ruido gaussiano (0.08)
- Blur para simular temblor
- Reducción de contraste

---

### 4. **`configs/expert_latam_jovenes.yaml`** - Experto Latinoamérica Jóvenes

Configuración para escritura de jóvenes latinoamericanos.

**Características del dominio:**
- Letra fluida y rápida
- Abreviaturas y anglicismos
- Vocabulario informal
- Menos acentuación
- Influencia digital

**Configuración:**
```yaml
expert:
  name: "expert_latam_jovenes"
  specialization: "young_latam_handwriting"
  language: "es-419"
  age_group: "18-35"
  type: "lora"  # Más eficiente para este caso
  
  params:
    rank: 8
    alpha: 16.0
    target_modules: ["q_proj", "v_proj", "out_proj"]

training:
  epochs: 25
  batch_size: 12  # Mayor con LoRA
  learning_rate: 0.0002
```

**Vocabulario extendido:**
- "xq" (por qué)
- "tmb" (también)
- "q" (que)
- "ok", "wow", "jaja"

**Augmentations específicas:**
- Menos extremos (8° rotación)
- Motion blur (escritura rápida)
- Elastic transform

---

## 🔄 Flujo de Trabajo

### 1. Entrenar Modelo Base

```bash
# Primero, entrenar el modelo base
python -m llarri.training.train_base --config configs/training.yaml
```

### 2. Preparar Datos del Experto

```bash
# Crear splits específicos para el dominio
mkdir -p data/splits/es_mayores
# ... copiar/crear datos específicos
```

### 3. Fine-tune Experto

```bash
# Opción A: Experto con Adapter
python -m llarri.training.finetune_expert \
    --base-model outputs/experiment_xxx/checkpoints/final_model.ckpt \
    --config configs/expert_es_mayores.yaml

# Opción B: Experto con LoRA
python -m llarri.training.finetune_expert \
    --base-model outputs/experiment_xxx/checkpoints/final_model.ckpt \
    --config configs/expert_latam_jovenes.yaml \
    --expert-type lora
```

### 4. Usar Experto en Inferencia

```python
from llarri.training.finetune_expert import ExpertModel

# Cargar experto
expert = ExpertModel.load_from_checkpoint(
    "outputs/experts/expert_es_mayores/checkpoints/expert_es_mayores_final.ckpt"
)

# Inferencia
predictions = expert.generate(pixel_values, max_length=128, num_beams=4)
```

---

## 📊 Comparación de Estrategias

| Estrategia | Parámetros | Tiempo | Datos | Uso Recomendado |
|-----------|-----------|---------|-------|-----------------|
| **Adapter** | ~1% | Medio | Moderados | Balance general |
| **LoRA** | ~0.1-1% | Rápido | Pocos | Múltiples expertos, recursos limitados |
| **Full** | 100% | Lento | Muchos | Máxima especialización |

### Adapter
✅ **Pros:**
- Buen balance eficiencia/capacidad
- Rápido de entrenar
- Múltiples adapters por modelo

❌ **Contras:**
- Más parámetros que LoRA
- Requiere integración en forward pass

### LoRA
✅ **Pros:**
- Muy eficiente en parámetros
- Múltiples LoRAs pueden combinarse
- No modifica arquitectura
- Ideal para muchos expertos

❌ **Contras:**
- Capacidad limitada vs full
- Requiere implementación cuidadosa

### Full Fine-tuning
✅ **Pros:**
- Máxima capacidad de adaptación
- Mejor performance con datos suficientes
- Más flexible

❌ **Contras:**
- Requiere más datos
- Más tiempo de entrenamiento
- Mayor riesgo de overfitting

---

## 🎯 Casos de Uso

### Caso 1: OCR para Documentos Históricos
```yaml
expert:
  name: "expert_historical_docs"
  type: "adapter"
  specialization: "historical_manuscripts"
```

### Caso 2: OCR para Formularios Médicos
```yaml
expert:
  name: "expert_medical_forms"
  type: "lora"
  specialization: "medical_handwriting"
```

### Caso 3: OCR Multilenguaje
```yaml
expert:
  name: "expert_catalan"
  type: "full"
  specialization: "catalan_language"
```

---

## 📈 Métricas de Éxito

Después del fine-tuning, evaluar:

1. **CER (Character Error Rate)**: Errores a nivel carácter
2. **WER (Word Error Rate)**: Errores a nivel palabra
3. **Accuracy**: Precisión general
4. **Especialización**: Mejora en casos específicos del dominio

**Ejemplo de resultados esperados:**
```
Modelo Base:
  - CER general: 5.2%
  - CER dominio específico: 8.5%

Expert es_mayores:
  - CER general: 5.5% (ligera degradación)
  - CER dominio específico: 3.2% (mejora significativa)
```

---

## 🚀 Próximos Pasos

Con los expertos entrenados, puedes:

1. **Selector de Estilo (FASE 7)**: Clasificador que decide qué experto usar
2. **Ensemble**: Combinar predicciones de múltiples expertos
3. **Active Learning (FASE 8)**: Mejorar expertos con feedback
4. **API (FASE 9)**: Exponer expertos vía REST API

---

## 📝 Notas Importantes

### Guardado de Expertos

El script guarda **dos archivos**:

1. **Checkpoint completo** (`expert_xxx_final.ckpt`):
   - Incluye encoder + expert + optimizer
   - Usar para continuar entrenamiento
   - ~500MB-1GB

2. **Solo expert head** (`expert_xxx_head.pt`):
   - Solo parámetros del expert
   - Usar para inferencia con modelo base
   - ~10-50MB

### Combinación de Expertos

Los expertos LoRA pueden combinarse:

```python
# Cargar múltiples LoRAs
lora_es_mayores = torch.load("expert_es_mayores_head.pt")
lora_latam_jovenes = torch.load("expert_latam_jovenes_head.pt")

# Combinar (promedio ponderado, concatenación, etc.)
combined_lora = combine_loras([lora_es_mayores, lora_latam_jovenes], weights=[0.6, 0.4])
```

### Memory Requirements

| Tipo | Training | Inference |
|------|----------|-----------|
| Adapter | 8-12GB | 6-8GB |
| LoRA | 6-10GB | 6-8GB |
| Full | 16-24GB | 8-12GB |

---

## ✅ Checklist de Implementación

- [x] AdapterLayer implementado
- [x] LoRALayer implementado
- [x] ExpertDecoder implementado
- [x] ExpertHead wrapper
- [x] ExpertModel Lightning module
- [x] Script finetune_expert.py
- [x] CLI completa
- [x] Config expert_es_mayores.yaml
- [x] Config expert_latam_jovenes.yaml
- [x] Documentación completa
- [x] Ejemplos de uso

---

## 🎉 FASE 6 COMPLETADA

Todos los componentes para fine-tuning de expertos están implementados y listos para usar.

**Implementado:** 26 de noviembre de 2025  
**Versión:** 1.0.0
