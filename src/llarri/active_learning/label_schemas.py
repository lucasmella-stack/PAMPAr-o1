"""
label_schemas.py - Esquemas de Etiquetado para Active Learning

Define esquemas Pydantic para validación de datos de etiquetado:
- LabelRequest: Solicitud de etiquetado de muestra
- LabelResponse: Respuesta con etiqueta asignada
- StyleLabelRequest/Response: Para etiquetado de estilos
- OCRLabelRequest/Response: Para etiquetado de texto OCR
- MultiTaskLabelResponse: Para etiquetado multi-atributo

Estos esquemas se usan en:
- API de etiquetado manual
- Validación de datos etiquetados
- Interfaz de etiquetado semi-automático

Uso:
    # Crear request
    request = LabelRequest(
        image_id="img_001",
        image_path="data/samples/img_001.jpg",
        uncertainty_score=0.85
    )
    
    # Validar response
    response = LabelResponse(
        image_id="img_001",
        text="Hola mundo",
        confidence=0.95
    )
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


# =============================================================================
# ETIQUETADO BÁSICO
# =============================================================================

class LabelRequest(BaseModel):
    """
    Solicitud para etiquetar una muestra.
    """
    image_id: str = Field(..., description="ID único de la imagen")
    image_path: str = Field(..., description="Path a la imagen")
    uncertainty_score: Optional[float] = Field(None, description="Score de incertidumbre")
    iteration: Optional[int] = Field(None, description="Iteración de active learning")
    
    @validator('uncertainty_score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("uncertainty_score debe estar en [0, 1]")
        return v


class LabelResponse(BaseModel):
    """
    Respuesta con etiqueta asignada.
    """
    image_id: str = Field(..., description="ID único de la imagen")
    text: str = Field(..., description="Texto transcrito")
    confidence: Optional[float] = Field(None, description="Confianza del etiquetador")
    labeler_id: Optional[str] = Field(None, description="ID del etiquetador")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    notes: Optional[str] = Field(None, description="Notas adicionales")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("confidence debe estar en [0, 1]")
        return v


# =============================================================================
# ETIQUETADO DE ESTILOS (PARA SELECTOR)
# =============================================================================

class StyleLabelRequest(BaseModel):
    """
    Solicitud para etiquetar estilo de escritura.
    """
    image_id: str
    image_path: str
    uncertainty_score: Optional[float] = None
    predicted_style: Optional[str] = Field(None, description="Estilo predicho por modelo")
    predicted_confidence: Optional[float] = Field(None, description="Confianza de predicción")


class StyleLabelResponse(BaseModel):
    """
    Respuesta con etiqueta de estilo.
    """
    image_id: str
    style_label: str = Field(..., description="Etiqueta de estilo asignada")
    confidence: Optional[float] = None
    labeler_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    # Validar que style_label sea uno de los permitidos
    @validator('style_label')
    def validate_style(cls, v):
        allowed_styles = ['es_mayores', 'latam_jovenes', 'general', 'other']
        if v not in allowed_styles:
            raise ValueError(f"style_label debe ser uno de: {allowed_styles}")
        return v


# =============================================================================
# ETIQUETADO OCR (PARA MODELO BASE/EXPERTOS)
# =============================================================================

class OCRLabelRequest(BaseModel):
    """
    Solicitud para etiquetar texto OCR.
    """
    image_id: str
    image_path: str
    uncertainty_score: Optional[float] = None
    predicted_text: Optional[str] = Field(None, description="Texto predicho por modelo")
    predicted_confidence: Optional[float] = None
    context: Optional[str] = Field(None, description="Contexto del documento")


class OCRLabelResponse(BaseModel):
    """
    Respuesta con transcripción OCR.
    """
    image_id: str
    text: str = Field(..., description="Texto transcrito correctamente")
    language: Optional[str] = Field("es", description="Idioma del texto")
    quality: Optional[str] = Field(None, description="Calidad de la imagen: good/medium/poor")
    labeler_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    corrections: Optional[str] = Field(None, description="Correcciones respecto a predicción")
    
    @validator('quality')
    def validate_quality(cls, v):
        if v is not None and v not in ['good', 'medium', 'poor']:
            raise ValueError("quality debe ser: good, medium, poor")
        return v


# =============================================================================
# ETIQUETADO MULTI-TASK (MÚLTIPLES ATRIBUTOS)
# =============================================================================

class MultiTaskLabelResponse(BaseModel):
    """
    Respuesta con múltiples atributos etiquetados.
    
    Útil para selector multi-task que predice:
    - age_group: edad del escritor
    - region: región geográfica
    - formality: nivel de formalidad
    - quality: calidad de escritura
    """
    image_id: str
    text: Optional[str] = Field(None, description="Texto (si aplica)")
    
    # Atributos de estilo
    age_group: Optional[str] = Field(None, description="joven/adulto/mayor")
    region: Optional[str] = Field(None, description="españa/latam/otro")
    formality: Optional[str] = Field(None, description="formal/informal")
    quality: Optional[str] = Field(None, description="clara/irregular/difícil")
    
    # Metadatos
    confidence: Optional[float] = None
    labeler_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('age_group')
    def validate_age_group(cls, v):
        if v is not None and v not in ['joven', 'adulto', 'mayor']:
            raise ValueError("age_group debe ser: joven, adulto, mayor")
        return v
    
    @validator('region')
    def validate_region(cls, v):
        if v is not None and v not in ['españa', 'latam', 'otro']:
            raise ValueError("region debe ser: españa, latam, otro")
        return v
    
    @validator('formality')
    def validate_formality(cls, v):
        if v is not None and v not in ['formal', 'informal']:
            raise ValueError("formality debe ser: formal, informal")
        return v
    
    @validator('quality')
    def validate_quality(cls, v):
        if v is not None and v not in ['clara', 'irregular', 'difícil']:
            raise ValueError("quality debe ser: clara, irregular, difícil")
        return v


# =============================================================================
# BATCH DE ETIQUETAS
# =============================================================================

class LabelBatch(BaseModel):
    """
    Batch de etiquetas para procesar en conjunto.
    """
    iteration: int = Field(..., description="Iteración de active learning")
    samples: List[LabelRequest] = Field(..., description="Lista de muestras a etiquetar")
    total_samples: int = Field(..., description="Total de muestras en el batch")
    strategy: Optional[str] = Field(None, description="Estrategia de muestreo usada")
    avg_uncertainty: Optional[float] = Field(None, description="Incertidumbre promedio")


class LabelBatchResponse(BaseModel):
    """
    Respuesta con batch de etiquetas completadas.
    """
    iteration: int
    labels: List[LabelResponse] = Field(..., description="Lista de etiquetas asignadas")
    completed_samples: int = Field(..., description="Muestras completadas")
    labeler_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    time_elapsed: Optional[float] = Field(None, description="Tiempo en segundos")


# =============================================================================
# VALIDACIÓN Y FEEDBACK
# =============================================================================

class LabelValidation(BaseModel):
    """
    Validación de una etiqueta por revisor.
    """
    image_id: str
    original_label: str = Field(..., description="Etiqueta original")
    validated_label: str = Field(..., description="Etiqueta validada")
    is_correct: bool = Field(..., description="Si la etiqueta original era correcta")
    validator_id: str = Field(..., description="ID del validador")
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None


class LabelerFeedback(BaseModel):
    """
    Feedback del etiquetador sobre dificultad de muestra.
    """
    image_id: str
    difficulty: str = Field(..., description="easy/medium/hard/impossible")
    issues: Optional[List[str]] = Field(None, description="Problemas encontrados")
    suggestions: Optional[str] = Field(None, description="Sugerencias de mejora")
    labeler_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v not in ['easy', 'medium', 'hard', 'impossible']:
            raise ValueError("difficulty debe ser: easy, medium, hard, impossible")
        return v


# =============================================================================
# ESTADÍSTICAS DE ETIQUETADO
# =============================================================================

class LabelingStats(BaseModel):
    """
    Estadísticas de una sesión de etiquetado.
    """
    labeler_id: str
    total_labeled: int = Field(..., description="Total de muestras etiquetadas")
    avg_time_per_sample: float = Field(..., description="Tiempo promedio por muestra (segundos)")
    avg_confidence: Optional[float] = Field(None, description="Confianza promedio")
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    start_time: datetime
    end_time: datetime
    
    @property
    def total_time(self) -> float:
        """Tiempo total de sesión en segundos."""
        return (self.end_time - self.start_time).total_seconds()


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def validate_label_file(jsonl_path: str, schema: BaseModel = LabelResponse) -> List[BaseModel]:
    """
    Valida un archivo JSONL de etiquetas.
    
    Args:
        jsonl_path: Path al archivo JSONL
        schema: Schema Pydantic para validar
    
    Returns:
        Lista de objetos validados
    
    Raises:
        ValidationError si hay errores
    """
    import json
    from pydantic import ValidationError
    
    validated = []
    errors = []
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                obj = schema(**data)
                validated.append(obj)
            except ValidationError as e:
                errors.append((line_num, str(e)))
            except json.JSONDecodeError as e:
                errors.append((line_num, f"JSON decode error: {e}"))
    
    if errors:
        print(f"⚠️  Encontrados {len(errors)} errores:")
        for line_num, error in errors[:5]:  # Mostrar primeros 5
            print(f"   Línea {line_num}: {error}")
        raise ValueError(f"Validación falló con {len(errors)} errores")
    
    print(f"✅ {len(validated)} etiquetas validadas correctamente")
    return validated


def export_label_requests(
    requests: List[LabelRequest],
    output_path: str,
    format: str = 'jsonl'
):
    """
    Exporta requests de etiquetado a archivo.
    
    Args:
        requests: Lista de LabelRequest
        output_path: Path de salida
        format: 'jsonl' o 'csv'
    """
    import json
    import pandas as pd
    
    if format == 'jsonl':
        with open(output_path, 'w') as f:
            for req in requests:
                f.write(req.json() + '\n')
    
    elif format == 'csv':
        data = [req.dict() for req in requests]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"💾 {len(requests)} requests exportados a {output_path}")


def import_label_responses(
    input_path: str,
    schema: BaseModel = LabelResponse
) -> List[BaseModel]:
    """
    Importa responses de etiquetado desde archivo.
    
    Args:
        input_path: Path al archivo JSONL
        schema: Schema para validar
    
    Returns:
        Lista de responses validados
    """
    return validate_label_file(input_path, schema)

