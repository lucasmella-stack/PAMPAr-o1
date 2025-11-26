"""
schemas.py - Esquemas Pydantic para la API REST

Define todos los modelos de request/response para la API de OCR:
- PredictionRequest/Response: Predicción de imagen única
- BatchPredictionRequest/Response: Predicción de múltiples imágenes
- ModelInfo: Información de modelos cargados
- HealthResponse: Estado de salud de la API
- StylePrediction: Predicción de estilo de escritura
- EnsembleResult: Resultado con información del ensemble

Validaciones incluidas:
- Base64 válido
- Tamaño máximo de imagen
- Límite de batch size
- Confianza en rango [0, 1]
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import base64


# =============================================================================
# REQUESTS
# =============================================================================

class PredictionRequest(BaseModel):
    """
    Request para predicción OCR de imagen única.
    """
    image_base64: str = Field(
        ..., 
        description="Imagen codificada en base64 (PNG, JPG, WebP)"
    )
    model_name: Optional[str] = Field(
        None,
        description="Nombre del modelo a usar (default: ensemble)"
    )
    max_length: Optional[int] = Field(
        128,
        ge=1,
        le=512,
        description="Longitud máxima del texto generado"
    )
    num_beams: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Número de beams para beam search"
    )
    return_confidence: Optional[bool] = Field(
        True,
        description="Si retornar score de confianza"
    )
    return_style: Optional[bool] = Field(
        False,
        description="Si retornar clasificación de estilo"
    )
    
    @validator('image_base64')
    def validate_base64(cls, v):
        """Valida que el string sea base64 válido."""
        try:
            # Remover prefijo data:image si existe
            if ',' in v:
                v = v.split(',')[1]
            
            decoded = base64.b64decode(v)
            
            # Verificar tamaño máximo (10MB)
            if len(decoded) > 10 * 1024 * 1024:
                raise ValueError("Imagen excede tamaño máximo (10MB)")
            
            return v
        except Exception as e:
            raise ValueError(f"Base64 inválido: {str(e)}")


class ImageURLRequest(BaseModel):
    """
    Request para predicción OCR desde URL.
    """
    image_url: str = Field(
        ...,
        description="URL de la imagen a procesar"
    )
    model_name: Optional[str] = None
    max_length: Optional[int] = Field(128, ge=1, le=512)
    num_beams: Optional[int] = Field(4, ge=1, le=10)


class BatchPredictionRequest(BaseModel):
    """
    Request para predicción OCR de múltiples imágenes.
    """
    images: List[str] = Field(
        ...,
        min_items=1,
        max_items=32,
        description="Lista de imágenes en base64 (máximo 32)"
    )
    model_name: Optional[str] = None
    max_length: Optional[int] = Field(128, ge=1, le=512)
    num_beams: Optional[int] = Field(4, ge=1, le=10)
    return_confidence: Optional[bool] = True
    return_style: Optional[bool] = False
    
    @validator('images')
    def validate_batch_size(cls, v):
        if len(v) > 32:
            raise ValueError("Máximo 32 imágenes por batch")
        return v


class StylePredictionRequest(BaseModel):
    """
    Request para clasificación de estilo de escritura.
    """
    image_base64: str = Field(..., description="Imagen en base64")
    return_probabilities: Optional[bool] = Field(
        False,
        description="Si retornar probabilidades de todas las clases"
    )


# =============================================================================
# RESPONSES
# =============================================================================

class PredictionResponse(BaseModel):
    """
    Response de predicción OCR individual.
    """
    text: str = Field(..., description="Texto reconocido")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confianza de la predicción"
    )
    model_used: Optional[str] = Field(
        None,
        description="Nombre del modelo que procesó la imagen"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Tiempo de procesamiento en milisegundos"
    )


class StylePrediction(BaseModel):
    """
    Predicción de estilo de escritura.
    """
    style_class: str = Field(..., description="Clase de estilo predicha")
    style_id: int = Field(..., description="ID numérico de la clase")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Probabilidades de cada clase"
    )


class EnsemblePredictionResponse(BaseModel):
    """
    Response completo del ensemble con información detallada.
    """
    text: str = Field(..., description="Texto reconocido")
    confidence: Optional[float] = None
    
    # Información del ensemble
    expert_used: str = Field(..., description="Experto que procesó la imagen")
    selector_confidence: float = Field(..., description="Confianza del selector")
    used_fallback: bool = Field(
        ..., 
        description="Si se usó modelo base como fallback"
    )
    
    # Estilo
    style: Optional[StylePrediction] = None
    
    # Metadata
    processing_time_ms: float
    model_version: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """
    Response de predicción batch.
    """
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Lista de predicciones"
    )
    total_images: int
    successful: int
    failed: int
    total_processing_time_ms: float
    errors: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Errores por imagen (si hubo)"
    )


class ErrorResponse(BaseModel):
    """
    Response de error.
    """
    error: str = Field(..., description="Mensaje de error")
    error_code: str = Field(..., description="Código de error")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# MODELOS E INFO
# =============================================================================

class ModelInfo(BaseModel):
    """
    Información de un modelo cargado.
    """
    name: str = Field(..., description="Nombre del modelo")
    type: str = Field(..., description="Tipo: base, expert, selector, ensemble")
    status: str = Field(..., description="Estado: loaded, loading, error")
    checkpoint_path: Optional[str] = None
    
    # Capacidades
    supports_ocr: bool = True
    supports_style_classification: bool = False
    
    # Estadísticas
    total_predictions: int = 0
    avg_latency_ms: Optional[float] = None
    last_used: Optional[datetime] = None


class ModelsListResponse(BaseModel):
    """
    Lista de modelos disponibles.
    """
    models: List[ModelInfo]
    default_model: str
    total_loaded: int


class HealthResponse(BaseModel):
    """
    Response de health check.
    """
    status: str = Field(..., description="Estado: healthy, degraded, unhealthy")
    version: str = Field(..., description="Versión de la API")
    uptime_seconds: float
    
    # Estado de componentes
    models_loaded: int
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    
    # Métricas
    total_requests: int = 0
    avg_response_time_ms: Optional[float] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)


class APIConfig(BaseModel):
    """
    Configuración de la API (para endpoint /config).
    """
    max_batch_size: int = 32
    max_image_size_mb: float = 10.0
    default_max_length: int = 128
    default_num_beams: int = 4
    confidence_threshold: float = 0.7
    supported_formats: List[str] = ["png", "jpg", "jpeg", "webp"]
    rate_limit_per_minute: Optional[int] = None


# =============================================================================
# ACTIVE LEARNING
# =============================================================================

class UncertainSampleResponse(BaseModel):
    """
    Muestra con alta incertidumbre para active learning.
    """
    image_id: str
    uncertainty_score: float
    predicted_text: str
    predicted_style: Optional[str] = None
    needs_review: bool = True


class ActiveLearningStatusResponse(BaseModel):
    """
    Estado del sistema de active learning.
    """
    enabled: bool
    current_iteration: int
    samples_in_queue: int
    last_update: Optional[datetime] = None


# =============================================================================
# UTILIDADES
# =============================================================================

def decode_base64_image(base64_str: str) -> bytes:
    """
    Decodifica imagen base64 a bytes.
    
    Maneja prefijos como 'data:image/png;base64,'
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    return base64.b64decode(base64_str)


def encode_image_base64(image_bytes: bytes, format: str = "png") -> str:
    """
    Codifica bytes de imagen a base64 con prefijo.
    """
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{b64_str}"

