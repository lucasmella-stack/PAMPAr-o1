"""
main.py - API REST de llarri-01

API FastAPI completa para OCR de escritura manuscrita con:
- Endpoint de predicción individual
- Endpoint de predicción batch
- Endpoint de clasificación de estilo
- Sistema ensemble con routing automático
- Health check y métricas
- Documentación OpenAPI automática

Características:
- Soporte para base64 y URLs
- Selección dinámica de modelo/experto
- Métricas de latencia y uso
- CORS configurado
- Rate limiting (opcional)
- Logging estructurado

Uso:
    uvicorn llarri.api.main:app --host 0.0.0.0 --port 8000
    
    # Con hot reload (desarrollo)
    uvicorn llarri.api.main:app --reload
"""

import os
import io
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    StylePredictionRequest,
    StylePrediction,
    EnsemblePredictionResponse,
    ModelInfo,
    ModelsListResponse,
    HealthResponse,
    APIConfig,
    ErrorResponse,
    decode_base64_image
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

class APIState:
    """Estado global de la API."""
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.ensemble = None
        self.selector = None
        self.processor = None
        self.transform = None
        
        self.start_time = datetime.now()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = APIConfig()
        
        # Rutas de modelos (configurables via env vars)
        self.base_model_path = os.getenv('LLARRI_BASE_MODEL', 'outputs/base_model/best.ckpt')
        self.selector_path = os.getenv('LLARRI_SELECTOR', 'outputs/selector/best.ckpt')
        self.experts_dir = os.getenv('LLARRI_EXPERTS_DIR', 'outputs/experts')

state = APIState()


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación."""
    # Startup
    logger.info("🚀 Iniciando API llarri-01...")
    await load_models()
    logger.info("✅ API lista para recibir requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando API...")
    # Liberar memoria GPU si aplica
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("👋 API cerrada correctamente")


async def load_models():
    """Carga los modelos al iniciar la API."""
    try:
        # Cargar transforms
        from llarri.data.transforms import get_transforms
        state.transform = get_transforms(augment=False)
        
        # Intentar cargar modelo base
        if Path(state.base_model_path).exists():
            logger.info(f"📂 Cargando modelo base desde {state.base_model_path}")
            from llarri.models.llarri_base_model import LlarriBaseModel
            
            base_model = LlarriBaseModel.load_from_checkpoint(state.base_model_path)
            base_model = base_model.to(state.device)
            base_model.eval()
            
            state.models['base'] = base_model
            state.processor = base_model.processor
            logger.info("✅ Modelo base cargado")
        else:
            logger.warning(f"⚠️  Modelo base no encontrado en {state.base_model_path}")
        
        # Intentar cargar selector
        if Path(state.selector_path).exists():
            logger.info(f"📂 Cargando selector desde {state.selector_path}")
            from llarri.training.train_selector import StyleSelectorModule
            
            selector_module = StyleSelectorModule.load_from_checkpoint(state.selector_path)
            state.selector = selector_module.selector.to(state.device)
            state.selector.eval()
            
            state.models['selector'] = state.selector
            logger.info("✅ Selector cargado")
        else:
            logger.warning(f"⚠️  Selector no encontrado en {state.selector_path}")
        
        # Intentar cargar expertos
        experts_path = Path(state.experts_dir)
        if experts_path.exists():
            from llarri.training.finetune_expert import ExpertModel
            
            for expert_dir in experts_path.iterdir():
                if expert_dir.is_dir():
                    checkpoint = expert_dir / "best.ckpt"
                    if checkpoint.exists():
                        logger.info(f"📂 Cargando experto {expert_dir.name}")
                        expert = ExpertModel.load_from_checkpoint(str(checkpoint))
                        expert = expert.to(state.device)
                        expert.eval()
                        
                        state.models[f"expert_{expert_dir.name}"] = expert
                        logger.info(f"✅ Experto {expert_dir.name} cargado")
        
        # Crear ensemble si hay suficientes componentes
        if 'base' in state.models and state.selector is not None:
            logger.info("🔧 Creando ensemble...")
            from llarri.inference.ensemble import EnsembleInference
            
            expert_paths = {
                name: f"{state.experts_dir}/{name.replace('expert_', '')}/best.ckpt"
                for name in state.models.keys()
                if name.startswith('expert_')
            }
            
            if expert_paths:
                # Ensemble completo disponible (no cargamos de nuevo, usamos modelos existentes)
                state.ensemble = {
                    'selector': state.selector,
                    'base': state.models['base'],
                    'experts': {name: state.models[name] for name in state.models if name.startswith('expert_')}
                }
                logger.info("✅ Ensemble configurado")
        
        logger.info(f"📊 Total modelos cargados: {len(state.models)}")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {e}")
        raise


# =============================================================================
# CREAR APP
# =============================================================================

app = FastAPI(
    title="llarri-01 OCR API",
    description="""
    API de OCR especializada en escritura manuscrita con sistema de expertos.
    
    ## Características
    - 🔤 OCR de alta precisión para escritura manuscrita
    - 🎯 Sistema ensemble con expertos especializados
    - 🎨 Clasificación de estilo de escritura
    - 📦 Procesamiento batch
    - ⚡ Optimizado para GPU
    
    ## Modelos
    - **base**: Modelo general ViT + TrOCR
    - **expert_es_mayores**: Especializado en escritura de adultos mayores
    - **expert_latam_jovenes**: Especializado en escritura juvenil latinoamericana
    - **ensemble**: Selección automática del mejor experto
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Middleware para métricas de latencia."""
    start_time = time.time()
    
    response = await call_next(request)
    
    latency_ms = (time.time() - start_time) * 1000
    state.total_requests += 1
    state.total_latency_ms += latency_ms
    
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    
    return response


# =============================================================================
# ENDPOINTS - PREDICCIÓN
# =============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["OCR"])
async def predict(request: PredictionRequest):
    """
    Realiza OCR en una imagen.
    
    - **image_base64**: Imagen codificada en base64 (PNG, JPG, WebP)
    - **model_name**: Modelo a usar (default: ensemble si disponible, sino base)
    - **max_length**: Longitud máxima del texto generado
    - **num_beams**: Beams para beam search
    
    Retorna el texto reconocido y opcionalmente confianza y estilo.
    """
    start_time = time.time()
    
    try:
        # Decodificar imagen
        image_bytes = decode_base64_image(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Aplicar transforms
        image_tensor = state.transform(image).unsqueeze(0).to(state.device)
        
        # Seleccionar modelo
        model_name = request.model_name or ('ensemble' if state.ensemble else 'base')
        
        if model_name == 'ensemble' and state.ensemble:
            # Usar ensemble con routing
            result = await predict_with_ensemble(
                image_tensor, 
                request.max_length, 
                request.num_beams
            )
            text = result['text']
            model_used = result['expert_used']
        
        elif model_name in state.models:
            # Usar modelo específico
            model = state.models[model_name]
            text = await generate_text(
                model, 
                image_tensor, 
                request.max_length, 
                request.num_beams
            )
            model_used = model_name
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo '{model_name}' no disponible"
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            text=text,
            confidence=0.95,  # TODO: Calcular confianza real
            model_used=model_used,
            processing_time_ms=processing_time_ms
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["OCR"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza OCR en múltiples imágenes (máximo 32).
    
    Retorna lista de predicciones con información de éxitos y errores.
    """
    start_time = time.time()
    
    predictions = []
    errors = []
    
    for idx, img_b64 in enumerate(request.images):
        try:
            # Procesar cada imagen
            image_bytes = decode_base64_image(img_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = state.transform(image).unsqueeze(0).to(state.device)
            
            # Predecir
            model_name = request.model_name or 'base'
            if model_name in state.models:
                text = await generate_text(
                    state.models[model_name],
                    image_tensor,
                    request.max_length,
                    request.num_beams
                )
                
                predictions.append(PredictionResponse(
                    text=text,
                    confidence=0.95,
                    model_used=model_name
                ))
            else:
                errors.append({"index": idx, "error": f"Modelo '{model_name}' no disponible"})
        
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
            predictions.append(PredictionResponse(
                text="",
                confidence=0.0,
                model_used="error"
            ))
    
    total_time_ms = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(request.images),
        successful=len(request.images) - len(errors),
        failed=len(errors),
        total_processing_time_ms=total_time_ms,
        errors=errors if errors else None
    )


@app.post("/predict/style", response_model=StylePrediction, tags=["Style"])
async def predict_style(request: StylePredictionRequest):
    """
    Clasifica el estilo de escritura de una imagen.
    
    Retorna la clase predicha y opcionalmente probabilidades de todas las clases.
    """
    if state.selector is None:
        raise HTTPException(
            status_code=503, 
            detail="Selector de estilo no disponible"
        )
    
    try:
        # Decodificar imagen
        image_bytes = decode_base64_image(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = state.transform(image).unsqueeze(0).to(state.device)
        
        # Predecir estilo
        import torch.nn.functional as F
        
        with torch.no_grad():
            logits = state.selector(image_tensor)
            probs = F.softmax(logits, dim=-1)
            
            class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, class_id].item()
        
        # Mapear clase
        class_names = {
            0: "es_mayores",
            1: "latam_jovenes", 
            2: "general"
        }
        
        style_class = class_names.get(class_id, f"class_{class_id}")
        
        # Probabilidades por clase
        probabilities = None
        if request.return_probabilities:
            probabilities = {
                class_names.get(i, f"class_{i}"): float(probs[0, i])
                for i in range(probs.shape[1])
            }
        
        return StylePrediction(
            style_class=style_class,
            style_id=class_id,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        logger.error(f"Error en predicción de estilo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINTS - INFO Y HEALTH
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check del servicio.
    
    Retorna estado de salud, modelos cargados, uso de GPU, y métricas.
    """
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    
    avg_latency = None
    if state.total_requests > 0:
        avg_latency = state.total_latency_ms / state.total_requests
    
    # Determinar estado
    if len(state.models) == 0:
        status = "unhealthy"
    elif 'base' not in state.models:
        status = "degraded"
    else:
        status = "healthy"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=uptime,
        models_loaded=len(state.models),
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used_mb=gpu_memory,
        total_requests=state.total_requests,
        avg_response_time_ms=avg_latency
    )


@app.get("/models", response_model=ModelsListResponse, tags=["System"])
async def list_models():
    """
    Lista los modelos disponibles.
    """
    models_info = []
    
    for name, model in state.models.items():
        model_type = "base" if name == "base" else ("selector" if name == "selector" else "expert")
        
        models_info.append(ModelInfo(
            name=name,
            type=model_type,
            status="loaded",
            supports_ocr=name != "selector",
            supports_style_classification=name == "selector"
        ))
    
    return ModelsListResponse(
        models=models_info,
        default_model="ensemble" if state.ensemble else "base",
        total_loaded=len(models_info)
    )


@app.get("/config", response_model=APIConfig, tags=["System"])
async def get_config():
    """
    Retorna la configuración actual de la API.
    """
    return state.config


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

async def generate_text(
    model,
    image_tensor: torch.Tensor,
    max_length: int,
    num_beams: int
) -> str:
    """Genera texto usando un modelo."""
    with torch.no_grad():
        batch = {'pixel_values': image_tensor}
        
        generated_ids = model.generate(
            batch,
            max_length=max_length,
            num_beams=num_beams
        )
        
        text = state.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
    
    return text


async def predict_with_ensemble(
    image_tensor: torch.Tensor,
    max_length: int,
    num_beams: int
) -> Dict:
    """Predicción usando el ensemble con routing."""
    import torch.nn.functional as F
    
    # 1. Clasificar estilo
    with torch.no_grad():
        logits = state.selector(image_tensor)
        probs = F.softmax(logits, dim=-1)
        class_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, class_id].item()
    
    # 2. Routing
    class_to_expert = {
        0: "expert_es_mayores",
        1: "expert_latam_jovenes",
        2: "base"
    }
    
    expert_name = class_to_expert.get(class_id, "base")
    
    # Fallback si baja confianza
    if confidence < state.config.confidence_threshold:
        expert_name = "base"
    
    # 3. Usar experto o base
    if expert_name in state.ensemble.get('experts', {}):
        model = state.ensemble['experts'][expert_name]
    else:
        model = state.ensemble['base']
        expert_name = "base"
    
    # 4. Generar texto
    text = await generate_text(model, image_tensor, max_length, num_beams)
    
    return {
        'text': text,
        'expert_used': expert_name,
        'selector_confidence': confidence,
        'class_id': class_id
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Error interno del servidor",
            error_code="INTERNAL_ERROR",
            details={"exception": str(exc)}
        ).dict()
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "llarri.api.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )

