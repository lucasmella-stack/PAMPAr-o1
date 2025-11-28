"""
ensemble_ocr.py - Ensemble OCR combinando múltiples modelos

Combina:
- LlarriBaseModel (TrOCR especializado)
- MiniCPM-V (modelo multimodal poderoso)
- SpanishLanguageModel (post-procesamiento)

Estrategias de ensemble:
1. verify_if_low_conf: Solo verifica si LLARRI tiene baja confianza
2. always_verify: Siempre verifica con MiniCPM
3. consensus: Acepta si múltiples modelos coinciden
4. rerank: Genera candidatos y re-rankea con LM

Uso:
    ensemble = EnsembleOCR()
    result = ensemble.predict(image)
    print(result.text, result.confidence, result.strategy_used)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Estrategias de ensemble disponibles."""
    VERIFY_IF_LOW_CONF = "verify_if_low_conf"
    ALWAYS_VERIFY = "always_verify"
    CONSENSUS = "consensus"
    RERANK = "rerank"
    LLARRI_ONLY = "llarri_only"  # Solo LLARRI + LM
    MINICPM_ONLY = "minicpm_only"  # Solo MiniCPM + LM


@dataclass
class EnsembleConfig:
    """Configuración del ensemble."""
    strategy: EnsembleStrategy = EnsembleStrategy.VERIFY_IF_LOW_CONF
    
    # Thresholds
    low_confidence_threshold: float = 0.85  # Debajo de esto, verificar
    consensus_threshold: float = 0.9  # Para estrategia consensus
    
    # Pesos para scoring
    llarri_weight: float = 0.5
    minicpm_weight: float = 0.3
    lm_weight: float = 0.2
    
    # Comportamiento
    use_language_model: bool = True
    fallback_on_error: bool = True  # Si MiniCPM falla, usar solo LLARRI
    
    # Logging
    verbose: bool = False


@dataclass
class EnsemblePrediction:
    """Resultado de una predicción del ensemble."""
    text: str
    confidence: float
    
    # Metadata
    strategy_used: str = ""
    llarri_prediction: Optional[str] = None
    llarri_confidence: Optional[float] = None
    minicpm_prediction: Optional[str] = None
    minicpm_confidence: Optional[float] = None
    lm_corrected: bool = False
    
    # Timing
    total_time_ms: float = 0.0
    llarri_time_ms: float = 0.0
    minicpm_time_ms: float = 0.0
    lm_time_ms: float = 0.0
    
    # Debug info
    decision_reason: str = ""
    all_candidates: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para logging/API."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
            "llarri_prediction": self.llarri_prediction,
            "minicpm_prediction": self.minicpm_prediction,
            "lm_corrected": self.lm_corrected,
            "total_time_ms": self.total_time_ms,
            "decision_reason": self.decision_reason,
        }


class EnsembleOCR:
    """
    Ensemble OCR que combina múltiples modelos para mayor precisión.
    
    Arquitectura:
    
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Imagen    │────▶│   LLARRI    │────▶│  Predicción │
    └─────────────┘     │   (TrOCR)   │     │  + Confianza│
                        └─────────────┘     └──────┬──────┘
                                                   │
                        ┌─────────────┐            │
                        │  MiniCPM-V  │◀───────────┤ (si conf < threshold)
                        │ (Verificar) │            │
                        └──────┬──────┘            │
                               │                   │
                               ▼                   ▼
                        ┌─────────────────────────────┐
                        │     Language Model          │
                        │  (Re-rank / Corregir)       │
                        └─────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    Predicción Final         │
                        │  + Confianza + Metadata     │
                        └─────────────────────────────┘
    """
    
    def __init__(
        self,
        llarri_model: Optional[Any] = None,
        minicpm_adapter: Optional[Any] = None,
        language_model: Optional[Any] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Inicializa el ensemble.
        
        Args:
            llarri_model: Instancia de LlarriBaseModel
            minicpm_adapter: Instancia de MiniCPMAdapter
            language_model: Instancia de SpanishLanguageModel
            config: Configuración del ensemble
        """
        self.config = config or EnsembleConfig()
        
        # Componentes (se cargan lazy si no se proporcionan)
        self._llarri = llarri_model
        self._minicpm = minicpm_adapter
        self._lm = language_model
        
        # Estado
        self._initialized = False
    
    def _ensure_initialized(self):
        """Carga componentes de forma lazy."""
        if self._initialized:
            return
        
        # Cargar LLARRI si no está
        if self._llarri is None:
            try:
                from llarri.models.llarri_base_model import LlarriBaseModel
                self._llarri = LlarriBaseModel()
                self._llarri.eval()
                logger.info("LlarriBaseModel cargado")
            except Exception as e:
                logger.error(f"Error cargando LlarriBaseModel: {e}")
        
        # Cargar MiniCPM adapter si no está
        if self._minicpm is None:
            try:
                from llarri.inference.minicpm_adapter import MiniCPMAdapter
                self._minicpm = MiniCPMAdapter()  # Usa config de env
                logger.info(f"MiniCPMAdapter cargado en modo: {self._minicpm.config.mode.value}")
            except Exception as e:
                logger.warning(f"MiniCPM no disponible: {e}")
        
        # Cargar Language Model si no está
        if self._lm is None and self.config.use_language_model:
            try:
                from llarri.inference.language_model import SpanishLanguageModel
                self._lm = SpanishLanguageModel()
                logger.info("SpanishLanguageModel cargado")
            except Exception as e:
                logger.warning(f"Language Model no disponible: {e}")
        
        self._initialized = True
    
    def predict(
        self,
        image: Any,
        strategy: Optional[EnsembleStrategy] = None,
        **kwargs
    ) -> EnsemblePrediction:
        """
        Genera predicción usando el ensemble.
        
        Args:
            image: PIL.Image, path, o tensor
            strategy: Estrategia a usar (override config)
            **kwargs: Parámetros adicionales
            
        Returns:
            EnsemblePrediction con resultado y metadata
        """
        self._ensure_initialized()
        
        strategy = strategy or self.config.strategy
        start_time = time.time()
        
        # Dispatch según estrategia
        if strategy == EnsembleStrategy.LLARRI_ONLY:
            result = self._predict_llarri_only(image)
        elif strategy == EnsembleStrategy.MINICPM_ONLY:
            result = self._predict_minicpm_only(image)
        elif strategy == EnsembleStrategy.VERIFY_IF_LOW_CONF:
            result = self._predict_verify_if_low_conf(image)
        elif strategy == EnsembleStrategy.ALWAYS_VERIFY:
            result = self._predict_always_verify(image)
        elif strategy == EnsembleStrategy.CONSENSUS:
            result = self._predict_consensus(image)
        elif strategy == EnsembleStrategy.RERANK:
            result = self._predict_rerank(image)
        else:
            result = self._predict_verify_if_low_conf(image)
        
        result.total_time_ms = (time.time() - start_time) * 1000
        result.strategy_used = strategy.value
        
        if self.config.verbose:
            logger.info(f"Ensemble prediction: {result.to_dict()}")
        
        return result
    
    def _predict_llarri_only(self, image: Any) -> EnsemblePrediction:
        """Solo usa LLARRI + Language Model."""
        result = EnsemblePrediction(text="", confidence=0.0)
        
        # LLARRI prediction
        start = time.time()
        if self._llarri:
            try:
                text = self._llarri.predict(image, preprocess=True, use_language_model=False)
                result.llarri_prediction = text
                result.llarri_confidence = 0.9  # TODO: obtener confianza real
                result.text = text
                result.confidence = result.llarri_confidence
            except Exception as e:
                logger.error(f"Error en LLARRI: {e}")
                result.decision_reason = f"LLARRI error: {e}"
        result.llarri_time_ms = (time.time() - start) * 1000
        
        # Language Model correction
        if self._lm and result.text:
            start = time.time()
            corrected = self._lm.correct_text(result.text)
            if corrected != result.text:
                result.text = corrected
                result.lm_corrected = True
            result.lm_time_ms = (time.time() - start) * 1000
        
        result.decision_reason = "LLARRI + LM"
        return result
    
    def _predict_minicpm_only(self, image: Any) -> EnsemblePrediction:
        """Solo usa MiniCPM + Language Model."""
        result = EnsemblePrediction(text="", confidence=0.0)
        
        # MiniCPM prediction
        start = time.time()
        if self._minicpm and self._minicpm.is_available():
            try:
                text, score = self._minicpm.transcribe_image(image)
                result.minicpm_prediction = text
                result.minicpm_confidence = score
                result.text = text
                result.confidence = score
            except Exception as e:
                logger.error(f"Error en MiniCPM: {e}")
        result.minicpm_time_ms = (time.time() - start) * 1000
        
        # Language Model correction
        if self._lm and result.text:
            start = time.time()
            corrected = self._lm.correct_text(result.text)
            if corrected != result.text:
                result.text = corrected
                result.lm_corrected = True
            result.lm_time_ms = (time.time() - start) * 1000
        
        result.decision_reason = "MiniCPM + LM"
        return result
    
    def _predict_verify_if_low_conf(self, image: Any) -> EnsemblePrediction:
        """
        Estrategia principal: LLARRI primero, verificar con MiniCPM si baja confianza.
        """
        result = EnsemblePrediction(text="", confidence=0.0)
        
        # 1. LLARRI prediction
        start = time.time()
        if self._llarri:
            try:
                text = self._llarri.predict(image, preprocess=True, use_language_model=False)
                result.llarri_prediction = text
                # Estimar confianza basada en longitud y caracteres especiales
                result.llarri_confidence = self._estimate_confidence(text)
            except Exception as e:
                logger.error(f"Error en LLARRI: {e}")
                result.llarri_confidence = 0.0
        result.llarri_time_ms = (time.time() - start) * 1000
        
        # 2. Verificar con MiniCPM si baja confianza
        need_verification = (
            result.llarri_confidence < self.config.low_confidence_threshold
            or not result.llarri_prediction
        )
        
        if need_verification and self._minicpm and self._minicpm.is_available():
            start = time.time()
            try:
                text, score = self._minicpm.transcribe_image(image)
                result.minicpm_prediction = text
                result.minicpm_confidence = score
            except Exception as e:
                logger.warning(f"Error en verificación MiniCPM: {e}")
            result.minicpm_time_ms = (time.time() - start) * 1000
        
        # 3. Decidir mejor resultado
        result = self._decide_best_prediction(result)
        
        # 4. Language Model correction
        if self._lm and result.text:
            start = time.time()
            corrected = self._lm.correct_text(result.text)
            if corrected != result.text:
                result.all_candidates.append(result.text)
                result.text = corrected
                result.lm_corrected = True
            result.lm_time_ms = (time.time() - start) * 1000
        
        return result
    
    def _predict_always_verify(self, image: Any) -> EnsemblePrediction:
        """Siempre ejecuta ambos modelos y combina."""
        result = EnsemblePrediction(text="", confidence=0.0)
        
        # LLARRI
        start = time.time()
        if self._llarri:
            try:
                text = self._llarri.predict(image, preprocess=True, use_language_model=False)
                result.llarri_prediction = text
                result.llarri_confidence = self._estimate_confidence(text)
            except Exception as e:
                logger.error(f"Error en LLARRI: {e}")
        result.llarri_time_ms = (time.time() - start) * 1000
        
        # MiniCPM
        start = time.time()
        if self._minicpm and self._minicpm.is_available():
            try:
                text, score = self._minicpm.transcribe_image(image)
                result.minicpm_prediction = text
                result.minicpm_confidence = score
            except Exception as e:
                logger.warning(f"Error en MiniCPM: {e}")
        result.minicpm_time_ms = (time.time() - start) * 1000
        
        # Decidir
        result = self._decide_best_prediction(result)
        
        # LM
        if self._lm and result.text:
            start = time.time()
            corrected = self._lm.correct_text(result.text)
            if corrected != result.text:
                result.text = corrected
                result.lm_corrected = True
            result.lm_time_ms = (time.time() - start) * 1000
        
        return result
    
    def _predict_consensus(self, image: Any) -> EnsemblePrediction:
        """Requiere acuerdo entre modelos."""
        result = self._predict_always_verify(image)
        
        # Verificar consenso
        if result.llarri_prediction and result.minicpm_prediction:
            # Normalizar para comparación
            llarri_norm = result.llarri_prediction.lower().strip()
            minicpm_norm = result.minicpm_prediction.lower().strip()
            
            if llarri_norm == minicpm_norm:
                result.confidence = 0.99  # Alto consenso
                result.decision_reason = "Consensus: models agree"
            else:
                # Calcular similitud
                similarity = self._text_similarity(llarri_norm, minicpm_norm)
                if similarity > self.config.consensus_threshold:
                    result.confidence = similarity
                    result.decision_reason = f"Near consensus: {similarity:.2%} similar"
                else:
                    result.confidence = max(
                        result.llarri_confidence or 0,
                        result.minicpm_confidence or 0
                    ) * 0.8  # Penalizar por falta de consenso
                    result.decision_reason = f"No consensus: {similarity:.2%} similar"
        
        return result
    
    def _predict_rerank(self, image: Any) -> EnsemblePrediction:
        """Genera múltiples candidatos y re-rankea con LM."""
        result = self._predict_always_verify(image)
        
        # Recolectar todos los candidatos
        candidates = []
        if result.llarri_prediction:
            candidates.append(result.llarri_prediction)
        if result.minicpm_prediction:
            candidates.append(result.minicpm_prediction)
        
        # Agregar correcciones del LM como candidatos
        if self._lm and candidates:
            for c in list(candidates):
                corrected = self._lm.correct_text(c)
                if corrected not in candidates:
                    candidates.append(corrected)
        
        result.all_candidates = candidates
        
        # Re-rankear con LM
        if self._lm and len(candidates) > 1:
            start = time.time()
            ranked = self._lm.rerank_candidates(candidates)
            result.text = ranked[0] if ranked else result.text
            result.lm_time_ms = (time.time() - start) * 1000
            result.lm_corrected = True
            result.decision_reason = f"Reranked from {len(candidates)} candidates"
        
        return result
    
    def _decide_best_prediction(self, result: EnsemblePrediction) -> EnsemblePrediction:
        """Decide la mejor predicción entre LLARRI y MiniCPM."""
        llarri_text = result.llarri_prediction
        llarri_conf = result.llarri_confidence or 0
        minicpm_text = result.minicpm_prediction
        minicpm_conf = result.minicpm_confidence or 0
        
        # Caso 1: Solo LLARRI disponible
        if llarri_text and not minicpm_text:
            result.text = llarri_text
            result.confidence = llarri_conf
            result.decision_reason = "Only LLARRI available"
            return result
        
        # Caso 2: Solo MiniCPM disponible
        if minicpm_text and not llarri_text:
            result.text = minicpm_text
            result.confidence = minicpm_conf
            result.decision_reason = "Only MiniCPM available"
            return result
        
        # Caso 3: Ambos disponibles
        if llarri_text and minicpm_text:
            # Si coinciden → alta confianza
            if llarri_text.lower().strip() == minicpm_text.lower().strip():
                result.text = llarri_text
                result.confidence = max(llarri_conf, minicpm_conf, 0.95)
                result.decision_reason = "Both models agree"
                return result
            
            # Si difieren → usar scores ponderados
            llarri_score = llarri_conf * self.config.llarri_weight
            minicpm_score = minicpm_conf * self.config.minicpm_weight
            
            # Bonus si LM valida
            if self._lm:
                llarri_lm = self._lm.score_text(llarri_text)
                minicpm_lm = self._lm.score_text(minicpm_text)
                llarri_score += llarri_lm * self.config.lm_weight
                minicpm_score += minicpm_lm * self.config.lm_weight
            
            if llarri_score >= minicpm_score:
                result.text = llarri_text
                result.confidence = llarri_conf
                result.decision_reason = f"LLARRI won ({llarri_score:.3f} vs {minicpm_score:.3f})"
            else:
                result.text = minicpm_text
                result.confidence = minicpm_conf
                result.decision_reason = f"MiniCPM won ({minicpm_score:.3f} vs {llarri_score:.3f})"
            
            return result
        
        # Caso 4: Ninguno disponible
        result.text = ""
        result.confidence = 0.0
        result.decision_reason = "No predictions available"
        return result
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estima confianza basada en características del texto.
        
        Heurísticas:
        - Texto vacío → 0
        - Muchos caracteres raros → baja
        - Longitud razonable → alta
        - Palabras válidas (LM) → alta
        """
        if not text:
            return 0.0
        
        confidence = 0.8  # Base
        
        # Penalizar texto muy corto
        if len(text) < 3:
            confidence *= 0.5
        
        # Penalizar caracteres raros
        special_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?-\'\"') / max(1, len(text))
        if special_ratio > 0.2:
            confidence *= (1 - special_ratio)
        
        # Bonus si LM dice que es válido
        if self._lm:
            lm_score = self._lm.score_text(text)
            confidence = (confidence + lm_score) / 2
        
        return min(1.0, max(0.0, confidence))
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud entre dos textos."""
        if not text1 or not text2:
            return 0.0
        
        if text1 == text2:
            return 1.0
        
        # Levenshtein distance
        try:
            import Levenshtein
            distance = Levenshtein.distance(text1, text2)
            max_len = max(len(text1), len(text2))
            return 1 - (distance / max_len)
        except ImportError:
            # Fallback simple
            common = set(text1.split()) & set(text2.split())
            total = set(text1.split()) | set(text2.split())
            return len(common) / max(1, len(total))


# Función de conveniencia
def create_ensemble(
    strategy: str = "verify_if_low_conf",
    **kwargs
) -> EnsembleOCR:
    """
    Crea un EnsembleOCR con la configuración especificada.
    
    Args:
        strategy: Estrategia de ensemble
        **kwargs: Configuración adicional
        
    Returns:
        EnsembleOCR configurado
    """
    config = EnsembleConfig(
        strategy=EnsembleStrategy(strategy),
        **{k: v for k, v in kwargs.items() if hasattr(EnsembleConfig, k)}
    )
    return EnsembleOCR(config=config)
