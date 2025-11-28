"""
minicpm_adapter.py - Adapter para MiniCPM-V como verificador/ensemble OCR

MiniCPM-V es un modelo multimodal de 8B parámetros que supera a GPT-4o
en tareas de OCR. Este adapter permite usarlo como:
- Verificador de predicciones de LLARRI
- Modelo de ensemble para mayor precisión
- Modelo de comprensión de documentos (contexto)

Modos de operación:
- local_hf: Carga el modelo localmente con HuggingFace transformers
- local_quantized: Carga con quantización 4/8-bit (menos VRAM)
- remote_rest: Llama a un servicio REST (HuggingFace Inference, etc.)
- mock: Para testing sin modelo real

Uso:
    adapter = MiniCPMAdapter(mode='local_hf')
    text, score = adapter.transcribe_image(image)
    
    # O con comprensión contextual
    context = adapter.chat_image(image, "¿Qué tipo de documento es?")

Repositorio: https://github.com/OpenBMB/MiniCPM-V
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Union, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Configurar logging
logger = logging.getLogger(__name__)


class AdapterMode(Enum):
    """Modos de operación del adapter."""
    LOCAL_HF = "local_hf"
    LOCAL_QUANTIZED = "local_quantized"
    REMOTE_REST = "remote_rest"
    MOCK = "mock"


@dataclass
class MiniCPMConfig:
    """Configuración para MiniCPM-V adapter."""
    mode: AdapterMode = AdapterMode.MOCK
    model_name: str = "openbmb/MiniCPM-V-2_6"
    device: str = "cuda"  # cuda, cpu, auto
    quantize: Optional[str] = None  # "4bit", "8bit", None
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # Para modo remote
    rest_url: Optional[str] = None
    rest_api_key: Optional[str] = None
    rest_timeout: int = 30
    
    # Cache
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'MiniCPMConfig':
        """Carga configuración desde variables de entorno."""
        mode_str = os.getenv("MINICPM_MODE", "mock")
        try:
            mode = AdapterMode(mode_str)
        except ValueError:
            mode = AdapterMode.MOCK
        
        return cls(
            mode=mode,
            model_name=os.getenv("MINICPM_MODEL", "openbmb/MiniCPM-V-2_6"),
            device=os.getenv("MINICPM_DEVICE", "cuda"),
            quantize=os.getenv("MINICPM_QUANTIZE"),
            rest_url=os.getenv("MINICPM_REST_URL"),
            rest_api_key=os.getenv("MINICPM_API_KEY"),
        )


class BaseMiniCPMBackend(ABC):
    """Interfaz base para backends de MiniCPM."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el backend está disponible."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Transcribe texto de una imagen.
        
        Args:
            image: PIL.Image o path a imagen
            prompt: Prompt opcional para guiar la transcripción
            
        Returns:
            Tuple de (texto_transcrito, score_confianza)
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        """
        Chat con comprensión de imagen.
        
        Args:
            image: PIL.Image o path a imagen
            prompt: Pregunta o instrucción
            
        Returns:
            Respuesta del modelo
        """
        pass


class MockBackend(BaseMiniCPMBackend):
    """
    Backend mock para testing.
    Simula respuestas sin modelo real.
    """
    
    def is_available(self) -> bool:
        return True
    
    def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        # Simular transcripción básica
        return "[MOCK] Texto transcrito de imagen", 0.85
    
    def chat(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        return f"[MOCK] Respuesta a: {prompt}"


class LocalHFBackend(BaseMiniCPMBackend):
    """
    Backend que carga MiniCPM-V localmente con HuggingFace.
    
    Requiere:
    - GPU con >= 16GB VRAM (sin quantización)
    - O GPU con >= 8GB VRAM (con quantización 4-bit)
    """
    
    def __init__(self, config: MiniCPMConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._loaded = False
    
    def _load_model(self):
        """Carga el modelo de forma lazy."""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Cargando MiniCPM-V desde {self.config.model_name}...")
            
            # Configurar device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Cargar con o sin quantización
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if self.config.quantize == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    logger.info("Usando quantización 4-bit")
                except ImportError:
                    logger.warning("bitsandbytes no disponible, cargando sin quantización")
            
            elif self.config.quantize == "8bit":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Usando quantización 8-bit")
                except ImportError:
                    logger.warning("bitsandbytes no disponible, cargando sin quantización")
            
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            if not self.config.quantize:
                self._model = self._model.to(device)
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            self._model.eval()
            self._loaded = True
            logger.info("MiniCPM-V cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error cargando MiniCPM-V: {e}")
            raise
    
    def is_available(self) -> bool:
        try:
            import torch
            from transformers import AutoModel
            return True
        except ImportError:
            return False
    
    def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        self._load_model()
        
        from PIL import Image
        
        # Cargar imagen si es path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Prompt para OCR
        if prompt is None:
            prompt = "Transcribe exactamente el texto manuscrito en esta imagen. Solo devuelve el texto, sin explicaciones."
        
        # Generar respuesta
        try:
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            response = self._model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self._tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            
            # MiniCPM-V no devuelve score directo, estimamos basado en longitud
            # En producción, se podría calcular con log-probs
            estimated_score = 0.9
            
            return response, estimated_score
            
        except Exception as e:
            logger.error(f"Error en transcribe: {e}")
            return "", 0.0
    
    def chat(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        self._load_model()
        
        from PIL import Image
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        try:
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            response = self._model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self._tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return ""


class RemoteRESTBackend(BaseMiniCPMBackend):
    """
    Backend que llama a un servicio REST.
    
    Compatible con:
    - HuggingFace Inference API
    - Servidor custom con OpenAI-compatible API
    - Ollama con visión
    """
    
    def __init__(self, config: MiniCPMConfig):
        self.config = config
        self._session = None
    
    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            if self.config.rest_api_key:
                self._session.headers["Authorization"] = f"Bearer {self.config.rest_api_key}"
        return self._session
    
    def is_available(self) -> bool:
        if not self.config.rest_url:
            return False
        try:
            import requests
            response = requests.get(
                f"{self.config.rest_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return True  # Asumimos disponible si hay URL configurada
    
    def _image_to_base64(self, image: Any) -> str:
        """Convierte imagen a base64."""
        import base64
        from io import BytesIO
        from PIL import Image
        
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        if prompt is None:
            prompt = "Transcribe exactamente el texto manuscrito en esta imagen."
        
        try:
            session = self._get_session()
            
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self._image_to_base64(image)}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
            }
            
            response = session.post(
                f"{self.config.rest_url}/v1/chat/completions",
                json=payload,
                timeout=self.config.rest_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            
            # Intentar obtener score si está disponible
            score = 0.9
            if "usage" in data:
                # Heurística: más tokens generados → más confianza
                score = min(0.95, 0.7 + (len(text) / 500) * 0.25)
            
            return text, score
            
        except Exception as e:
            logger.error(f"Error en REST transcribe: {e}")
            return "", 0.0
    
    def chat(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        text, _ = self.transcribe(image, prompt)
        return text


class MiniCPMAdapter:
    """
    Adapter principal para MiniCPM-V.
    
    Encapsula la lógica de selección de backend y proporciona
    una interfaz unificada para transcripción y chat.
    
    Uso:
        # Modo automático (lee de env vars)
        adapter = MiniCPMAdapter()
        
        # Modo específico
        adapter = MiniCPMAdapter(mode='local_hf', device='cuda')
        
        # Transcribir
        text, score = adapter.transcribe_image("documento.jpg")
        
        # Chat con contexto
        tipo = adapter.chat_image("documento.jpg", "¿Qué tipo de documento es?")
    """
    
    def __init__(
        self,
        mode: Optional[str] = None,
        config: Optional[MiniCPMConfig] = None,
        **kwargs
    ):
        """
        Inicializa el adapter.
        
        Args:
            mode: Modo de operación ('local_hf', 'local_quantized', 'remote_rest', 'mock')
            config: Configuración completa (opcional)
            **kwargs: Parámetros adicionales para config
        """
        if config:
            self.config = config
        else:
            self.config = MiniCPMConfig.from_env()
        
        # Override con parámetros explícitos
        if mode:
            self.config.mode = AdapterMode(mode)
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Inicializar backend
        self._backend = self._create_backend()
        
        logger.info(f"MiniCPMAdapter inicializado en modo: {self.config.mode.value}")
    
    def _create_backend(self) -> BaseMiniCPMBackend:
        """Crea el backend según el modo configurado."""
        mode = self.config.mode
        
        if mode == AdapterMode.MOCK:
            return MockBackend()
        
        elif mode in (AdapterMode.LOCAL_HF, AdapterMode.LOCAL_QUANTIZED):
            if mode == AdapterMode.LOCAL_QUANTIZED and not self.config.quantize:
                self.config.quantize = "4bit"
            return LocalHFBackend(self.config)
        
        elif mode == AdapterMode.REMOTE_REST:
            if not self.config.rest_url:
                logger.warning("No REST URL configured, falling back to mock")
                return MockBackend()
            return RemoteRESTBackend(self.config)
        
        else:
            logger.warning(f"Modo desconocido: {mode}, usando mock")
            return MockBackend()
    
    def is_available(self) -> bool:
        """Verifica si el modelo está disponible."""
        return self._backend.is_available()
    
    def transcribe_image(
        self,
        image: Any,
        prompt: Optional[str] = None,
        language: str = "es",
    ) -> Tuple[str, float]:
        """
        Transcribe texto de una imagen.
        
        Args:
            image: PIL.Image, path, o numpy array
            prompt: Prompt personalizado (opcional)
            language: Idioma esperado ('es', 'en', etc.)
            
        Returns:
            Tuple de (texto, score_confianza)
        """
        # Prompt optimizado para español
        if prompt is None:
            if language == "es":
                prompt = (
                    "Transcribe exactamente el texto manuscrito en español de esta imagen. "
                    "Devuelve solo el texto transcrito, sin explicaciones ni formato adicional. "
                    "Preserva la puntuación y mayúsculas originales."
                )
            else:
                prompt = "Transcribe the handwritten text in this image exactly as written."
        
        return self._backend.transcribe(image, prompt)
    
    def chat_image(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        """
        Chat con comprensión de imagen.
        
        Args:
            image: Imagen a analizar
            prompt: Pregunta o instrucción
            
        Returns:
            Respuesta del modelo
        """
        return self._backend.chat(image, prompt)
    
    def get_document_context(
        self,
        image: Any,
    ) -> dict:
        """
        Analiza el contexto del documento.
        
        Útil para entender qué tipo de documento es y qué campos esperar.
        
        Returns:
            Dict con información del documento:
            {
                'document_type': 'formulario' | 'carta' | 'nota' | ...,
                'expected_fields': ['nombre', 'dirección', ...],
                'language': 'es',
                'confidence': 0.95
            }
        """
        prompt = """Analiza esta imagen de documento y responde en JSON:
{
    "document_type": "tipo de documento (formulario, carta, nota, factura, etc.)",
    "expected_fields": ["lista", "de", "campos", "detectados"],
    "language": "idioma del texto",
    "num_lines": número aproximado de líneas de texto
}
Solo devuelve el JSON, sin explicación."""
        
        response = self.chat_image(image, prompt)
        
        # Intentar parsear JSON
        try:
            import json
            # Limpiar respuesta
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except Exception:
            return {
                "document_type": "unknown",
                "expected_fields": [],
                "language": "es",
                "raw_response": response
            }
    
    def verify_transcription(
        self,
        image: Any,
        transcription: str,
    ) -> Tuple[bool, str, float]:
        """
        Verifica si una transcripción es correcta.
        
        Args:
            image: Imagen original
            transcription: Transcripción a verificar
            
        Returns:
            Tuple de (es_correcto, transcripción_sugerida, confianza)
        """
        prompt = f"""Verifica si esta transcripción del texto manuscrito es correcta:

Transcripción a verificar: "{transcription}"

Si es correcta, responde: CORRECTO
Si hay errores, responde: INCORRECTO: [transcripción corregida]

Solo responde con una de esas dos opciones."""
        
        response = self.chat_image(image, prompt)
        
        if "CORRECTO" in response and "INCORRECTO" not in response:
            return True, transcription, 0.95
        
        elif "INCORRECTO" in response:
            # Extraer corrección
            parts = response.split("INCORRECTO:")
            if len(parts) > 1:
                corrected = parts[1].strip()
                return False, corrected, 0.85
        
        # Fallback: hacer transcripción propia
        new_text, score = self.transcribe_image(image)
        is_same = new_text.lower().strip() == transcription.lower().strip()
        return is_same, new_text if not is_same else transcription, score


# Función de conveniencia para crear adapter
def create_minicpm_adapter(
    mode: str = "mock",
    **kwargs
) -> MiniCPMAdapter:
    """
    Crea un MiniCPMAdapter con la configuración especificada.
    
    Args:
        mode: 'mock', 'local_hf', 'local_quantized', 'remote_rest'
        **kwargs: Configuración adicional
        
    Returns:
        MiniCPMAdapter configurado
    """
    return MiniCPMAdapter(mode=mode, **kwargs)
