"""
Integración de MiniCPM-V como verificador/complemento para LLARRI OCR.

MiniCPM-V 4.5 es un modelo multimodal de 8B parámetros que supera a GPT-4o
en OCR y comprensión de documentos. Se puede usar para:

1. Verificar predicciones de LLARRI
2. Entender contexto de documentos completos
3. OCR de respaldo para casos difíciles
4. Extracción estructurada de formularios

Repositorio: https://github.com/OpenBMB/MiniCPM-V

Uso:
    verifier = MiniCPMVerifier()
    
    # Verificar una predicción
    is_correct, suggestion = verifier.verify("Holla mundo", image)
    
    # OCR directo
    text = verifier.ocr(image)
    
    # Entender documento
    fields = verifier.extract_fields(image)
"""

from __future__ import annotations

import warnings
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import re

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# MiniCPM-V imports (opcional)
HAS_MINICPM = False
try:
    from transformers import AutoModel, AutoTokenizer
    # Verificar si el modelo está disponible
    HAS_MINICPM = True
except ImportError:
    pass


class MiniCPMVerifier:
    """
    Verificador OCR usando MiniCPM-V.
    
    MiniCPM-V es un modelo Vision-Language que puede:
    - Leer texto en imágenes con alta precisión
    - Entender el contexto de documentos
    - Extraer información estructurada
    
    Se puede usar como complemento a LLARRI para mejorar la precisión.
    """
    
    # Modelos disponibles ordenados por tamaño/capacidad
    AVAILABLE_MODELS = {
        'minicpm-v-4.5': 'openbmb/MiniCPM-V-4_5',           # Mejor calidad, 18GB GPU
        'minicpm-v-4.5-int4': 'openbmb/MiniCPM-V-4_5-int4', # Cuantizado, 9GB GPU
        'minicpm-o-2.6': 'openbmb/MiniCPM-o-2_6',           # Con audio, 18GB GPU
        'minicpm-o-2.6-int4': 'openbmb/MiniCPM-o-2_6-int4', # Cuantizado, 9GB GPU
    }
    
    # Prompts optimizados para OCR en español
    PROMPTS = {
        'ocr_simple': 'Transcribe exactamente el texto manuscrito en esta imagen. Solo responde con el texto, sin explicaciones.',
        'ocr_detailed': 'Lee cuidadosamente el texto manuscrito en esta imagen y transcríbelo exactamente como está escrito, preservando mayúsculas, puntuación y acentos.',
        'verify': 'El texto "{text}" fue reconocido de esta imagen. ¿Es correcto? Si hay errores, proporciona la corrección.',
        'extract_fields': 'Esta imagen contiene un formulario o documento. Extrae todos los campos y sus valores en formato JSON.',
        'document_type': '¿Qué tipo de documento es este? Describe brevemente su estructura y contenido.',
        'handwriting_quality': 'Evalúa la calidad de la escritura manuscrita en esta imagen (legible, parcialmente legible, difícil de leer).',
    }
    
    def __init__(
        self,
        model_name: str = 'minicpm-v-4.5-int4',  # int4 por defecto para menor uso de VRAM
        device: Optional[str] = None,
        torch_dtype: Optional[str] = 'bfloat16',
        load_on_init: bool = False,  # No cargar modelo hasta que se use
    ):
        """
        Inicializa el verificador MiniCPM-V.
        
        Args:
            model_name: Nombre del modelo a usar (ver AVAILABLE_MODELS)
            device: Dispositivo ('cuda', 'cpu', 'mps'). Auto-detecta si None.
            torch_dtype: Tipo de datos ('bfloat16', 'float16', 'float32')
            load_on_init: Si True, carga el modelo inmediatamente
        """
        self.model_name = model_name
        self.model_id = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        
        # Auto-detectar dispositivo
        if device is None:
            if HAS_TORCH and torch.cuda.is_available():
                self.device = 'cuda'
            elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        if load_on_init:
            self._load_model()
    
    @property
    def is_available(self) -> bool:
        """Verifica si MiniCPM-V está disponible."""
        return HAS_MINICPM and HAS_TORCH and HAS_PIL
    
    @property
    def model(self):
        """Carga el modelo bajo demanda."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Carga el tokenizer bajo demanda."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self):
        """Carga el modelo y tokenizer de HuggingFace."""
        if not self.is_available:
            raise RuntimeError(
                "MiniCPM-V no está disponible. Instala las dependencias:\n"
                "pip install torch transformers pillow"
            )
        
        print(f"[MiniCPM] Cargando modelo {self.model_id}...")
        print(f"[MiniCPM] Dispositivo: {self.device}")
        
        # Determinar dtype
        if self.torch_dtype == 'bfloat16':
            dtype = torch.bfloat16
        elif self.torch_dtype == 'float16':
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        try:
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                attn_implementation='sdpa',  # Más eficiente que 'eager'
                torch_dtype=dtype,
            )
            self._model = self._model.eval()
            
            if self.device == 'cuda':
                self._model = self._model.cuda()
            elif self.device == 'mps':
                self._model = self._model.to('mps')
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            print(f"[MiniCPM] Modelo cargado exitosamente")
            
        except Exception as e:
            raise RuntimeError(f"Error cargando MiniCPM-V: {e}")
    
    def _prepare_image(self, image: Union[str, Path, 'Image.Image']) -> 'Image.Image':
        """Prepara la imagen para el modelo."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            return image.convert('RGB')
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
    
    def chat(
        self,
        image: Union[str, Path, 'Image.Image'],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """
        Envía una imagen y prompt al modelo.
        
        Args:
            image: Imagen (path o PIL Image)
            prompt: Pregunta o instrucción
            max_new_tokens: Máximo de tokens a generar
            temperature: Temperatura de sampling
            
        Returns:
            Respuesta del modelo
        """
        img = self._prepare_image(image)
        
        msgs = [{'role': 'user', 'content': [img, prompt]}]
        
        with torch.no_grad():
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        
        return answer
    
    def ocr(
        self,
        image: Union[str, Path, 'Image.Image'],
        detailed: bool = False,
    ) -> str:
        """
        Realiza OCR usando MiniCPM-V.
        
        Args:
            image: Imagen a procesar
            detailed: Si True, usa prompt más detallado
            
        Returns:
            Texto reconocido
        """
        prompt = self.PROMPTS['ocr_detailed'] if detailed else self.PROMPTS['ocr_simple']
        return self.chat(image, prompt)
    
    def verify(
        self,
        text: str,
        image: Union[str, Path, 'Image.Image'],
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica si un texto OCR es correcto.
        
        Args:
            text: Texto a verificar
            image: Imagen original
            
        Returns:
            Tupla (es_correcto, sugerencia_si_incorrecto)
        """
        prompt = self.PROMPTS['verify'].format(text=text)
        response = self.chat(image, prompt)
        
        # Parsear respuesta
        response_lower = response.lower()
        
        if 'correcto' in response_lower and 'incorrecto' not in response_lower:
            return True, None
        elif 'incorrecto' in response_lower or 'error' in response_lower:
            # Intentar extraer la corrección
            # Buscar patrones como "debería ser: X" o "corrección: X"
            patterns = [
                r'deber[íi]a ser[:\s]+["\']?([^"\']+)["\']?',
                r'correcci[oó]n[:\s]+["\']?([^"\']+)["\']?',
                r'es[:\s]+["\']?([^"\']+)["\']?',
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return False, match.group(1).strip()
            
            # Si no encontramos corrección específica, hacer OCR directo
            correction = self.ocr(image)
            return False, correction
        else:
            # No está claro, asumir que está bien
            return True, None
    
    def extract_fields(
        self,
        image: Union[str, Path, 'Image.Image'],
    ) -> Dict[str, Any]:
        """
        Extrae campos estructurados de un documento.
        
        Args:
            image: Imagen del documento
            
        Returns:
            Diccionario con campos extraídos
        """
        prompt = self.PROMPTS['extract_fields']
        response = self.chat(image, prompt, max_new_tokens=1024)
        
        # Intentar parsear JSON
        try:
            import json
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Si falla JSON, devolver como texto
        return {'raw_response': response}
    
    def analyze_document(
        self,
        image: Union[str, Path, 'Image.Image'],
    ) -> Dict[str, Any]:
        """
        Análisis completo de un documento.
        
        Args:
            image: Imagen del documento
            
        Returns:
            Diccionario con análisis completo
        """
        img = self._prepare_image(image)
        
        # 1. Identificar tipo de documento
        doc_type = self.chat(img, self.PROMPTS['document_type'])
        
        # 2. Evaluar calidad de escritura
        quality = self.chat(img, self.PROMPTS['handwriting_quality'])
        
        # 3. Extraer texto
        text = self.ocr(img, detailed=True)
        
        # 4. Extraer campos si es formulario
        fields = {}
        if 'formulario' in doc_type.lower() or 'form' in doc_type.lower():
            fields = self.extract_fields(img)
        
        return {
            'document_type': doc_type,
            'handwriting_quality': quality,
            'full_text': text,
            'fields': fields,
        }


class EnsembleOCR:
    """
    Ensemble que combina LLARRI y MiniCPM-V para mejor precisión.
    
    Estrategia:
    1. LLARRI hace OCR primario (rápido, especializado)
    2. Si confianza baja → MiniCPM-V verifica/corrige
    3. Language Model re-rankea candidatos
    
    Uso:
        ensemble = EnsembleOCR(llarri_model, minicpm_verifier, language_model)
        text, confidence = ensemble.predict(image)
    """
    
    def __init__(
        self,
        llarri_model,                    # LlarriBaseModel
        minicpm_verifier: Optional[MiniCPMVerifier] = None,
        language_model = None,           # SpanishLanguageModel
        confidence_threshold: float = 0.8,  # Umbral para verificación
        always_verify: bool = False,     # Siempre usar MiniCPM-V
    ):
        """
        Inicializa el ensemble.
        
        Args:
            llarri_model: Modelo LLARRI (TrOCR fine-tuned)
            minicpm_verifier: Verificador MiniCPM-V (opcional)
            language_model: Modelo de lenguaje español (opcional)
            confidence_threshold: Umbral de confianza para verificar
            always_verify: Si True, siempre verifica con MiniCPM-V
        """
        self.llarri = llarri_model
        self.minicpm = minicpm_verifier
        self.lm = language_model
        self.confidence_threshold = confidence_threshold
        self.always_verify = always_verify
    
    def predict(
        self,
        image: Union[str, Path, 'Image.Image'],
        return_details: bool = False,
    ) -> Union[str, Tuple[str, float], Dict[str, Any]]:
        """
        Predice texto usando el ensemble.
        
        Args:
            image: Imagen a procesar
            return_details: Si True, retorna detalles del proceso
            
        Returns:
            Texto predicho, o (texto, confianza), o dict con detalles
        """
        details = {
            'llarri_text': None,
            'llarri_confidence': None,
            'minicpm_text': None,
            'minicpm_verified': None,
            'final_text': None,
            'final_confidence': None,
            'method': 'llarri_only',
        }
        
        # 1. Predicción primaria con LLARRI
        if hasattr(self.llarri, 'predict_with_confidence'):
            llarri_text, llarri_conf = self.llarri.predict_with_confidence(image)
        else:
            llarri_text = self.llarri.predict(image)
            llarri_conf = 0.9  # Asumir alta confianza si no hay método
        
        details['llarri_text'] = llarri_text
        details['llarri_confidence'] = llarri_conf
        
        # 2. Verificar con MiniCPM-V si es necesario
        need_verification = (
            self.always_verify or 
            llarri_conf < self.confidence_threshold
        )
        
        if need_verification and self.minicpm is not None:
            try:
                is_correct, suggestion = self.minicpm.verify(llarri_text, image)
                details['minicpm_verified'] = is_correct
                
                if not is_correct and suggestion:
                    details['minicpm_text'] = suggestion
                    details['method'] = 'minicpm_corrected'
                    
                    # Usar Language Model para decidir entre candidatos
                    if self.lm is not None:
                        candidates = [llarri_text, suggestion]
                        ranked = self.lm.rerank_candidates(candidates)
                        final_text = ranked[0]
                        details['method'] = 'lm_ranked'
                    else:
                        final_text = suggestion
                else:
                    final_text = llarri_text
                    details['method'] = 'minicpm_verified'
                    
            except Exception as e:
                warnings.warn(f"MiniCPM verification failed: {e}")
                final_text = llarri_text
        else:
            final_text = llarri_text
        
        # 3. Aplicar Language Model para corrección final
        if self.lm is not None:
            final_text = self.lm.correct_text(final_text)
            details['method'] += '+lm_corrected'
        
        details['final_text'] = final_text
        details['final_confidence'] = self._estimate_confidence(details)
        
        if return_details:
            return details
        else:
            return final_text
    
    def _estimate_confidence(self, details: Dict[str, Any]) -> float:
        """Estima la confianza final basada en el proceso."""
        base_conf = details.get('llarri_confidence', 0.5)
        
        # Boost de confianza si MiniCPM verificó
        if details.get('minicpm_verified') is True:
            return min(0.99, base_conf + 0.1)
        
        # Reducir si hubo corrección
        if details.get('method', '').startswith('minicpm_corrected'):
            return 0.85
        
        return base_conf


def create_minicpm_verifier(
    model_size: str = 'int4',  # 'full' o 'int4'
    load_now: bool = False,
) -> Optional[MiniCPMVerifier]:
    """
    Factory function para crear verificador MiniCPM-V.
    
    Args:
        model_size: 'full' (18GB VRAM) o 'int4' (9GB VRAM)
        load_now: Si True, carga el modelo inmediatamente
        
    Returns:
        MiniCPMVerifier o None si no está disponible
    """
    if not HAS_MINICPM or not HAS_TORCH:
        print("[MiniCPM] No disponible. Instala: pip install torch transformers")
        return None
    
    model_name = 'minicpm-v-4.5' if model_size == 'full' else 'minicpm-v-4.5-int4'
    
    return MiniCPMVerifier(
        model_name=model_name,
        load_on_init=load_now,
    )


# Ejemplo de uso
if __name__ == "__main__":
    print("=== MiniCPM-V Verifier Demo ===")
    print()
    
    # Verificar disponibilidad
    print(f"PyTorch disponible: {HAS_TORCH}")
    print(f"PIL disponible: {HAS_PIL}")
    print(f"MiniCPM disponible: {HAS_MINICPM}")
    
    if HAS_MINICPM:
        print()
        print("Modelos disponibles:")
        for name, model_id in MiniCPMVerifier.AVAILABLE_MODELS.items():
            print(f"  - {name}: {model_id}")
        
        print()
        print("Para usar:")
        print("  verifier = MiniCPMVerifier('minicpm-v-4.5-int4')")
        print("  text = verifier.ocr('imagen.jpg')")
        print("  is_ok, correction = verifier.verify('texto', 'imagen.jpg')")
