#!/usr/bin/env python3
"""
distill_from_minicpm.py - Genera dataset enriquecido usando MiniCPM como teacher

Este script usa MiniCPM-V para generar "soft labels" que luego se usarán
para entrenar LLARRI mediante destilación de conocimiento.

El proceso:
1. Carga imágenes de tu dataset
2. MiniCPM genera transcripciones + scores de confianza
3. Guarda todo en un dataset enriquecido
4. Este dataset se usa para entrenar LLARRI

Uso:
    # Con MiniCPM local (necesita GPU potente)
    python scripts/distill_from_minicpm.py --input data/images --output data/distilled
    
    # Con servidor remoto (si tenés MiniCPM en otro server)
    python scripts/distill_from_minicpm.py --input data/images --output data/distilled --remote http://server:8080
    
    # Modo offline (usa predicciones existentes)
    python scripts/distill_from_minicpm.py --input data/images --output data/distilled --offline

Requisitos para MiniCPM local:
    - GPU con 8GB+ VRAM (o usar cuantización 4-bit para 4GB)
    - pip install transformers accelerate bitsandbytes
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DistilledSample:
    """Un sample con predicción del teacher."""
    image_path: str
    image_hash: str  # Para verificar integridad
    
    # Ground truth (si existe)
    ground_truth: Optional[str] = None
    
    # Predicciones del teacher (MiniCPM)
    teacher_prediction: str = ""
    teacher_confidence: float = 0.0
    teacher_logits: Optional[List[float]] = None  # Soft labels
    
    # Metadata
    timestamp: str = ""
    teacher_model: str = ""
    processing_time_ms: float = 0.0
    
    # Verificación adicional
    teacher_verification: Optional[str] = None  # Segunda pasada de verificación
    consensus_score: float = 0.0  # Acuerdo entre pasadas


@dataclass
class DistillationConfig:
    """Configuración para la destilación."""
    # Teacher model
    teacher_model_name: str = "openbmb/MiniCPM-V-2_6"
    use_quantization: bool = True  # 4-bit para GPUs pequeñas
    quantization_bits: int = 4
    
    # Procesamiento
    batch_size: int = 1  # MiniCPM es pesado, mejor de a uno
    max_length: int = 256
    num_workers: int = 4
    
    # Verificación (opcional, mejora calidad)
    verify_predictions: bool = True
    min_confidence_threshold: float = 0.5
    
    # Remote server (si no tenés GPU potente)
    use_remote: bool = False
    remote_url: str = ""
    
    # Prompts
    transcription_prompt: str = "Transcribe exactamente el texto manuscrito en esta imagen. Solo responde con el texto, sin explicaciones."
    verification_prompt: str = "Verifica si esta transcripción es correcta: '{text}'. Si es correcta responde CORRECTO, si no responde con la corrección."


class MiniCPMTeacher:
    """
    Teacher model usando MiniCPM-V para generar soft labels.
    
    MiniCPM-V es un modelo multimodal open source de OpenBMB.
    Puede entender imágenes y generar texto.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Inicializando MiniCPM Teacher en device: {self.device}")
    
    def load_model(self):
        """Carga el modelo MiniCPM-V."""
        if self.config.use_remote:
            logger.info(f"Usando servidor remoto: {self.config.remote_url}")
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Cargando modelo: {self.config.teacher_model_name}")
            
            # Configuración de cuantización para GPUs pequeñas
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.config.use_quantization and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if self.config.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    else:  # 8-bit
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                    
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                    logger.info(f"Usando cuantización {self.config.quantization_bits}-bit")
                    
                except ImportError:
                    logger.warning("bitsandbytes no instalado, cargando sin cuantización")
            
            # Cargar modelo
            self.model = AutoModel.from_pretrained(
                self.config.teacher_model_name,
                **load_kwargs
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.teacher_model_name,
                trust_remote_code=True
            )
            
            if not self.config.use_quantization:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Modelo MiniCPM-V cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error cargando MiniCPM-V: {e}")
            logger.info("Tip: Instala con: pip install transformers accelerate bitsandbytes")
            raise
    
    def transcribe(self, image: Image.Image) -> Tuple[str, float]:
        """
        Genera transcripción para una imagen.
        
        Returns:
            Tuple de (texto, confianza)
        """
        if self.config.use_remote:
            return self._transcribe_remote(image)
        
        if self.model is None:
            self.load_model()
        
        try:
            # Preparar mensaje para MiniCPM-V
            msgs = [{'role': 'user', 'content': [image, self.config.transcription_prompt]}]
            
            # Generar respuesta
            with torch.no_grad():
                response = self.model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.config.max_length,
                )
            
            # Extraer texto y estimar confianza
            text = response.strip() if isinstance(response, str) else str(response)
            confidence = self._estimate_confidence(text)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error en transcripción: {e}")
            return "", 0.0
    
    def verify(self, image: Image.Image, text: str) -> Tuple[str, bool]:
        """
        Verifica una transcripción existente.
        
        Returns:
            Tuple de (texto_verificado, es_correcto)
        """
        if self.config.use_remote:
            return self._verify_remote(image, text)
        
        if self.model is None:
            self.load_model()
        
        try:
            prompt = self.config.verification_prompt.format(text=text)
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            with torch.no_grad():
                response = self.model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.config.max_length,
                )
            
            response_text = response.strip() if isinstance(response, str) else str(response)
            
            if "CORRECTO" in response_text.upper():
                return text, True
            else:
                # Extraer corrección
                return response_text, False
                
        except Exception as e:
            logger.error(f"Error en verificación: {e}")
            return text, False
    
    def _transcribe_remote(self, image: Image.Image) -> Tuple[str, float]:
        """Transcribe usando servidor remoto."""
        import requests
        import base64
        from io import BytesIO
        
        # Convertir imagen a base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        try:
            response = requests.post(
                f"{self.config.remote_url}/transcribe",
                json={
                    "image": img_b64,
                    "prompt": self.config.transcription_prompt,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("text", ""), data.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error en servidor remoto: {e}")
            return "", 0.0
    
    def _verify_remote(self, image: Image.Image, text: str) -> Tuple[str, bool]:
        """Verifica usando servidor remoto."""
        import requests
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        try:
            response = requests.post(
                f"{self.config.remote_url}/verify",
                json={
                    "image": img_b64,
                    "text": text,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("text", text), data.get("is_correct", False)
            
        except Exception as e:
            logger.error(f"Error en servidor remoto: {e}")
            return text, False
    
    def _estimate_confidence(self, text: str) -> float:
        """Estima confianza basada en características del texto."""
        if not text:
            return 0.0
        
        confidence = 0.8
        
        # Penalizar texto muy corto
        if len(text) < 3:
            confidence *= 0.5
        
        # Penalizar caracteres raros
        special_count = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?-\'"áéíóúñüÁÉÍÓÚÑÜ')
        if len(text) > 0:
            special_ratio = special_count / len(text)
            confidence *= (1 - special_ratio * 0.5)
        
        # Bonus si parece español
        spanish_chars = set('áéíóúñüÁÉÍÓÚÑÜ¿¡')
        if any(c in spanish_chars for c in text):
            confidence *= 1.1
        
        return min(1.0, max(0.0, confidence))


class DistillationDatasetGenerator:
    """
    Generador de dataset para destilación.
    
    Toma imágenes y genera un dataset enriquecido con predicciones
    del teacher model (MiniCPM).
    """
    
    def __init__(
        self,
        config: DistillationConfig,
        input_dir: str,
        output_dir: str,
        ground_truth_file: Optional[str] = None,
    ):
        self.config = config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ground_truth_file = ground_truth_file
        
        # Cargar ground truth si existe
        self.ground_truth = {}
        if ground_truth_file and Path(ground_truth_file).exists():
            self._load_ground_truth()
        
        # Crear teacher
        self.teacher = MiniCPMTeacher(config)
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_ground_truth(self):
        """Carga ground truth desde archivo."""
        gt_path = Path(self.ground_truth_file)
        
        if gt_path.suffix == '.json':
            with open(gt_path) as f:
                self.ground_truth = json.load(f)
        elif gt_path.suffix == '.txt':
            # Formato: imagen.png\ttexto
            with open(gt_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.ground_truth[parts[0]] = parts[1]
        
        logger.info(f"Cargado ground truth con {len(self.ground_truth)} entradas")
    
    def _get_image_hash(self, image_path: Path) -> str:
        """Calcula hash de imagen para verificar integridad."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _find_images(self) -> List[Path]:
        """Encuentra todas las imágenes en el directorio."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        images = []
        
        for ext in extensions:
            images.extend(self.input_dir.rglob(f'*{ext}'))
            images.extend(self.input_dir.rglob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def generate(self, resume: bool = True) -> str:
        """
        Genera el dataset destilado.
        
        Args:
            resume: Si continuar desde donde se quedó
            
        Returns:
            Path al archivo de salida
        """
        import time
        
        # Encontrar imágenes
        images = self._find_images()
        logger.info(f"Encontradas {len(images)} imágenes")
        
        if not images:
            logger.error(f"No se encontraron imágenes en {self.input_dir}")
            return ""
        
        # Archivo de salida
        output_file = self.output_dir / "distilled_dataset.jsonl"
        
        # Cargar progreso existente si resume=True
        processed = set()
        if resume and output_file.exists():
            with open(output_file) as f:
                for line in f:
                    sample = json.loads(line)
                    processed.add(sample['image_path'])
            logger.info(f"Continuando desde {len(processed)} samples procesados")
        
        # Procesar imágenes
        samples = []
        mode = 'a' if resume else 'w'
        
        with open(output_file, mode) as f:
            for image_path in tqdm(images, desc="Generando dataset"):
                rel_path = str(image_path.relative_to(self.input_dir))
                
                # Saltar si ya procesado
                if rel_path in processed:
                    continue
                
                try:
                    # Cargar imagen
                    image = Image.open(image_path).convert('RGB')
                    
                    # Generar predicción del teacher
                    start_time = time.time()
                    text, confidence = self.teacher.transcribe(image)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Verificar si está configurado
                    verification = None
                    consensus = confidence
                    
                    if self.config.verify_predictions and confidence >= self.config.min_confidence_threshold:
                        verified_text, is_correct = self.teacher.verify(image, text)
                        verification = verified_text
                        
                        if is_correct:
                            consensus = min(1.0, confidence * 1.2)
                        else:
                            # Usar texto verificado
                            text = verified_text
                            consensus = confidence * 0.8
                    
                    # Crear sample
                    sample = DistilledSample(
                        image_path=rel_path,
                        image_hash=self._get_image_hash(image_path),
                        ground_truth=self.ground_truth.get(rel_path),
                        teacher_prediction=text,
                        teacher_confidence=confidence,
                        timestamp=datetime.now().isoformat(),
                        teacher_model=self.config.teacher_model_name,
                        processing_time_ms=processing_time,
                        teacher_verification=verification,
                        consensus_score=consensus,
                    )
                    
                    # Guardar
                    f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
                    f.flush()
                    
                    samples.append(sample)
                    
                except Exception as e:
                    logger.error(f"Error procesando {image_path}: {e}")
                    continue
        
        # Generar estadísticas
        self._generate_stats(samples, output_file)
        
        logger.info(f"Dataset generado: {output_file}")
        logger.info(f"Total samples: {len(samples)}")
        
        return str(output_file)
    
    def _generate_stats(self, samples: List[DistilledSample], output_file: Path):
        """Genera estadísticas del dataset."""
        if not samples:
            return
        
        stats = {
            "total_samples": len(samples),
            "samples_with_ground_truth": sum(1 for s in samples if s.ground_truth),
            "avg_confidence": sum(s.teacher_confidence for s in samples) / len(samples),
            "avg_processing_time_ms": sum(s.processing_time_ms for s in samples) / len(samples),
            "high_confidence_samples": sum(1 for s in samples if s.teacher_confidence > 0.8),
            "low_confidence_samples": sum(1 for s in samples if s.teacher_confidence < 0.5),
            "verified_samples": sum(1 for s in samples if s.teacher_verification),
            "teacher_model": self.config.teacher_model_name,
            "generation_date": datetime.now().isoformat(),
        }
        
        stats_file = output_file.parent / "distillation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Estadísticas guardadas en {stats_file}")
        logger.info(f"Confianza promedio: {stats['avg_confidence']:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset destilado usando MiniCPM como teacher"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directorio con imágenes de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Directorio de salida para dataset destilado"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help="Archivo con ground truth (opcional)"
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        help="URL de servidor MiniCPM remoto"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="No usar cuantización (requiere más VRAM)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="No verificar predicciones (más rápido)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="No continuar desde progreso anterior"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/MiniCPM-V-2_6",
        help="Nombre del modelo teacher"
    )
    
    args = parser.parse_args()
    
    # Crear configuración
    config = DistillationConfig(
        teacher_model_name=args.model,
        use_quantization=not args.no_quantization,
        verify_predictions=not args.no_verify,
        use_remote=args.remote is not None,
        remote_url=args.remote or "",
    )
    
    # Crear generador
    generator = DistillationDatasetGenerator(
        config=config,
        input_dir=args.input,
        output_dir=args.output,
        ground_truth_file=args.ground_truth,
    )
    
    # Generar
    output_file = generator.generate(resume=not args.no_resume)
    
    if output_file:
        print(f"\n✅ Dataset generado: {output_file}")
        print(f"\nPróximo paso:")
        print(f"  python scripts/run_distillation.py --dataset {output_file}")


if __name__ == "__main__":
    main()
