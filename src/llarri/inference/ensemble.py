"""
ensemble.py - Sistema de Inferencia Ensemble

Combina el selector de estilo con múltiples expertos para hacer inferencia óptima:
1. El selector clasifica el estilo de la imagen
2. Se selecciona el experto apropiado
3. El experto realiza la predicción

Estrategias soportadas:
- Routing simple: Selector → Experto único
- Routing con confianza: Si baja confianza, usar modelo base
- Ensemble voting: Combinar predicciones de múltiples expertos
- Fallback cascada: Intentar experto específico, luego base

Uso:
    ensemble = EnsembleInference(
        selector_path="outputs/selector/best.ckpt",
        expert_paths={
            "expert_es_mayores": "outputs/experts/es_mayores.ckpt",
            "expert_latam_jovenes": "outputs/experts/latam_jovenes.ckpt"
        },
        base_model_path="outputs/base/best.ckpt"
    )
    
    result = ensemble.predict("image.jpg")
    # result = {
    #     "text": "Hola mundo",
    #     "confidence": 0.95,
    #     "expert_used": "expert_es_mayores",
    #     "selector_confidence": 0.87
    # }
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image

from llarri.models.selector_style import (
    StyleSelector, 
    ExpertRouter,
    SIMPLE_CLASS_TO_EXPERT,
    SIMPLE_CLASS_NAMES
)
from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.training.finetune_expert import ExpertModel
from llarri.data.transforms import get_transforms


class EnsembleInference:
    """
    Sistema de inferencia ensemble que combina selector + expertos.
    
    Arquitectura:
        Image → StyleSelector → ExpertRouter → Expert → Text
                                              ↘ Base Model (fallback)
    """
    def __init__(
        self,
        selector_path: str,
        expert_paths: Dict[str, str],
        base_model_path: str,
        class_to_expert: Optional[Dict[int, str]] = None,
        class_names: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.7,
        device: str = "auto"
    ):
        """
        Args:
            selector_path: Checkpoint del selector entrenado
            expert_paths: {"expert_name": "path/to/checkpoint"}
            base_model_path: Checkpoint del modelo base
            class_to_expert: Mapeo {class_id: expert_name}
            class_names: Nombres de clases {class_id: "name"}
            confidence_threshold: Umbral de confianza para usar experto (vs base)
            device: "auto", "cuda", "cpu"
        """
        self.confidence_threshold = confidence_threshold
        
        # Determinar device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Inicializando EnsembleInference en {self.device}")
        
        # Cargar selector
        print(f"📂 Cargando selector desde {selector_path}")
        from llarri.training.train_selector import StyleSelectorModule
        selector_module = StyleSelectorModule.load_from_checkpoint(selector_path)
        self.selector = selector_module.selector.to(self.device)
        self.selector.eval()
        
        # Configurar router
        self.class_to_expert = class_to_expert or SIMPLE_CLASS_TO_EXPERT
        self.class_names = class_names or SIMPLE_CLASS_NAMES
        self.router = ExpertRouter(self.class_to_expert, self.class_names)
        
        # Cargar modelo base
        print(f"📂 Cargando modelo base desde {base_model_path}")
        self.base_model = LlarriBaseModel.load_from_checkpoint(base_model_path)
        self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        
        # Cargar expertos
        print(f"📂 Cargando {len(expert_paths)} expertos...")
        self.experts = {}
        for expert_name, expert_path in expert_paths.items():
            print(f"   - {expert_name}: {expert_path}")
            expert = ExpertModel.load_from_checkpoint(expert_path)
            expert = expert.to(self.device)
            expert.eval()
            self.experts[expert_name] = expert
        
        # Transformaciones de imagen
        self.transform = get_transforms(augment=False)
        
        print("✅ Ensemble inicializado correctamente")
        print(f"   Expertos disponibles: {list(self.experts.keys())}")
        print(f"   Mapeo clases → expertos: {self.class_to_expert}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesar imagen para inferencia.
        
        Args:
            image: Path a imagen, PIL Image, o Tensor ya procesado
        
        Returns:
            Tensor (1, C, H, W) listo para modelo
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image = self.transform(image)
        
        # Asegurar batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def select_expert(
        self, 
        image: torch.Tensor,
        return_confidence: bool = True
    ) -> Union[str, Tuple[str, float, int]]:
        """
        Determina qué experto usar para una imagen.
        
        Args:
            image: Tensor (1, C, H, W)
            return_confidence: Si retornar también confianza y class_id
        
        Returns:
            expert_name (str) o (expert_name, confidence, class_id)
        """
        with torch.no_grad():
            # Obtener probabilidades del selector
            probs = self.selector.predict_proba(image)  # (1, num_classes)
            max_prob, class_id = torch.max(probs, dim=-1)
            
            max_prob = max_prob.item()
            class_id = class_id.item()
        
        # Routing con confianza
        if max_prob >= self.confidence_threshold:
            expert_name = self.class_to_expert.get(class_id, "base")
        else:
            expert_name = "base"  # Fallback si baja confianza
        
        if return_confidence:
            return expert_name, max_prob, class_id
        return expert_name
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        max_length: int = 128,
        num_beams: int = 4,
        return_details: bool = True
    ) -> Union[str, Dict]:
        """
        Realiza predicción completa: selector → experto → texto.
        
        Args:
            image: Imagen de entrada
            max_length: Longitud máxima de generación
            num_beams: Beams para beam search
            return_details: Si retornar detalles adicionales
        
        Returns:
            Si return_details=False: texto predicho (str)
            Si return_details=True: dict con texto, confianza, experto usado, etc.
        """
        # Preprocesar
        image_tensor = self.preprocess_image(image)
        
        # Seleccionar experto
        expert_name, selector_confidence, class_id = self.select_expert(
            image_tensor, return_confidence=True
        )
        
        # Obtener modelo apropiado
        if expert_name == "base":
            model = self.base_model
        else:
            model = self.experts.get(expert_name, self.base_model)
        
        # Generar texto
        with torch.no_grad():
            # Preparar batch para el modelo
            batch = {'pixel_values': image_tensor}
            
            # Generar con el modelo
            generated_ids = model.generate(
                batch,
                max_length=max_length,
                num_beams=num_beams
            )
            
            # Decodificar
            text = model.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
        
        # Retornar resultado
        if not return_details:
            return text
        
        return {
            "text": text,
            "expert_used": expert_name,
            "class_name": self.class_names.get(class_id, f"class_{class_id}"),
            "class_id": class_id,
            "selector_confidence": selector_confidence,
            "used_fallback": expert_name == "base" and selector_confidence < self.confidence_threshold
        }
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        max_length: int = 128,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Predicción en batch para múltiples imágenes.
        
        Agrupa imágenes por experto para mayor eficiencia.
        """
        results = []
        
        # Primera pasada: clasificar todas las imágenes
        image_tensors = [self.preprocess_image(img) for img in images]
        all_tensors = torch.cat(image_tensors, dim=0)  # (N, C, H, W)
        
        with torch.no_grad():
            # Clasificar todas las imágenes
            probs = self.selector.predict_proba(all_tensors)  # (N, num_classes)
            max_probs, class_ids = torch.max(probs, dim=-1)
        
        # Agrupar por experto
        expert_groups = {}
        for idx, (class_id, confidence) in enumerate(zip(class_ids, max_probs)):
            class_id = int(class_id.item())
            confidence = float(confidence.item())
            
            if confidence >= self.confidence_threshold:
                expert_name = self.class_to_expert.get(class_id, "base")
            else:
                expert_name = "base"
            
            if expert_name not in expert_groups:
                expert_groups[expert_name] = []
            
            expert_groups[expert_name].append({
                "idx": idx,
                "image_tensor": image_tensors[idx],
                "class_id": class_id,
                "confidence": confidence
            })
        
        # Procesar por experto
        expert_results = {}
        for expert_name, items in expert_groups.items():
            print(f"🔄 Procesando {len(items)} imágenes con {expert_name}")
            
            # Seleccionar modelo
            if expert_name == "base":
                model = self.base_model
            else:
                model = self.experts.get(expert_name, self.base_model)
            
            # Procesar en mini-batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i+batch_size]
                
                # Preparar batch
                batch_tensors = torch.cat([item["image_tensor"] for item in batch_items], dim=0)
                batch = {'pixel_values': batch_tensors}
                
                # Generar
                with torch.no_grad():
                    generated_ids = model.generate(batch, max_length=max_length, num_beams=num_beams)
                    texts = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Guardar resultados
                for item, text in zip(batch_items, texts):
                    expert_results[item["idx"]] = {
                        "text": text,
                        "expert_used": expert_name,
                        "class_id": item["class_id"],
                        "class_name": self.class_names.get(item["class_id"], f"class_{item['class_id']}"),
                        "selector_confidence": item["confidence"]
                    }
        
        # Ordenar resultados por índice original
        results = [expert_results[i] for i in range(len(images))]
        
        return results


class VotingEnsemble(EnsembleInference):
    """
    Ensemble que usa voting: múltiples expertos votan por la mejor predicción.
    
    Útil cuando no hay un selector, o se quiere combinar múltiples expertos.
    """
    def __init__(
        self,
        expert_paths: Dict[str, str],
        base_model_path: str,
        voting_strategy: str = "majority",
        device: str = "auto"
    ):
        """
        Args:
            voting_strategy: "majority" (voto mayoritario), "confidence" (max confianza)
        """
        self.voting_strategy = voting_strategy
        
        # Determinar device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🗳️  Inicializando VotingEnsemble ({voting_strategy}) en {self.device}")
        
        # Cargar modelo base
        self.base_model = LlarriBaseModel.load_from_checkpoint(base_model_path)
        self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        
        # Cargar expertos
        self.experts = {}
        for expert_name, expert_path in expert_paths.items():
            expert = ExpertModel.load_from_checkpoint(expert_path)
            expert = expert.to(self.device)
            expert.eval()
            self.experts[expert_name] = expert
        
        self.transform = get_transforms(augment=False)
        
        print(f"✅ VotingEnsemble inicializado con {len(self.experts)} expertos")
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        max_length: int = 128,
        num_beams: int = 4,
        return_details: bool = True
    ) -> Union[str, Dict]:
        """
        Predice usando voting de múltiples expertos.
        """
        image_tensor = self.preprocess_image(image)
        batch = {'pixel_values': image_tensor}
        
        # Obtener predicciones de todos los expertos
        predictions = {}
        
        with torch.no_grad():
            # Base model
            generated_ids = self.base_model.generate(batch, max_length=max_length, num_beams=num_beams)
            text = self.base_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions["base"] = text
            
            # Experts
            for expert_name, expert in self.experts.items():
                generated_ids = expert.generate(batch, max_length=max_length, num_beams=num_beams)
                text = expert.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                predictions[expert_name] = text
        
        # Voting
        if self.voting_strategy == "majority":
            # Voto mayoritario
            from collections import Counter
            vote_counts = Counter(predictions.values())
            final_text = vote_counts.most_common(1)[0][0]
        
        elif self.voting_strategy == "confidence":
            # Seleccionar por confianza (usar primera predicción como proxy)
            final_text = predictions["base"]  # Simplificación
        
        else:
            final_text = predictions["base"]
        
        if not return_details:
            return final_text
        
        return {
            "text": final_text,
            "all_predictions": predictions,
            "voting_strategy": self.voting_strategy
        }


# =============================================================================
# UTILIDADES
# =============================================================================

def load_ensemble_from_config(config_path: str, device: str = "auto") -> EnsembleInference:
    """
    Carga ensemble desde archivo de configuración YAML.
    
    Formato esperado:
    ```yaml
    selector:
      path: outputs/selector/best.ckpt
      confidence_threshold: 0.7
    
    base_model:
      path: outputs/base/best.ckpt
    
    experts:
      expert_es_mayores: outputs/experts/es_mayores.ckpt
      expert_latam_jovenes: outputs/experts/latam_jovenes.ckpt
    
    class_to_expert:
      0: expert_es_mayores
      1: expert_latam_jovenes
      2: base
    ```
    """
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    ensemble = EnsembleInference(
        selector_path=config['selector']['path'],
        expert_paths=config['experts'],
        base_model_path=config['base_model']['path'],
        class_to_expert=config.get('class_to_expert'),
        confidence_threshold=config['selector'].get('confidence_threshold', 0.7),
        device=device
    )
    
    return ensemble

