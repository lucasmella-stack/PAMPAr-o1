"""
selector_style.py - Clasificador de Estilo de Escritura

Este módulo implementa clasificadores que determinan qué experto usar
basándose en características de la imagen de escritura manuscrita.

Estrategias de clasificación:
1. Simple CNN: Clasificador convolucional ligero
2. ViT-based: Usa features del encoder ViT existente
3. Multi-task: Clasifica múltiples atributos (edad, región, formalidad)

El selector se entrena en un dataset etiquetado con:
- Estilo de escritura (fluida, temblorosa, irregular)
- Grupo demográfico (joven, adulto, mayor)
- Región (España, Latinoamérica)
- Formalidad (formal, informal)

Uso:
    # Clasificador simple
    selector = StyleSelectorCNN(num_classes=3)
    
    # Basado en ViT encoder
    selector = StyleSelectorViT(encoder, num_classes=3)
    
    # Multi-task
    selector = MultiTaskStyleSelector(encoder, task_configs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl


class StyleSelectorCNN(nn.Module):
    """
    Clasificador CNN ligero para determinar estilo de escritura.
    
    Arquitectura simple: Conv -> Pool -> Conv -> Pool -> FC
    Rápido e independiente del modelo principal.
    """
    def __init__(
        self, 
        num_classes: int = 3,
        input_channels: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Feature extractor convolucional
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ StyleSelectorCNN creado con {num_classes} clases")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Imagen (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


class StyleSelectorViT(nn.Module):
    """
    Clasificador basado en features del encoder ViT existente.
    
    Ventajas:
    - Reutiliza encoder ya entrenado
    - No requiere entrenar desde cero
    - Features más ricas
    
    Usa el [CLS] token o pooled output del ViT.
    """
    def __init__(
        self,
        vit_encoder,
        num_classes: int = 3,
        freeze_encoder: bool = True,
        use_cls_token: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        
        # Encoder ViT (puede congelarse)
        self.encoder = vit_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Dimensión de features del ViT
        hidden_size = self.encoder.hidden_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"✅ StyleSelectorViT creado con {num_classes} clases")
        print(f"   Encoder congelado: {freeze_encoder}")
        print(f"   Parámetros entrenables: {trainable:,} / {total:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Imagen (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Obtener features del ViT
        encoder_output = self.encoder(x)  # (B, seq_len, hidden_size)
        
        # Usar [CLS] token (primer token) o promedio
        if self.use_cls_token:
            features = encoder_output[:, 0, :]  # (B, hidden_size)
        else:
            features = encoder_output.mean(dim=1)  # (B, hidden_size)
        
        # Clasificar
        logits = self.classifier(features)
        return logits


class MultiTaskStyleSelector(nn.Module):
    """
    Selector multi-task que clasifica múltiples atributos simultáneamente.
    
    Tasks:
    - age_group: joven/adulto/mayor
    - region: españa/latam/otro
    - formality: formal/informal
    - writing_quality: clara/irregular/dificil
    
    Comparte encoder, tiene heads separados por task.
    """
    def __init__(
        self,
        vit_encoder,
        task_configs: Dict[str, int],  # {"task_name": num_classes}
        freeze_encoder: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.task_configs = task_configs
        
        # Encoder compartido
        self.encoder = vit_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size = self.encoder.hidden_size
        
        # Shared feature projection
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
        
        print(f"✅ MultiTaskStyleSelector creado")
        print(f"   Tasks: {list(task_configs.keys())}")
        print(f"   Encoder congelado: {freeze_encoder}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Imagen (B, C, H, W)
        Returns:
            Dict con logits por task: {"task_name": (B, num_classes)}
        """
        # Obtener features del encoder
        encoder_output = self.encoder(x)
        features = encoder_output[:, 0, :]  # [CLS] token
        
        # Proyección compartida
        shared_features = self.shared_projection(features)
        
        # Clasificar para cada task
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
        
        return outputs


class StyleSelector(nn.Module):
    """
    Wrapper unificado para diferentes tipos de selectores.
    
    Factory class que crea el tipo apropiado.
    """
    def __init__(
        self,
        selector_type: str = "cnn",
        num_classes: int = 3,
        vit_encoder = None,
        task_configs: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.selector_type = selector_type
        
        if selector_type == "cnn":
            self.selector = StyleSelectorCNN(num_classes=num_classes, **kwargs)
        
        elif selector_type == "vit":
            if vit_encoder is None:
                raise ValueError("vit_encoder required for selector_type='vit'")
            self.selector = StyleSelectorViT(
                vit_encoder=vit_encoder,
                num_classes=num_classes,
                **kwargs
            )
        
        elif selector_type == "multitask":
            if vit_encoder is None:
                raise ValueError("vit_encoder required for selector_type='multitask'")
            if task_configs is None:
                raise ValueError("task_configs required for selector_type='multitask'")
            self.selector = MultiTaskStyleSelector(
                vit_encoder=vit_encoder,
                task_configs=task_configs,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown selector_type: {selector_type}")
    
    def forward(self, x: torch.Tensor):
        return self.selector.forward(x)
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice la clase con mayor probabilidad.
        
        Returns:
            class_ids: (B,) tensor con índices de clase
        """
        logits = self.forward(x)
        
        if isinstance(logits, dict):
            # Multi-task: retornar primera task
            first_task = list(logits.keys())[0]
            logits = logits[first_task]
        
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades de cada clase.
        
        Returns:
            probs: (B, num_classes) con softmax
        """
        logits = self.forward(x)
        
        if isinstance(logits, dict):
            # Multi-task: retornar primera task
            first_task = list(logits.keys())[0]
            logits = logits[first_task]
        
        return F.softmax(logits, dim=-1)


# =============================================================================
# MAPPING DE CLASES A EXPERTOS
# =============================================================================

class ExpertRouter:
    """
    Enruta imágenes al experto apropiado basado en predicción del selector.
    
    Mapeo configurable de clase → experto.
    """
    def __init__(
        self,
        class_to_expert: Dict[int, str],
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            class_to_expert: Mapeo {class_id: expert_name}
            class_names: Nombres legibles de clases (opcional)
        """
        self.class_to_expert = class_to_expert
        self.class_names = class_names or {i: f"class_{i}" for i in class_to_expert.keys()}
    
    def route(self, class_predictions: torch.Tensor) -> List[str]:
        """
        Determina qué experto usar para cada muestra.
        
        Args:
            class_predictions: (B,) tensor con class_ids
        
        Returns:
            Lista de nombres de expertos, uno por muestra
        """
        expert_names = []
        for class_id in class_predictions.cpu().numpy():
            expert_name = self.class_to_expert.get(int(class_id), "base")
            expert_names.append(expert_name)
        return expert_names
    
    def route_with_confidence(
        self,
        class_probs: torch.Tensor,
        confidence_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Enruta con verificación de confianza.
        
        Si la confianza es baja, usa modelo base.
        
        Returns:
            Lista de tuplas (expert_name, confidence)
        """
        class_predictions = torch.argmax(class_probs, dim=-1)
        max_probs = torch.max(class_probs, dim=-1)[0]
        
        results = []
        for class_id, confidence in zip(class_predictions, max_probs):
            class_id = int(class_id.cpu().item())
            confidence = float(confidence.cpu().item())
            
            if confidence >= confidence_threshold:
                expert_name = self.class_to_expert.get(class_id, "base")
            else:
                expert_name = "base"  # Fallback a modelo base
            
            results.append((expert_name, confidence))
        
        return results


# =============================================================================
# EJEMPLO DE CONFIGURACIÓN
# =============================================================================

# Mapeo simple: 3 clases → 3 expertos
SIMPLE_CLASS_TO_EXPERT = {
    0: "expert_es_mayores",
    1: "expert_latam_jovenes",
    2: "base"
}

SIMPLE_CLASS_NAMES = {
    0: "España Mayores",
    1: "Latinoamérica Jóvenes",
    2: "General"
}

# Mapeo multi-task
MULTITASK_CONFIGS = {
    "age_group": 3,      # joven/adulto/mayor
    "region": 3,         # españa/latam/otro
    "formality": 2,      # formal/informal
    "quality": 3         # clara/irregular/difícil
}

# Combinación de attributes → expert
def multitask_to_expert(predictions: Dict[str, int]) -> str:
    """
    Lógica personalizada para mapear múltiples attributes a un experto.
    
    Args:
        predictions: {"task_name": class_id}
    
    Returns:
        expert_name
    """
    age = predictions.get("age_group", 1)
    region = predictions.get("region", 2)
    
    # Lógica de routing
    if age == 2 and region == 0:  # Mayor + España
        return "expert_es_mayores"
    elif age == 0 and region == 1:  # Joven + Latam
        return "expert_latam_jovenes"
    else:
        return "base"

