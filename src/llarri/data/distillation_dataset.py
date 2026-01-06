"""
distillation_dataset.py - Dataset para entrenamiento con destilación

Este módulo proporciona un Dataset de PyTorch que carga:
1. Imágenes originales
2. Labels reales (ground truth)
3. Soft labels del teacher (MiniCPM)

El dataset se genera previamente con scripts/distill_from_minicpm.py
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """
    Dataset para destilación de conocimiento.
    
    Carga samples con:
    - Imagen original
    - Ground truth (si existe)
    - Predicción del teacher
    - Confianza del teacher
    - Soft labels (logits del teacher, si disponibles)
    """
    
    def __init__(
        self,
        distilled_file: str,
        images_dir: str,
        tokenizer: Any,
        transform: Optional[Any] = None,
        max_length: int = 256,
        use_teacher_as_label: bool = False,
        min_confidence: float = 0.0,
        include_low_confidence: bool = True,
    ):
        """
        Args:
            distilled_file: Path al archivo .jsonl generado por distill_from_minicpm.py
            images_dir: Directorio raíz de las imágenes
            tokenizer: Tokenizer del modelo (TrOCR)
            transform: Transformaciones de imagen (opcional)
            max_length: Longitud máxima de secuencia
            use_teacher_as_label: Si usar predicción del teacher como label cuando no hay GT
            min_confidence: Filtrar samples con confianza menor
            include_low_confidence: Incluir samples de baja confianza (para hard mining)
        """
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.use_teacher_as_label = use_teacher_as_label
        self.min_confidence = min_confidence
        self.include_low_confidence = include_low_confidence
        
        # Cargar datos destilados
        self.samples = self._load_distilled_data(distilled_file)
        
        logger.info(f"Cargados {len(self.samples)} samples para destilación")
    
    def _load_distilled_data(self, distilled_file: str) -> List[Dict[str, Any]]:
        """Carga y filtra datos destilados."""
        samples = []
        
        with open(distilled_file) as f:
            for line in f:
                sample = json.loads(line)
                
                # Filtrar por confianza si está configurado
                confidence = sample.get('teacher_confidence', 0.0)
                
                if confidence < self.min_confidence and not self.include_low_confidence:
                    continue
                
                # Verificar que la imagen existe
                img_path = self.images_dir / sample['image_path']
                if not img_path.exists():
                    logger.warning(f"Imagen no encontrada: {img_path}")
                    continue
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Obtiene un sample para entrenamiento."""
        sample = self.samples[idx]
        
        # Cargar imagen
        img_path = self.images_dir / sample['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        else:
            # Transformación por defecto
            image = self._default_transform(image)
        
        # Obtener label (ground truth o teacher prediction)
        if sample.get('ground_truth'):
            label_text = sample['ground_truth']
        elif self.use_teacher_as_label:
            label_text = sample.get('teacher_prediction', '')
        else:
            label_text = ''
        
        # Tokenizar label
        if label_text and self.tokenizer:
            labels = self.tokenizer(
                label_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            labels_ids = labels['input_ids'].squeeze(0)
        else:
            labels_ids = torch.full((self.max_length,), -100, dtype=torch.long)
        
        # Preparar output
        output = {
            'images': image,
            'labels': labels_ids,
            'teacher_confidence': torch.tensor(sample.get('teacher_confidence', 0.0)),
        }
        
        # Agregar soft labels si existen
        if 'teacher_logits' in sample and sample['teacher_logits']:
            teacher_logits = torch.tensor(sample['teacher_logits'])
            output['teacher_logits'] = teacher_logits
        
        # Metadata adicional
        output['image_path'] = sample['image_path']
        output['teacher_prediction'] = sample.get('teacher_prediction', '')
        
        return output
    
    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Transformación por defecto para imágenes."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Tamaño estándar ViT
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transform(image)
    
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Obtiene distribución de confianzas para análisis."""
        distribution = {
            'very_low (0-0.3)': 0,
            'low (0.3-0.5)': 0,
            'medium (0.5-0.7)': 0,
            'high (0.7-0.9)': 0,
            'very_high (0.9-1.0)': 0,
        }
        
        for sample in self.samples:
            conf = sample.get('teacher_confidence', 0.0)
            if conf < 0.3:
                distribution['very_low (0-0.3)'] += 1
            elif conf < 0.5:
                distribution['low (0.3-0.5)'] += 1
            elif conf < 0.7:
                distribution['medium (0.5-0.7)'] += 1
            elif conf < 0.9:
                distribution['high (0.7-0.9)'] += 1
            else:
                distribution['very_high (0.9-1.0)'] += 1
        
        return distribution


class WeightedDistillationDataset(DistillationDataset):
    """
    Dataset con ponderación basada en confianza del teacher.
    
    Samples con alta confianza del teacher tienen más peso,
    lo que ayuda al modelo a aprender más de predicciones confiables.
    """
    
    def __init__(
        self,
        *args,
        confidence_weight_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.confidence_weight_scale = confidence_weight_scale
        self._compute_weights()
    
    def _compute_weights(self):
        """Calcula pesos para cada sample basado en confianza."""
        confidences = [s.get('teacher_confidence', 0.5) for s in self.samples]
        
        # Normalizar y escalar
        min_conf = min(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 1
        range_conf = max_conf - min_conf or 1
        
        self.weights = []
        for conf in confidences:
            normalized = (conf - min_conf) / range_conf
            weight = 1.0 + normalized * (self.confidence_weight_scale - 1.0)
            self.weights.append(weight)
    
    def get_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """Obtiene sampler ponderado para DataLoader."""
        return torch.utils.data.WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.weights),
            replacement=True,
        )


class CurriculumDistillationDataset(DistillationDataset):
    """
    Dataset con curriculum learning basado en dificultad.
    
    Empieza entrenando con samples "fáciles" (alta confianza)
    y progresivamente incluye samples más difíciles.
    """
    
    def __init__(
        self,
        *args,
        initial_percentile: float = 0.3,  # Empezar con 30% más fáciles
        final_percentile: float = 1.0,    # Terminar con 100%
        num_stages: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.initial_percentile = initial_percentile
        self.final_percentile = final_percentile
        self.num_stages = num_stages
        self.current_stage = 0
        
        # Ordenar samples por confianza (más fáciles primero)
        self._sort_by_difficulty()
    
    def _sort_by_difficulty(self):
        """Ordena samples por dificultad (confianza inversa)."""
        self.samples = sorted(
            self.samples,
            key=lambda x: x.get('teacher_confidence', 0.0),
            reverse=True  # Alta confianza = más fácil
        )
        
        self.all_samples = self.samples.copy()
        self._update_active_samples()
    
    def _update_active_samples(self):
        """Actualiza samples activos según la etapa actual."""
        progress = self.current_stage / max(1, self.num_stages - 1)
        current_percentile = self.initial_percentile + progress * (self.final_percentile - self.initial_percentile)
        
        num_samples = int(len(self.all_samples) * current_percentile)
        self.samples = self.all_samples[:num_samples]
        
        logger.info(f"Curriculum stage {self.current_stage}: usando {len(self.samples)}/{len(self.all_samples)} samples ({current_percentile:.0%})")
    
    def advance_stage(self):
        """Avanza a la siguiente etapa del curriculum."""
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            self._update_active_samples()
    
    def reset(self):
        """Reinicia al stage inicial."""
        self.current_stage = 0
        self._update_active_samples()


class MixedDistillationDataset(Dataset):
    """
    Dataset que mezcla datos destilados con datos originales.
    
    Útil cuando tenés tanto datos con soft labels del teacher
    como datos adicionales sin procesar por el teacher.
    """
    
    def __init__(
        self,
        distilled_dataset: DistillationDataset,
        original_dataset: Dataset,
        distilled_ratio: float = 0.7,  # 70% destilados, 30% originales
    ):
        self.distilled_dataset = distilled_dataset
        self.original_dataset = original_dataset
        self.distilled_ratio = distilled_ratio
        
        # Calcular tamaño efectivo
        self.distilled_count = int(len(distilled_dataset) * distilled_ratio)
        self.original_count = len(original_dataset)
        
        logger.info(f"MixedDataset: {self.distilled_count} destilados + {self.original_count} originales")
    
    def __len__(self) -> int:
        return self.distilled_count + self.original_count
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.distilled_count:
            # Sample destilado
            actual_idx = int(idx / self.distilled_ratio) % len(self.distilled_dataset)
            sample = self.distilled_dataset[actual_idx]
            sample['is_distilled'] = torch.tensor(1)
        else:
            # Sample original (sin soft labels)
            original_idx = idx - self.distilled_count
            sample = self.original_dataset[original_idx]
            
            # Asegurar formato compatible
            if 'teacher_logits' not in sample:
                sample['teacher_logits'] = None
            if 'teacher_confidence' not in sample:
                sample['teacher_confidence'] = torch.tensor(0.0)
            sample['is_distilled'] = torch.tensor(0)
        
        return sample


def create_distillation_dataloaders(
    distilled_file: str,
    images_dir: str,
    tokenizer: Any,
    batch_size: int = 4,
    val_split: float = 0.1,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    use_curriculum: bool = False,
    transform: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento y validación.
    
    Args:
        distilled_file: Path al archivo .jsonl con datos destilados
        images_dir: Directorio de imágenes
        tokenizer: Tokenizer del modelo
        batch_size: Tamaño de batch
        val_split: Fracción para validación
        num_workers: Workers para carga de datos
        use_weighted_sampling: Usar sampling ponderado por confianza
        use_curriculum: Usar curriculum learning
        transform: Transformaciones de imagen
        
    Returns:
        Tuple de (train_dataloader, val_dataloader)
    """
    # Seleccionar tipo de dataset
    if use_curriculum:
        DatasetClass = CurriculumDistillationDataset
    elif use_weighted_sampling:
        DatasetClass = WeightedDistillationDataset
    else:
        DatasetClass = DistillationDataset
    
    # Crear dataset completo
    full_dataset = DatasetClass(
        distilled_file=distilled_file,
        images_dir=images_dir,
        tokenizer=tokenizer,
        transform=transform,
        use_teacher_as_label=True,
    )
    
    # Split train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Crear DataLoaders
    if use_weighted_sampling and hasattr(full_dataset, 'get_sampler'):
        # Usar sampler ponderado solo para train
        train_sampler = full_dataset.get_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=distillation_collate_fn,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=distillation_collate_fn,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=distillation_collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"DataLoaders creados: {train_size} train, {val_size} val")
    
    return train_loader, val_loader


def distillation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function para batches de destilación.
    
    Maneja casos donde algunos samples tienen soft labels y otros no.
    """
    # Separar campos
    images = torch.stack([item['images'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    confidences = torch.stack([item['teacher_confidence'] for item in batch])
    
    result = {
        'images': images,
        'labels': labels,
        'teacher_confidence': confidences,
    }
    
    # Soft labels (pueden ser None para algunos samples)
    if all('teacher_logits' in item and item['teacher_logits'] is not None for item in batch):
        teacher_logits = torch.stack([item['teacher_logits'] for item in batch])
        result['teacher_logits'] = teacher_logits
    
    return result
