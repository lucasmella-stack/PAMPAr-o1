"""
distillation_trainer.py - Entrenador para destilación de conocimiento

Este módulo implementa el entrenamiento de LLARRI usando conocimiento
destilado de MiniCPM-V.

La destilación de conocimiento funciona así:
1. Teacher (MiniCPM) genera predicciones "soft" (con probabilidades)
2. Student (LLARRI) aprende a imitar esas predicciones
3. Se combina pérdida de imitación + pérdida de etiquetas reales

Loss = α * CrossEntropy(student, hard_labels) + β * KL_Divergence(student, teacher_soft_labels)

Ventajas:
- LLARRI absorbe conocimiento de MiniCPM
- En producción solo necesitás LLARRI
- Modelo más pequeño con rendimiento similar al grande
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuración para destilación."""
    
    # Pesos de pérdidas
    alpha_hard: float = 0.3  # Peso de CrossEntropy con ground truth
    alpha_soft: float = 0.7  # Peso de KL divergence con teacher
    
    # Temperatura para suavizar distribuciones
    temperature: float = 2.0
    
    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 10
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    # Gradient
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints/distillation"


class DistillationLoss(nn.Module):
    """
    Pérdida combinada para destilación de conocimiento.
    
    Combina:
    1. CrossEntropy loss con etiquetas reales (hard labels)
    2. KL Divergence con predicciones del teacher (soft labels)
    
    La temperatura T > 1 "suaviza" las probabilidades del teacher,
    haciendo que el student aprenda más de las distribuciones completas
    y no solo del token más probable.
    """
    
    def __init__(
        self,
        alpha_hard: float = 0.3,
        alpha_soft: float = 0.7,
        temperature: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha_hard = alpha_hard
        self.alpha_soft = alpha_soft
        self.temperature = temperature
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula pérdida de destilación.
        
        Args:
            student_logits: Logits del modelo student [batch, seq, vocab]
            hard_labels: Etiquetas reales [batch, seq]
            teacher_logits: Logits del teacher (si disponibles)
            teacher_confidence: Confianza del teacher para ponderar
            
        Returns:
            Tuple de (loss total, diccionario de métricas)
        """
        # Reshape para CrossEntropy
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # 1. Hard loss (CrossEntropy con ground truth)
        hard_loss = self.ce_loss(
            student_logits.view(-1, vocab_size),
            hard_labels.view(-1)
        )
        
        metrics = {"hard_loss": hard_loss.item()}
        
        # 2. Soft loss (KL Divergence con teacher)
        if teacher_logits is not None:
            # Aplicar temperatura
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # KL Divergence
            soft_loss = self.kl_loss(
                student_soft.view(-1, vocab_size),
                teacher_soft.view(-1, vocab_size)
            )
            
            # Escalar por T^2 (corrección estándar en destilación)
            soft_loss = soft_loss * (self.temperature ** 2)
            
            # Ponderar por confianza del teacher si está disponible
            if teacher_confidence is not None:
                # Confianza como peso [batch, 1, 1]
                confidence_weight = teacher_confidence.view(-1, 1, 1)
                soft_loss = soft_loss * confidence_weight.mean()
            
            metrics["soft_loss"] = soft_loss.item()
            
            # Combinar pérdidas
            total_loss = self.alpha_hard * hard_loss + self.alpha_soft * soft_loss
            
        else:
            # Sin teacher, solo usar hard loss
            total_loss = hard_loss
            metrics["soft_loss"] = 0.0
        
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics


class DistillationTrainer(pl.LightningModule):
    """
    Lightning module para entrenar LLARRI con destilación.
    
    Entrena el modelo student (LLARRI) para imitar al teacher (MiniCPM)
    mientras también aprende de las etiquetas reales.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        config: DistillationConfig,
        tokenizer: Any = None,
    ):
        super().__init__()
        
        self.student = student_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Pérdida de destilación
        self.loss_fn = DistillationLoss(
            alpha_hard=config.alpha_hard,
            alpha_soft=config.alpha_soft,
            temperature=config.temperature,
        )
        
        # Métricas
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Guardar hiperparámetros
        self.save_hyperparameters(ignore=['student_model', 'tokenizer'])
    
    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass a través del modelo student."""
        return self.student(images, labels)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Un paso de entrenamiento."""
        images = batch['images']
        hard_labels = batch['labels']
        
        # Soft labels del teacher (precalculados)
        teacher_logits = batch.get('teacher_logits')
        teacher_confidence = batch.get('teacher_confidence')
        
        # Forward del student
        student_outputs = self(images, hard_labels)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
        
        # Calcular pérdida
        loss, metrics = self.loss_fn(
            student_logits,
            hard_labels,
            teacher_logits,
            teacher_confidence,
        )
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_hard_loss', metrics['hard_loss'])
        self.log('train_soft_loss', metrics['soft_loss'])
        
        self.training_step_outputs.append({
            'loss': loss.detach(),
            **{k: torch.tensor(v) for k, v in metrics.items()}
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Un paso de validación."""
        images = batch['images']
        hard_labels = batch['labels']
        
        teacher_logits = batch.get('teacher_logits')
        teacher_confidence = batch.get('teacher_confidence')
        
        # Forward
        with torch.no_grad():
            student_outputs = self(images, hard_labels)
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
        
        # Calcular pérdida
        loss, metrics = self.loss_fn(
            student_logits,
            hard_labels,
            teacher_logits,
            teacher_confidence,
        )
        
        # Calcular accuracy de caracteres
        predictions = student_logits.argmax(dim=-1)
        mask = hard_labels != -100
        correct = (predictions == hard_labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        self.validation_step_outputs.append({
            'val_loss': loss.detach(),
            'val_accuracy': accuracy.detach(),
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Resumen de época de entrenamiento."""
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            logger.info(f"Epoch {self.current_epoch} - Train Loss: {avg_loss:.4f}")
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Resumen de época de validación."""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['val_accuracy'] for x in self.validation_step_outputs]).mean()
            logger.info(f"Epoch {self.current_epoch} - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configura optimizador y scheduler."""
        # Separar parámetros para weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.student.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.student.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        # Scheduler con warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }


class ProgressiveDistillation:
    """
    Destilación progresiva en múltiples etapas.
    
    Etapa 1: Entrenar solo con soft labels (alta confianza)
    Etapa 2: Agregar hard labels
    Etapa 3: Fine-tune con todos los datos
    
    Esto permite una transición más suave del conocimiento.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        config: DistillationConfig,
        tokenizer: Any = None,
    ):
        self.student = student_model
        self.config = config
        self.tokenizer = tokenizer
        self.current_stage = 0
    
    def stage_configs(self) -> List[DistillationConfig]:
        """Configuraciones para cada etapa."""
        return [
            # Etapa 1: Solo soft labels
            DistillationConfig(
                alpha_hard=0.0,
                alpha_soft=1.0,
                temperature=3.0,
                learning_rate=5e-5,
                max_epochs=3,
            ),
            # Etapa 2: Mixto
            DistillationConfig(
                alpha_hard=0.3,
                alpha_soft=0.7,
                temperature=2.0,
                learning_rate=2e-5,
                max_epochs=5,
            ),
            # Etapa 3: Más hard labels
            DistillationConfig(
                alpha_hard=0.5,
                alpha_soft=0.5,
                temperature=1.5,
                learning_rate=1e-5,
                max_epochs=5,
            ),
        ]
    
    def train_stage(
        self,
        stage: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> pl.Trainer:
        """Entrena una etapa específica."""
        
        configs = self.stage_configs()
        if stage >= len(configs):
            raise ValueError(f"Etapa {stage} no existe")
        
        stage_config = configs[stage]
        logger.info(f"\n{'='*50}")
        logger.info(f"Iniciando Etapa {stage + 1}/{len(configs)}")
        logger.info(f"alpha_hard={stage_config.alpha_hard}, alpha_soft={stage_config.alpha_soft}")
        logger.info(f"temperature={stage_config.temperature}, lr={stage_config.learning_rate}")
        logger.info(f"{'='*50}\n")
        
        # Crear trainer para esta etapa
        distill_trainer = DistillationTrainer(
            student_model=self.student,
            config=stage_config,
            tokenizer=self.tokenizer,
        )
        
        # Callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{self.config.checkpoint_dir}/stage_{stage}",
                filename=f"distill-stage{stage}-{{epoch}}-{{val_loss:.4f}}",
                monitor='val_loss',
                mode='min',
                save_top_k=2,
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=stage_config.patience,
                mode='min',
            ),
        ]
        
        # PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=stage_config.max_epochs,
            accelerator='auto',
            devices=1,
            precision='16-mixed' if stage_config.use_amp else 32,
            gradient_clip_val=stage_config.gradient_clip_val,
            accumulate_grad_batches=stage_config.accumulate_grad_batches,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
        
        # Entrenar
        trainer.fit(
            distill_trainer,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        self.current_stage = stage + 1
        
        return trainer
    
    def train_all_stages(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        """Entrena todas las etapas secuencialmente."""
        
        for stage in range(len(self.stage_configs())):
            self.train_stage(stage, train_dataloader, val_dataloader)
        
        logger.info("\n✅ Destilación progresiva completada!")
        logger.info(f"Modelo final guardado en: {self.config.checkpoint_dir}")


def create_distillation_trainer(
    student_model: nn.Module,
    config_path: Optional[str] = None,
    **kwargs,
) -> DistillationTrainer:
    """
    Factory function para crear un DistillationTrainer.
    
    Args:
        student_model: Modelo LLARRI a entrenar
        config_path: Path a archivo YAML de configuración
        **kwargs: Override de parámetros de configuración
        
    Returns:
        DistillationTrainer configurado
    """
    # Cargar config si existe
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    
    # Override con kwargs
    config_dict.update(kwargs)
    
    # Crear config
    config = DistillationConfig(**config_dict)
    
    # Crear trainer
    return DistillationTrainer(
        student_model=student_model,
        config=config,
    )
