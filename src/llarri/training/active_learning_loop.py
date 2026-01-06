"""
active_learning_loop.py - Loop de Active Learning

Implementa el ciclo completo de aprendizaje activo para mejorar modelos
de forma iterativa seleccionando las muestras más informativas:

Flujo del loop:
1. Entrenar modelo inicial con pequeño dataset seed
2. Predecir sobre pool no etiquetado
3. Seleccionar N muestras más inciertas
4. Etiquetar (manual o semi-automático)
5. Agregar al dataset de entrenamiento
6. Re-entrenar modelo
7. Evaluar mejora
8. Repetir hasta criterio de parada

Criterios de parada:
- Máximo de iteraciones alcanzado
- Performance deseado alcanzado
- Pool agotado
- Budget de etiquetado agotado

Uso:
    python src/llarri/training/active_learning_loop.py \\
        --initial_model outputs/base_model/best.ckpt \\
        --pool_data data/unlabeled_pool.jsonl \\
        --seed_data data/splits/train_seed.jsonl \\
        --val_data data/splits/val.jsonl \\
        --n_iterations 10 \\
        --samples_per_iteration 100 \\
        --strategy entropy
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ..models.llarri_base_model import LlarriBaseModel
from ..models.selector_style import StyleSelector
from ..active_learning.sampler_uncertain import (
    UncertaintySampler,
    DiversitySampler,
    EnsembleUncertaintySampler
)
from ..data.transforms import get_transforms
from .train_base import train as train_base_model
from .finetune_expert import finetune_expert
from .train_selector import train_selector


class ActiveLearningLoop:
    """
    Gestor del loop de active learning.
    
    Mantiene estado del loop: dataset actual, modelos, métricas, historial.
    """
    def __init__(
        self,
        initial_model_path: str,
        pool_data_path: str,
        seed_data_path: str,
        val_data_path: str,
        sampler_strategy: str = 'entropy',
        samples_per_iteration: int = 100,
        max_iterations: int = 10,
        output_dir: str = 'outputs/active_learning',
        model_type: str = 'base',  # 'base', 'expert', 'selector'
        device: str = 'auto'
    ):
        """
        Args:
            initial_model_path: Checkpoint del modelo inicial
            pool_data_path: JSONL con datos no etiquetados
            seed_data_path: JSONL con datos seed iniciales
            val_data_path: JSONL con datos de validación
            sampler_strategy: Estrategia de muestreo
            samples_per_iteration: Muestras a seleccionar por iteración
            max_iterations: Máximo de iteraciones
            output_dir: Directorio para outputs
            model_type: Tipo de modelo ('base', 'expert', 'selector')
            device: Device para inferencia
        """
        self.pool_data_path = pool_data_path
        self.seed_data_path = seed_data_path
        self.val_data_path = val_data_path
        self.sampler_strategy = sampler_strategy
        self.samples_per_iteration = samples_per_iteration
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar modelo inicial
        print(f"📂 Cargando modelo inicial desde {initial_model_path}")
        if model_type == 'base':
            self.model = LlarriBaseModel.load_from_checkpoint(initial_model_path)
        elif model_type == 'selector':
            from .train_selector import StyleSelectorModule
            self.model = StyleSelectorModule.load_from_checkpoint(initial_model_path)
            self.model = self.model.selector
        else:
            self.model = torch.load(initial_model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Cargar datos
        print("📂 Cargando datos...")
        self.pool_data = pd.read_json(pool_data_path, lines=True)
        self.seed_data = pd.read_json(seed_data_path, lines=True)
        self.val_data = pd.read_json(val_data_path, lines=True)
        
        print(f"   Pool: {len(self.pool_data)} muestras")
        print(f"   Seed: {len(self.seed_data)} muestras")
        print(f"   Val: {len(self.val_data)} muestras")
        
        # Crear sampler
        self.sampler = self._create_sampler()
        
        # Estado del loop
        self.current_iteration = 0
        self.labeled_data = self.seed_data.copy()
        self.remaining_pool = self.pool_data.copy()
        
        # Historial de métricas
        self.history = {
            'iteration': [],
            'train_size': [],
            'val_metric': [],
            'selected_samples': [],
            'avg_uncertainty': []
        }
        
        # Transforms
        self.transform = get_transforms(augment=False)
        
        print("✅ ActiveLearningLoop inicializado")
    
    def _create_sampler(self):
        """Crea el sampler apropiado según la estrategia."""
        if self.sampler_strategy in ['least_confidence', 'margin', 'entropy', 'ratio']:
            return UncertaintySampler(
                model=self.model,
                strategy=self.sampler_strategy,
                device=self.device
            )
        
        elif self.sampler_strategy == 'diversity':
            return DiversitySampler(
                model=self.model,
                strategy='entropy',
                diversity_weight=0.3,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.sampler_strategy}")
    
    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """Carga y preprocesa imágenes."""
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)
        return torch.stack(images)
    
    def select_samples(self) -> Tuple[List[int], np.ndarray]:
        """
        Selecciona muestras del pool usando el sampler.
        
        Returns:
            selected_indices: Índices en remaining_pool
            uncertainty_scores: Scores de incertidumbre
        """
        print(f"\n🎯 Iteración {self.current_iteration + 1}/{self.max_iterations}")
        print(f"   Pool restante: {len(self.remaining_pool)} muestras")
        
        # Cargar imágenes del pool
        pool_images = self.load_images(self.remaining_pool['image_path'].tolist())
        
        # Seleccionar usando sampler
        selected_indices, scores = self.sampler.select_samples(
            pool_images=pool_images,
            n_samples=self.samples_per_iteration,
            return_scores=True
        )
        
        return selected_indices, scores
    
    def label_samples(
        self,
        selected_indices: List[int],
        labeling_mode: str = 'manual'
    ) -> pd.DataFrame:
        """
        Etiqueta las muestras seleccionadas.
        
        Args:
            selected_indices: Índices de muestras en remaining_pool
            labeling_mode: 'manual', 'semi_automatic', 'oracle'
        
        Returns:
            labeled_df: DataFrame con muestras etiquetadas
        """
        selected_samples = self.remaining_pool.iloc[selected_indices].copy()
        
        if labeling_mode == 'manual':
            print("\n📝 Etiquetado manual requerido")
            print(f"   {len(selected_samples)} muestras seleccionadas")
            print(f"   Guardar etiquetas en: {self.output_dir}/iteration_{self.current_iteration}_to_label.jsonl")
            
            # Guardar muestras a etiquetar
            output_path = self.output_dir / f"iteration_{self.current_iteration}_to_label.jsonl"
            selected_samples.to_json(output_path, orient='records', lines=True)
            
            print(f"   ⚠️  Etiqueta las muestras y guarda como: {self.output_dir}/iteration_{self.current_iteration}_labeled.jsonl")
            print("   Presiona Enter cuando hayas terminado...")
            input()
            
            # Cargar etiquetas
            labeled_path = self.output_dir / f"iteration_{self.current_iteration}_labeled.jsonl"
            if not labeled_path.exists():
                raise FileNotFoundError(f"No se encontró {labeled_path}")
            
            labeled_df = pd.read_json(labeled_path, lines=True)
        
        elif labeling_mode == 'oracle':
            # Simular oracle (datos ya tienen etiquetas)
            print("🔮 Modo oracle: usando etiquetas ground truth")
            labeled_df = selected_samples
            
            # Verificar que tengan etiquetas
            if 'text' not in labeled_df.columns and 'style_label' not in labeled_df.columns:
                raise ValueError("Oracle mode requires samples to have labels")
        
        elif labeling_mode == 'semi_automatic':
            print("🤖 Etiquetado semi-automático")
            # Usar modelo para pre-etiquetar, luego revisar manualmente
            # TODO: Implementar
            labeled_df = selected_samples
        
        else:
            raise ValueError(f"Unknown labeling_mode: {labeling_mode}")
        
        return labeled_df
    
    def update_datasets(
        self,
        selected_indices: List[int],
        labeled_samples: pd.DataFrame
    ):
        """
        Actualiza dataset de entrenamiento y pool.
        """
        # Agregar nuevas etiquetas al dataset de entrenamiento
        self.labeled_data = pd.concat([self.labeled_data, labeled_samples], ignore_index=True)
        
        # Remover del pool
        self.remaining_pool = self.remaining_pool.drop(
            self.remaining_pool.index[selected_indices]
        ).reset_index(drop=True)
        
        print(f"✅ Datasets actualizados")
        print(f"   Training: {len(self.labeled_data)} muestras")
        print(f"   Pool: {len(self.remaining_pool)} muestras")
    
    def retrain_model(self) -> float:
        """
        Re-entrena el modelo con el dataset actualizado.
        
        Returns:
            val_metric: Métrica de validación
        """
        print("\n🔄 Re-entrenando modelo...")
        
        # Guardar dataset actual
        train_path = self.output_dir / f"iteration_{self.current_iteration}_train.jsonl"
        self.labeled_data.to_json(train_path, orient='records', lines=True)
        
        # Re-entrenar según tipo de modelo
        if self.model_type == 'base':
            # TODO: Llamar a train_base_model con dataset actualizado
            print("⚠️  Re-training de base model - implementación pendiente")
            val_metric = 0.0
        
        elif self.model_type == 'selector':
            # TODO: Llamar a train_selector con dataset actualizado
            print("⚠️  Re-training de selector - implementación pendiente")
            val_metric = 0.0
        
        else:
            val_metric = 0.0
        
        # Actualizar sampler con nuevo modelo
        self.sampler.model = self.model
        
        return val_metric
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evalúa el modelo actual en el conjunto de validación.
        
        Returns:
            metrics: Diccionario con métricas
        """
        print("\n📊 Evaluando modelo...")
        
        # Cargar imágenes de validación
        val_images = self.load_images(self.val_data['image_path'].tolist())
        
        with torch.no_grad():
            val_images = val_images.to(self.device)
            logits = self.model(val_images)
            
            if isinstance(logits, dict):
                first_task = list(logits.keys())[0]
                logits = logits[first_task]
            
            # Calcular accuracy
            if 'style_label' in self.val_data.columns:
                # Clasificación
                labels = torch.tensor(self.val_data['style_label'].tolist()).to(self.device)
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == labels).float().mean().item()
                
                metrics = {'accuracy': accuracy}
                print(f"   Accuracy: {accuracy:.4f}")
            
            else:
                # OCR - calcular CER/WER
                # TODO: Implementar métricas de OCR
                metrics = {'placeholder': 0.0}
        
        return metrics
    
    def run_iteration(self, labeling_mode: str = 'oracle') -> bool:
        """
        Ejecuta una iteración del loop.
        
        Returns:
            should_continue: Si debe continuar con siguiente iteración
        """
        # 1. Seleccionar muestras
        selected_indices, uncertainty_scores = self.select_samples()
        
        # 2. Etiquetar
        labeled_samples = self.label_samples(selected_indices, labeling_mode)
        
        # 3. Actualizar datasets
        self.update_datasets(selected_indices, labeled_samples)
        
        # 4. Re-entrenar
        val_metric = self.retrain_model()
        
        # 5. Evaluar
        metrics = self.evaluate()
        
        # 6. Guardar historial
        self.history['iteration'].append(self.current_iteration)
        self.history['train_size'].append(len(self.labeled_data))
        self.history['val_metric'].append(metrics.get('accuracy', val_metric))
        self.history['selected_samples'].append(len(selected_indices))
        self.history['avg_uncertainty'].append(uncertainty_scores.mean())
        
        self.current_iteration += 1
        
        # Verificar criterios de parada
        if self.current_iteration >= self.max_iterations:
            print("\n✅ Máximo de iteraciones alcanzado")
            return False
        
        if len(self.remaining_pool) < self.samples_per_iteration:
            print("\n✅ Pool agotado")
            return False
        
        return True
    
    def run(self, labeling_mode: str = 'oracle'):
        """
        Ejecuta el loop completo.
        """
        print("="*70)
        print("🔄 INICIANDO ACTIVE LEARNING LOOP")
        print("="*70)
        print(f"   Estrategia: {self.sampler_strategy}")
        print(f"   Muestras por iteración: {self.samples_per_iteration}")
        print(f"   Máximo de iteraciones: {self.max_iterations}")
        print(f"   Modo de etiquetado: {labeling_mode}")
        print()
        
        # Evaluación inicial
        print("📊 Evaluación inicial del modelo...")
        initial_metrics = self.evaluate()
        self.history['iteration'].append(0)
        self.history['train_size'].append(len(self.labeled_data))
        self.history['val_metric'].append(initial_metrics.get('accuracy', 0.0))
        self.history['selected_samples'].append(0)
        self.history['avg_uncertainty'].append(0.0)
        
        # Loop principal
        while True:
            should_continue = self.run_iteration(labeling_mode)
            if not should_continue:
                break
        
        # Resultados finales
        self.save_results()
        self.plot_results()
        
        print("\n" + "="*70)
        print("✅ ACTIVE LEARNING LOOP COMPLETADO")
        print("="*70)
        print(f"   Iteraciones: {self.current_iteration}")
        print(f"   Training final: {len(self.labeled_data)} muestras")
        print(f"   Mejora: {self.history['val_metric'][0]:.4f} → {self.history['val_metric'][-1]:.4f}")
        print(f"   Resultados en: {self.output_dir}")
    
    def save_results(self):
        """Guarda resultados del loop."""
        # Guardar historial
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Guardar dataset final
        final_train_path = self.output_dir / 'final_train.jsonl'
        self.labeled_data.to_json(final_train_path, orient='records', lines=True)
        
        print(f"💾 Resultados guardados en {self.output_dir}")
    
    def plot_results(self):
        """Genera gráficas de resultados."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training size
        axes[0, 0].plot(self.history['iteration'], self.history['train_size'], 'o-')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Training Size')
        axes[0, 0].set_title('Training Set Growth')
        axes[0, 0].grid(True)
        
        # Validation metric
        axes[0, 1].plot(self.history['iteration'], self.history['val_metric'], 'o-')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Validation Metric')
        axes[0, 1].set_title('Model Performance')
        axes[0, 1].grid(True)
        
        # Samples per iteration
        axes[1, 0].bar(self.history['iteration'][1:], self.history['selected_samples'][1:])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Selected Samples')
        axes[1, 0].set_title('Samples Selected per Iteration')
        axes[1, 0].grid(True)
        
        # Average uncertainty
        axes[1, 1].plot(self.history['iteration'][1:], self.history['avg_uncertainty'][1:], 'o-')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Avg Uncertainty')
        axes[1, 1].set_title('Average Uncertainty Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'results.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"📊 Gráficas guardadas en {plot_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Active Learning Loop")
    
    # Modelo y datos
    parser.add_argument('--initial_model', type=str, required=True,
                       help='Checkpoint del modelo inicial')
    parser.add_argument('--pool_data', type=str, required=True,
                       help='JSONL con pool no etiquetado')
    parser.add_argument('--seed_data', type=str, required=True,
                       help='JSONL con datos seed iniciales')
    parser.add_argument('--val_data', type=str, required=True,
                       help='JSONL con datos de validación')
    
    # Estrategia
    parser.add_argument('--strategy', type=str, default='entropy',
                       choices=['least_confidence', 'margin', 'entropy', 'ratio', 'diversity'],
                       help='Estrategia de muestreo')
    
    # Parámetros del loop
    parser.add_argument('--samples_per_iteration', type=int, default=100,
                       help='Muestras a seleccionar por iteración')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Máximo de iteraciones')
    
    # Etiquetado
    parser.add_argument('--labeling_mode', type=str, default='manual',
                       choices=['manual', 'oracle', 'semi_automatic'],
                       help='Modo de etiquetado')
    
    # Modelo
    parser.add_argument('--model_type', type=str, default='selector',
                       choices=['base', 'expert', 'selector'],
                       help='Tipo de modelo')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/active_learning')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Crear loop
    al_loop = ActiveLearningLoop(
        initial_model_path=args.initial_model,
        pool_data_path=args.pool_data,
        seed_data_path=args.seed_data,
        val_data_path=args.val_data,
        sampler_strategy=args.strategy,
        samples_per_iteration=args.samples_per_iteration,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        model_type=args.model_type,
        device=args.device
    )
    
    # Ejecutar
    al_loop.run(labeling_mode=args.labeling_mode)


if __name__ == '__main__':
    main()

