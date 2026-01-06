"""
sampler_uncertain.py - Estrategias de Muestreo por Incertidumbre

Implementa diferentes estrategias de active learning para seleccionar
las muestras más informativas del pool no etiquetado:

1. Least Confidence: Selecciona muestras con menor confianza máxima
2. Margin Sampling: Selecciona muestras con menor margen entre top-2 clases
3. Entropy: Selecciona muestras con mayor entropía de predicción
4. Diversity-based: Combina incertidumbre con diversidad (evita redundancia)
5. Ensemble Uncertainty: Usa desacuerdo entre múltiples modelos

Flujo típico:
    1. Modelo predice sobre pool no etiquetado
    2. Sampler calcula scores de incertidumbre
    3. Se seleccionan top N muestras más inciertas
    4. Usuario las etiqueta
    5. Se re-entrena con nuevas etiquetas

Uso:
    sampler = UncertaintySampler(model, strategy='entropy')
    selected_indices = sampler.select_samples(
        pool_images=unlabeled_images,
        n_samples=100
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class UncertaintySampler:
    """
    Sampler base para active learning basado en incertidumbre.
    
    Estrategias disponibles:
    - least_confidence: P(y_max) bajo
    - margin: Margen pequeño entre top-2 clases
    - entropy: Entropía alta de distribución
    - ratio: Ratio bajo entre top-2
    """
    def __init__(
        self,
        model,
        strategy: str = 'entropy',
        device: str = 'auto'
    ):
        """
        Args:
            model: Modelo de clasificación (StyleSelector u otro)
            strategy: Estrategia de muestreo
            device: Device para inferencia
        """
        self.model = model
        self.strategy = strategy
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ UncertaintySampler creado con estrategia: {strategy}")
    
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """
        Obtiene probabilidades de predicción para un batch de imágenes.
        
        Args:
            images: Tensor (B, C, H, W)
        
        Returns:
            probs: Tensor (B, num_classes) con probabilidades
        """
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            
            # Manejar multi-task (usar primera task)
            if isinstance(logits, dict):
                first_task = list(logits.keys())[0]
                logits = logits[first_task]
            
            probs = F.softmax(logits, dim=-1)
        
        return probs.cpu()
    
    def least_confidence_score(self, probs: torch.Tensor) -> np.ndarray:
        """
        Score = 1 - P(y_max)
        
        Mayor score = menor confianza = más incierto
        """
        max_probs = torch.max(probs, dim=-1)[0]
        scores = 1.0 - max_probs.numpy()
        return scores
    
    def margin_sampling_score(self, probs: torch.Tensor) -> np.ndarray:
        """
        Score = P(y_top1) - P(y_top2)
        
        Menor margen = más incierto entre las dos clases principales
        """
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1 = sorted_probs[:, 0]
        top2 = sorted_probs[:, 1]
        
        margins = top1 - top2
        scores = -margins.numpy()  # Negativo para que menor margen = mayor score
        return scores
    
    def entropy_score(self, probs: torch.Tensor) -> np.ndarray:
        """
        Score = -Σ P(y) * log(P(y))
        
        Mayor entropía = distribución más uniforme = más incierto
        """
        # Evitar log(0)
        probs = torch.clamp(probs, min=1e-10)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.numpy()
    
    def ratio_score(self, probs: torch.Tensor) -> np.ndarray:
        """
        Score = P(y_top2) / P(y_top1)
        
        Mayor ratio = las dos clases principales son similares = más incierto
        """
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1 = sorted_probs[:, 0]
        top2 = sorted_probs[:, 1]
        
        ratios = top2 / (top1 + 1e-10)
        return ratios.numpy()
    
    def calculate_uncertainty(self, probs: torch.Tensor) -> np.ndarray:
        """
        Calcula scores de incertidumbre según la estrategia.
        
        Args:
            probs: Probabilidades (B, num_classes)
        
        Returns:
            scores: Array (B,) con scores de incertidumbre (mayor = más incierto)
        """
        if self.strategy == 'least_confidence':
            return self.least_confidence_score(probs)
        
        elif self.strategy == 'margin':
            return self.margin_sampling_score(probs)
        
        elif self.strategy == 'entropy':
            return self.entropy_score(probs)
        
        elif self.strategy == 'ratio':
            return self.ratio_score(probs)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def select_samples(
        self,
        pool_images: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int,
        batch_size: int = 32,
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Selecciona las muestras más inciertas del pool.
        
        Args:
            pool_images: Lista o batch de imágenes no etiquetadas
            n_samples: Número de muestras a seleccionar
            batch_size: Batch size para inferencia
            return_scores: Si retornar también los scores
        
        Returns:
            selected_indices: Índices de muestras seleccionadas
            (scores): Scores de incertidumbre (si return_scores=True)
        """
        # Convertir a tensor si es lista
        if isinstance(pool_images, list):
            pool_images = torch.stack(pool_images)
        
        n_pool = len(pool_images)
        all_probs = []
        
        # Inferencia en batches
        print(f"🔄 Calculando incertidumbre para {n_pool} muestras...")
        for i in range(0, n_pool, batch_size):
            batch = pool_images[i:i+batch_size]
            probs = self.predict_proba(batch)
            all_probs.append(probs)
        
        all_probs = torch.cat(all_probs, dim=0)
        
        # Calcular scores de incertidumbre
        scores = self.calculate_uncertainty(all_probs)
        
        # Seleccionar top N
        n_samples = min(n_samples, n_pool)
        selected_indices = np.argsort(scores)[-n_samples:][::-1]
        
        print(f"✅ Seleccionadas {len(selected_indices)} muestras")
        print(f"   Score medio: {scores[selected_indices].mean():.4f}")
        print(f"   Score max: {scores.max():.4f}")
        print(f"   Score min: {scores.min():.4f}")
        
        if return_scores:
            return selected_indices, scores[selected_indices]
        return selected_indices


class DiversitySampler(UncertaintySampler):
    """
    Sampler que combina incertidumbre con diversidad.
    
    Evita seleccionar muestras redundantes eligiendo ejemplos diversos
    que además son inciertos.
    
    Estrategia:
    1. Calcular incertidumbre para todas las muestras
    2. Seleccionar top K más inciertas (K > n_samples)
    3. Clustering de esas K muestras
    4. Seleccionar n_samples distribuidas entre clusters
    """
    def __init__(
        self,
        model,
        strategy: str = 'entropy',
        diversity_weight: float = 0.3,
        device: str = 'auto'
    ):
        """
        Args:
            diversity_weight: Peso de diversidad vs incertidumbre (0-1)
        """
        super().__init__(model, strategy, device)
        self.diversity_weight = diversity_weight
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extrae features de las imágenes para calcular diversidad.
        
        Usa el encoder del modelo si disponible, sino usa las probabilidades.
        """
        with torch.no_grad():
            images = images.to(self.device)
            
            # Intentar obtener features del encoder
            if hasattr(self.model, 'encoder'):
                features = self.model.encoder(images)
                if features.ndim == 3:  # (B, seq_len, hidden)
                    features = features[:, 0, :]  # [CLS] token
            else:
                # Fallback: usar probabilidades como features
                logits = self.model(images)
                if isinstance(logits, dict):
                    first_task = list(logits.keys())[0]
                    logits = logits[first_task]
                features = F.softmax(logits, dim=-1)
        
        return features.cpu()
    
    def select_samples(
        self,
        pool_images: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int,
        batch_size: int = 32,
        top_k_factor: int = 5,
        return_scores: bool = False
    ):
        """
        Selecciona muestras inciertas y diversas.
        
        Args:
            top_k_factor: Multiplicador para candidatos iniciales (K = n_samples * top_k_factor)
        """
        if isinstance(pool_images, list):
            pool_images = torch.stack(pool_images)
        
        n_pool = len(pool_images)
        
        # 1. Calcular incertidumbre
        all_probs = []
        all_features = []
        
        print(f"🔄 Calculando incertidumbre y features para {n_pool} muestras...")
        for i in range(0, n_pool, batch_size):
            batch = pool_images[i:i+batch_size]
            
            # Probabilidades
            probs = self.predict_proba(batch)
            all_probs.append(probs)
            
            # Features
            features = self.extract_features(batch)
            all_features.append(features)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_features = torch.cat(all_features, dim=0)
        
        # Calcular scores de incertidumbre
        uncertainty_scores = self.calculate_uncertainty(all_probs)
        
        # 2. Seleccionar top K más inciertas como candidatas
        top_k = min(n_samples * top_k_factor, n_pool)
        top_k_indices = np.argsort(uncertainty_scores)[-top_k:]
        
        candidate_features = all_features[top_k_indices].numpy()
        candidate_scores = uncertainty_scores[top_k_indices]
        
        print(f"   Candidatas: {len(top_k_indices)}")
        
        # 3. Clustering para diversidad
        n_clusters = min(n_samples, len(candidate_features))
        
        if n_clusters < len(candidate_features):
            print(f"   Aplicando clustering (k={n_clusters})...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(candidate_features)
            
            # 4. Seleccionar una muestra por cluster (la más incierta)
            selected_local = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Seleccionar la más incierta del cluster
                cluster_scores = candidate_scores[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_scores)]
                selected_local.append(best_in_cluster)
            
            selected_indices = top_k_indices[selected_local]
        else:
            # Si K pequeño, retornar todas las candidatas
            selected_indices = top_k_indices
        
        print(f"✅ Seleccionadas {len(selected_indices)} muestras diversas")
        
        if return_scores:
            return selected_indices, uncertainty_scores[selected_indices]
        return selected_indices


class EnsembleUncertaintySampler:
    """
    Sampler que usa desacuerdo entre múltiples modelos (ensemble).
    
    Estrategia:
    1. Cada modelo predice probabilidades
    2. Se calcula desacuerdo (varianza, KL-divergence, etc.)
    3. Se seleccionan muestras con mayor desacuerdo
    
    Mayor desacuerdo = modelos no están de acuerdo = muestra difícil
    """
    def __init__(
        self,
        models: List,
        disagreement_metric: str = 'vote_entropy',
        device: str = 'auto'
    ):
        """
        Args:
            models: Lista de modelos del ensemble
            disagreement_metric: 'vote_entropy', 'kl_divergence', 'variance'
        """
        self.models = models
        self.disagreement_metric = disagreement_metric
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        print(f"✅ EnsembleUncertaintySampler creado con {len(models)} modelos")
    
    def predict_ensemble(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Obtiene predicciones de todos los modelos.
        
        Returns:
            Lista de probabilidades de cada modelo: [(B, C), (B, C), ...]
        """
        all_probs = []
        
        with torch.no_grad():
            images = images.to(self.device)
            
            for model in self.models:
                logits = model(images)
                
                if isinstance(logits, dict):
                    first_task = list(logits.keys())[0]
                    logits = logits[first_task]
                
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu())
        
        return all_probs
    
    def vote_entropy_score(self, all_probs: List[torch.Tensor]) -> np.ndarray:
        """
        Entropía sobre las predicciones (votos) de los modelos.
        
        Para cada muestra:
        1. Obtener clase predicha por cada modelo
        2. Calcular distribución de votos
        3. Calcular entropía de esa distribución
        """
        # Stack predictions: (n_models, batch, num_classes)
        stacked = torch.stack(all_probs, dim=0)
        
        # Obtener clases predichas por cada modelo
        votes = torch.argmax(stacked, dim=-1)  # (n_models, batch)
        
        # Para cada muestra, contar votos
        batch_size = votes.shape[1]
        num_classes = stacked.shape[2]
        
        scores = []
        for b in range(batch_size):
            sample_votes = votes[:, b]  # (n_models,)
            
            # Distribución de votos
            vote_counts = torch.bincount(sample_votes, minlength=num_classes).float()
            vote_dist = vote_counts / vote_counts.sum()
            
            # Entropía
            vote_dist = torch.clamp(vote_dist, min=1e-10)
            entropy = -torch.sum(vote_dist * torch.log(vote_dist))
            scores.append(entropy.item())
        
        return np.array(scores)
    
    def kl_divergence_score(self, all_probs: List[torch.Tensor]) -> np.ndarray:
        """
        KL-divergence promedio entre predicciones de los modelos.
        
        Mayor KL = modelos discrepan más = mayor incertidumbre
        """
        # Promedio de probabilidades
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (batch, num_classes)
        
        # KL de cada modelo respecto al promedio
        kl_scores = []
        for probs in all_probs:
            kl = F.kl_div(
                torch.log(probs + 1e-10),
                mean_probs,
                reduction='none'
            ).sum(dim=-1)  # (batch,)
            kl_scores.append(kl)
        
        # Promedio de KL
        mean_kl = torch.stack(kl_scores, dim=0).mean(dim=0)
        return mean_kl.numpy()
    
    def variance_score(self, all_probs: List[torch.Tensor]) -> np.ndarray:
        """
        Varianza de las probabilidades predichas.
        
        Mayor varianza = mayor desacuerdo = mayor incertidumbre
        """
        # Stack: (n_models, batch, num_classes)
        stacked = torch.stack(all_probs, dim=0)
        
        # Varianza sobre modelos
        variance = torch.var(stacked, dim=0)  # (batch, num_classes)
        
        # Score = suma de varianzas
        scores = variance.sum(dim=-1)
        return scores.numpy()
    
    def calculate_disagreement(self, all_probs: List[torch.Tensor]) -> np.ndarray:
        """
        Calcula score de desacuerdo entre modelos.
        """
        if self.disagreement_metric == 'vote_entropy':
            return self.vote_entropy_score(all_probs)
        
        elif self.disagreement_metric == 'kl_divergence':
            return self.kl_divergence_score(all_probs)
        
        elif self.disagreement_metric == 'variance':
            return self.variance_score(all_probs)
        
        else:
            raise ValueError(f"Unknown metric: {self.disagreement_metric}")
    
    def select_samples(
        self,
        pool_images: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int,
        batch_size: int = 32,
        return_scores: bool = False
    ):
        """
        Selecciona muestras con mayor desacuerdo entre modelos.
        """
        if isinstance(pool_images, list):
            pool_images = torch.stack(pool_images)
        
        n_pool = len(pool_images)
        all_disagreements = []
        
        print(f"🔄 Calculando desacuerdo de ensemble para {n_pool} muestras...")
        
        for i in range(0, n_pool, batch_size):
            batch = pool_images[i:i+batch_size]
            
            # Obtener predicciones del ensemble
            ensemble_probs = self.predict_ensemble(batch)
            
            # Calcular desacuerdo
            disagreement = self.calculate_disagreement(ensemble_probs)
            all_disagreements.append(disagreement)
        
        all_disagreements = np.concatenate(all_disagreements)
        
        # Seleccionar top N
        n_samples = min(n_samples, n_pool)
        selected_indices = np.argsort(all_disagreements)[-n_samples:][::-1]
        
        print(f"✅ Seleccionadas {len(selected_indices)} muestras")
        print(f"   Desacuerdo medio: {all_disagreements[selected_indices].mean():.4f}")
        
        if return_scores:
            return selected_indices, all_disagreements[selected_indices]
        return selected_indices


# =============================================================================
# UTILIDADES
# =============================================================================

def combine_strategies(
    samplers: List[UncertaintySampler],
    weights: List[float],
    pool_images: torch.Tensor,
    n_samples: int
) -> np.ndarray:
    """
    Combina múltiples estrategias de sampling.
    
    Args:
        samplers: Lista de samplers
        weights: Pesos de cada sampler (suman 1.0)
        pool_images: Pool de imágenes
        n_samples: Número a seleccionar
    
    Returns:
        selected_indices
    """
    assert len(samplers) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    
    # Obtener scores de cada sampler
    all_scores = []
    for sampler in samplers:
        _, scores = sampler.select_samples(pool_images, n_samples, return_scores=True)
        all_scores.append(scores)
    
    # Normalizar scores
    all_scores = [s / (s.max() + 1e-10) for s in all_scores]
    
    # Combinar con pesos
    combined_scores = sum(w * s for w, s in zip(weights, all_scores))
    
    # Seleccionar top N
    selected_indices = np.argsort(combined_scores)[-n_samples:][::-1]
    
    return selected_indices

