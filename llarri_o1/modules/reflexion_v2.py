# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Módulo REFLEXIÓN v2 con EARLY EXIT INTELIGENTE

CAMBIO CLAVE: La reflexión ahora analiza LOGITS y TOKENS PREDICHOS,
no solo embeddings. 

Early Exit ocurre SOLO si:
- Durante TRAINING: el token predicho coincide con el target
- Durante INFERENCE: el token predicho pasa validaciones de calidad

Concepto: "¿La respuesta predicha es correcta? ¿Puedo salir ya?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
import math


class ResultadoReflexion(NamedTuple):
    """Resultado de una caja de reflexión"""
    output: torch.Tensor  # Output embeddings (corregido o no)
    logits: torch.Tensor  # Logits (corregidos o no)
    score_confianza: torch.Tensor  # [batch] score de confianza
    early_exit: torch.Tensor  # [batch] bool, True si puede salir
    metricas: dict  # Métricas para debug


class CajaDetectorLogits(nn.Module):
    """
    CAJA REFLEXIÓN - DETECTOR (basado en LOGITS)
    
    Analiza los logits/probabilidades para detectar problemas:
    - Entropía muy baja = colapso (malo)
    - Entropía muy alta = confusión (malo)
    - Repetición del token anterior (malo)
    - Probabilidad máxima sospechosa (malo)
    """
    
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Thresholds configurables
        self.entropy_min = 0.3  # Debajo = colapso
        self.entropy_max = 4.5  # Arriba = muy confuso
        self.prob_max_threshold = 0.98  # Arriba = sospechoso
        
    def forward(
        self,
        logits: torch.Tensor,
        historial: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            historial: [batch, n_tokens] - tokens ya generados
            targets: [batch, seq_len] - targets reales (solo en training)
        Returns:
            metricas: dict con scores por métrica
        """
        batch, seq_len, vocab = logits.shape
        device = logits.device
        
        # Tomar solo el último token (el que estamos prediciendo)
        logits_last = logits[:, -1, :]  # [batch, vocab]
        probs = F.softmax(logits_last, dim=-1)
        
        # 1. ENTROPÍA REAL sobre distribución de vocabulario
        entropia = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch]
        entropia_normalizada = entropia / math.log(vocab)
        
        # Score de entropía: rango óptimo [0.3, 0.7] normalizado
        score_entropia = torch.ones(batch, device=device)
        score_entropia = torch.where(
            entropia_normalizada < 0.1,  # Muy baja = colapso
            torch.tensor(0.0, device=device),
            score_entropia
        )
        score_entropia = torch.where(
            entropia_normalizada > 0.9,  # Muy alta = confusión
            torch.tensor(0.3, device=device),
            score_entropia
        )
        
        # 2. TOKEN PREDICHO
        token_predicho = logits_last.argmax(dim=-1)  # [batch]
        prob_max = probs.max(dim=-1).values  # [batch]
        
        # Score de probabilidad: muy alta es sospechosa
        score_prob = torch.where(
            prob_max > self.prob_max_threshold,
            torch.tensor(0.3, device=device),  # Sospechoso
            torch.ones(batch, device=device)
        )
        
        # 3. REPETICIÓN: ¿el token predicho es igual al anterior?
        score_repeticion = torch.ones(batch, device=device)
        if historial is not None and historial.shape[1] > 0:
            token_anterior = historial[:, -1]
            es_repeticion = (token_predicho == token_anterior).float()
            score_repeticion = 1.0 - es_repeticion * 0.5  # Penalizar repetición
            
            # Penalizar más si aparece muchas veces en historial reciente
            ultimos_10 = historial[:, -min(10, historial.shape[1]):]
            for b in range(batch):
                cuenta = (ultimos_10[b] == token_predicho[b]).sum().float()
                if cuenta > 2:
                    score_repeticion[b] *= 0.5  # Penalizar fuerte
        
        # 4. COINCIDENCIA CON TARGET (solo en training)
        score_target = torch.ones(batch, device=device)
        coincide_target = torch.zeros(batch, device=device, dtype=torch.bool)
        if targets is not None:
            target_last = targets[:, -1]  # [batch]
            coincide_target = (token_predicho == target_last)
            score_target = coincide_target.float()
        
        # 5. VALIDACIÓN BÁSICA DEL TOKEN
        # Caracteres ASCII imprimibles: 32-126, o newline/tab
        es_valido = (
            ((token_predicho >= 32) & (token_predicho <= 126)) |
            (token_predicho == 10) |  # newline
            (token_predicho == 9) |   # tab
            (token_predicho == 13)    # carriage return
        )
        score_validez = es_valido.float()
        
        return {
            'score_entropia': score_entropia,
            'score_prob': score_prob,
            'score_repeticion': score_repeticion,
            'score_target': score_target,
            'score_validez': score_validez,
            'entropia': entropia,
            'entropia_normalizada': entropia_normalizada,
            'token_predicho': token_predicho,
            'prob_max': prob_max,
            'coincide_target': coincide_target
        }


class CajaEvaluadorInteligente(nn.Module):
    """
    CAJA REFLEXIÓN - EVALUADOR INTELIGENTE
    
    Decide si hacer early exit basándose en:
    - TRAINING: ¿coincide con el target?
    - INFERENCE: ¿pasa todas las validaciones?
    """
    
    def __init__(self):
        super().__init__()
        
        # Pesos para combinar métricas (inference)
        self.peso_entropia = 0.25
        self.peso_prob = 0.15
        self.peso_repeticion = 0.30
        self.peso_validez = 0.30
        
        # Threshold para early exit
        self.threshold_training = 1.0  # Solo si coincide EXACTO con target
        self.threshold_inference = 0.85
        
    def forward(
        self,
        metricas: dict,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            metricas: dict de CajaDetectorLogits
            is_training: si estamos en modo training
        Returns:
            score_final: [batch] - score de confianza 0-1
            early_exit_mask: [batch] - True si puede salir
        """
        batch = metricas['score_entropia'].shape[0]
        device = metricas['score_entropia'].device
        
        if is_training:
            # En TRAINING: early exit SOLO si coincide con target
            score_final = metricas['score_target']
            early_exit_mask = metricas['coincide_target']
        else:
            # En INFERENCE: combinar todas las métricas
            score_final = (
                self.peso_entropia * metricas['score_entropia'] +
                self.peso_prob * metricas['score_prob'] +
                self.peso_repeticion * metricas['score_repeticion'] +
                self.peso_validez * metricas['score_validez']
            )
            
            # Early exit solo si pasa TODAS las validaciones básicas
            early_exit_mask = (
                (metricas['score_entropia'] > 0.5) &
                (metricas['score_repeticion'] > 0.7) &
                (metricas['score_validez'] > 0.5) &
                (score_final > self.threshold_inference)
            )
        
        return score_final, early_exit_mask


class CajaCorrector(nn.Module):
    """
    CAJA REFLEXIÓN - CORRECTOR
    
    Ajusta los logits cuando detecta problemas:
    - Baja entropía → aumentar temperatura
    - Repetición → penalizar token repetido
    - Colapso → redistribuir probabilidad
    """
    
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.repetition_penalty = 1.5
        self.temperature_boost = 1.5  # Para cuando hay baja entropía
        
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        metricas: dict,
        historial: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            embeddings: [batch, seq_len, embed_dim]
            metricas: dict de CajaDetectorLogits
            historial: [batch, n_tokens]
        Returns:
            logits_corregidos: [batch, seq_len, vocab_size]
            embeddings_out: [batch, seq_len, embed_dim] (sin cambios)
        """
        batch, seq_len, vocab = logits.shape
        device = logits.device
        
        logits_corregidos = logits.clone()
        
        # Solo corregir el último token
        logits_last = logits_corregidos[:, -1, :]  # [batch, vocab]
        
        for b in range(batch):
            # 1. Si entropía muy baja → aumentar temperatura
            if metricas['entropia_normalizada'][b] < 0.15:
                logits_last[b] = logits_last[b] / self.temperature_boost
            
            # 2. Penalizar repetición
            if historial is not None and historial.shape[1] > 0:
                # Penalizar tokens recientes
                ultimos = historial[b, -min(15, historial.shape[1]):]
                for token in ultimos.unique():
                    cuenta = (ultimos == token).sum().item()
                    if cuenta > 0:
                        penalty = self.repetition_penalty ** cuenta
                        logits_last[b, token] = logits_last[b, token] / penalty
            
            # 3. Si probabilidad máxima muy alta y no es válido → redistribuir
            if metricas['prob_max'][b] > 0.95 and metricas['score_validez'][b] < 0.5:
                # El token dominante no es válido, redistribuir
                logits_last[b] = logits_last[b] / 2.0  # Suavizar mucho
        
        logits_corregidos[:, -1, :] = logits_last
        
        return logits_corregidos, embeddings


class ModuloReflexionV2(nn.Module):
    """
    Módulo de Reflexión v2 - Analiza LOGITS no embeddings
    
    Decide early exit basándose en si la predicción es CORRECTA,
    no solo si los embeddings son "coherentes".
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 256,
        nombre: str = "ReflexionV2"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.nombre = nombre
        
        # Las 3 cajas de reflexión
        self.detector = CajaDetectorLogits(vocab_size)
        self.evaluador = CajaEvaluadorInteligente()
        self.corrector = CajaCorrector(vocab_size)
        
        print(f"✓ ModuloReflexionV2 '{nombre}': analiza LOGITS, early_exit solo si correcto")
        
    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        historial: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> ResultadoReflexion:
        """
        Args:
            embeddings: [batch, seq_len, embed_dim] - output de las cajas neurales
            logits: [batch, seq_len, vocab_size] - logits preliminares
            historial: [batch, n_tokens] - tokens ya generados
            targets: [batch, seq_len] - targets reales (solo training)
        Returns:
            ResultadoReflexion con output, logits, scores, y flag de early exit
        """
        is_training = targets is not None
        
        # Caja 1: Detectar problemas en los logits
        metricas = self.detector(logits, historial, targets)
        
        # Caja 2: Evaluar si hacer early exit
        score_confianza, early_exit = self.evaluador(metricas, is_training)
        
        # Caja 3: Corregir si hay problemas
        logits_corregidos, embeddings_out = self.corrector(
            logits, embeddings, metricas, historial
        )
        
        return ResultadoReflexion(
            output=embeddings_out,
            logits=logits_corregidos,
            score_confianza=score_confianza,
            early_exit=early_exit,
            metricas=metricas
        )


class ModuloReflexionFinalV2(ModuloReflexionV2):
    """
    Reflexión Final con correcciones adicionales garantizadas.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 256
    ):
        super().__init__(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            nombre="ReflexionFinalV2"
        )
        
        # Corrector más agresivo para la reflexión final
        self.corrector.repetition_penalty = 2.0
        self.corrector.temperature_boost = 2.0


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Test ModuloReflexionV2")
    print("=" * 60)
    
    reflexion = ModuloReflexionV2(embed_dim=128, vocab_size=256)
    
    # Simular datos
    batch, seq_len = 4, 64
    embeddings = torch.randn(batch, seq_len, 128)
    logits = torch.randn(batch, seq_len, 256)
    targets = torch.randint(0, 256, (batch, seq_len))
    historial = torch.randint(0, 256, (batch, 20))
    
    # Test con targets (training)
    print("\n--- Test TRAINING (con targets) ---")
    resultado = reflexion(embeddings, logits, historial, targets)
    print(f"Score confianza: {resultado.score_confianza}")
    print(f"Early exit: {resultado.early_exit}")
    print(f"Coincide target: {resultado.metricas['coincide_target']}")
    
    # Test sin targets (inference)
    print("\n--- Test INFERENCE (sin targets) ---")
    resultado2 = reflexion(embeddings, logits, historial, None)
    print(f"Score confianza: {resultado2.score_confianza}")
    print(f"Early exit: {resultado2.early_exit}")
    
    # Simular caso de colapso (un token domina)
    print("\n--- Test COLAPSO (un token domina) ---")
    logits_colapso = torch.full((batch, seq_len, 256), -10.0)
    logits_colapso[:, :, 101] = 10.0  # Token 101 = 'e' domina
    resultado3 = reflexion(embeddings, logits_colapso, historial, None)
    print(f"Entropía normalizada: {resultado3.metricas['entropia_normalizada']}")
    print(f"Score entropía: {resultado3.metricas['score_entropia']}")
    print(f"Early exit: {resultado3.early_exit}")  # Debería ser False
    
    print("\n✅ ModuloReflexionV2 funcionando!")
