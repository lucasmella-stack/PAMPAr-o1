# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Módulo REFLEXIÓN con EARLY EXIT

Las cajas de reflexión comparan el output neuronal con el input original
para decidir:
1. Si ya está bien → EARLY EXIT (ahorrar cómputo)
2. Si está mal → CORREGIR (revertir hacia input)
3. Si está medio → CONTINUAR (seguir procesando)

Concepto: "¿Lo que aprendí tiene sentido? ¿Puedo salir ya?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
import math


class ResultadoReflexion(NamedTuple):
    """Resultado de una caja de reflexión"""
    output: torch.Tensor  # Output (corregido o no)
    score_confianza: torch.Tensor  # [batch] score de confianza
    early_exit: torch.Tensor  # [batch] bool, True si puede salir
    metricas: dict  # Métricas para debug


class CajaDetectarDiferencias(nn.Module):
    """
    CAJA REFLEXIÓN - DETECTOR
    
    Compara output neuronal vs input original.
    Calcula métricas de diferencia para decidir calidad.
    """
    
    def __init__(self, embed_dim: int = 128, n_escalas: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_escalas = n_escalas
        
    def forward(
        self,
        output_neural: torch.Tensor,
        input_original: torch.Tensor,
        historial: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            output_neural: [batch, seq_len, embed_dim] - lo que produjo la neuronal
            input_original: [batch, seq_len, embed_dim] - input sin modificar
            historial: [batch, n_tokens] - tokens ya generados (para repetición)
        Returns:
            metricas: dict con scores por métrica
        """
        batch = output_neural.shape[0]
        
        # 1. Diferencia de norma (¿cuánto cambió?)
        diff = output_neural - input_original
        norma_diff = diff.norm(dim=(1, 2))  # [batch]
        norma_input = input_original.norm(dim=(1, 2)) + 1e-6
        ratio_cambio = norma_diff / norma_input  # Cambio relativo
        
        # Score: cambio moderado es bueno (0.1-0.5), extremos son malos
        score_cambio = 1.0 - torch.abs(ratio_cambio - 0.3).clamp(0, 1)
        
        # 2. Entropía del output (¿está seguro pero no demasiado?)
        # Proyectamos a una distribución simplificada
        output_flat = output_neural.view(batch, -1)
        probs = F.softmax(output_flat[:, :256], dim=-1)  # Primeros 256 valores
        entropia = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        entropia_normalizada = entropia / math.log(256)  # Normalizar a [0,1]
        
        # Score: entropía media es buena (0.3-0.7)
        score_entropia = 1.0 - torch.abs(entropia_normalizada - 0.5).clamp(0, 0.5) * 2
        
        # 3. Coherencia direccional (¿el cambio va en dirección coherente?)
        # Coseno entre diff y input
        cos_sim = F.cosine_similarity(
            diff.view(batch, -1),
            input_original.view(batch, -1),
            dim=-1
        )
        score_coherencia = (cos_sim + 1) / 2  # Mapear [-1,1] a [0,1]
        
        # 4. Score de repetición (si hay historial)
        if historial is not None and historial.shape[1] > 0:
            # Contar tokens únicos en historial reciente
            ultimos = historial[:, -min(20, historial.shape[1]):]
            n_unicos = torch.tensor([
                len(torch.unique(ultimos[b])) 
                for b in range(batch)
            ], device=output_neural.device, dtype=torch.float)
            score_repeticion = n_unicos / ultimos.shape[1]  # Ratio de únicos
        else:
            score_repeticion = torch.ones(batch, device=output_neural.device)
        
        return {
            'score_cambio': score_cambio,
            'score_entropia': score_entropia,
            'score_coherencia': score_coherencia,
            'score_repeticion': score_repeticion,
            'ratio_cambio': ratio_cambio,
            'entropia': entropia_normalizada
        }


class CajaEvaluarValidez(nn.Module):
    """
    CAJA REFLEXIÓN - EVALUADOR
    
    Evalúa si los cambios de la neuronal son válidos.
    Usa las métricas del detector para calcular un score final.
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        # Pesos aprendibles para combinar métricas
        self.pesos = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        
        # Thresholds
        self.threshold_alto = 0.85  # Si score > esto, early exit
        self.threshold_bajo = 0.35  # Si score < esto, corregir fuerte
        
    def forward(self, metricas: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            metricas: dict de CajaDetectarDiferencias
        Returns:
            score_final: [batch] - score de confianza 0-1
            early_exit_mask: [batch] - True si puede salir
        """
        # Normalizar pesos
        pesos = F.softmax(self.pesos, dim=0)
        
        # Combinar scores
        score_final = (
            pesos[0] * metricas['score_cambio'] +
            pesos[1] * metricas['score_entropia'] +
            pesos[2] * metricas['score_coherencia'] +
            pesos[3] * metricas['score_repeticion']
        )
        
        # Determinar si puede hacer early exit
        early_exit_mask = score_final > self.threshold_alto
        
        return score_final, early_exit_mask


class CajaCorregirValidar(nn.Module):
    """
    CAJA REFLEXIÓN - CORRECTOR/VALIDADOR
    
    Si score alto: mantener output (validar)
    Si score bajo: mezclar con input original (corregir)
    Si score medio: continuar sin cambios
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.threshold_bajo = 0.35
        
        # Factor de corrección aprendible
        self.factor_correccion = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        output_neural: torch.Tensor,
        input_original: torch.Tensor,
        score_confianza: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            output_neural: [batch, seq_len, embed_dim]
            input_original: [batch, seq_len, embed_dim]
            score_confianza: [batch]
        Returns:
            output_corregido: [batch, seq_len, embed_dim]
        """
        batch = output_neural.shape[0]
        
        # Calcular alpha de mezcla basado en score
        # Score alto → alpha alto → mantener output
        # Score bajo → alpha bajo → revertir a input
        alpha = score_confianza.view(batch, 1, 1)
        
        # Para scores muy bajos, corregir más agresivamente
        mask_bajo = (score_confianza < self.threshold_bajo).view(batch, 1, 1)
        alpha_corregido = torch.where(
            mask_bajo,
            alpha * self.factor_correccion.clamp(0.1, 0.9),
            alpha
        )
        
        # Mezcla: alpha * output + (1-alpha) * input
        output_corregido = alpha_corregido * output_neural + (1 - alpha_corregido) * input_original
        
        return output_corregido


class ModuloReflexion(nn.Module):
    """
    Módulo de Reflexión completo (3 cajas)
    
    Compara output neuronal con input original para:
    - Detectar problemas
    - Evaluar calidad
    - Corregir si necesario
    - Decidir early exit
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        n_escalas: int = 8,
        threshold_alto: float = 0.85,
        threshold_bajo: float = 0.35,
        nombre: str = "Reflexion"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nombre = nombre
        
        # Las 3 cajas de reflexión
        self.detector = CajaDetectarDiferencias(embed_dim, n_escalas)
        self.evaluador = CajaEvaluarValidez(embed_dim)
        self.corrector = CajaCorregirValidar(embed_dim)
        
        # Configurar thresholds
        self.evaluador.threshold_alto = threshold_alto
        self.evaluador.threshold_bajo = threshold_bajo
        self.corrector.threshold_bajo = threshold_bajo
        
        print(f"✓ ModuloReflexion '{nombre}': early_exit>{threshold_alto:.2f}, corregir<{threshold_bajo:.2f}")
        
    def forward(
        self,
        output_neural: torch.Tensor,
        input_original: torch.Tensor,
        historial: Optional[torch.Tensor] = None
    ) -> ResultadoReflexion:
        """
        Args:
            output_neural: [batch, seq_len, embed_dim] - output de las cajas neurales
            input_original: [batch, seq_len, embed_dim] - input original (residuo)
            historial: [batch, n_tokens] - tokens ya generados
        Returns:
            ResultadoReflexion con output, scores, y flag de early exit
        """
        # Caja 1: Detectar diferencias
        metricas = self.detector(output_neural, input_original, historial)
        
        # Caja 2: Evaluar validez
        score_confianza, early_exit = self.evaluador(metricas)
        
        # Caja 3: Corregir si necesario
        output_final = self.corrector(output_neural, input_original, score_confianza)
        
        return ResultadoReflexion(
            output=output_final,
            score_confianza=score_confianza,
            early_exit=early_exit,
            metricas=metricas
        )


class ModuloReflexionFinal(ModuloReflexion):
    """
    Reflexión Final (Cajas 25-26) con correcciones matemáticas adicionales.
    
    Además de la reflexión normal, aplica:
    - Penalización de repetición en logits
    - Temperatura adaptativa
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 256,
        n_escalas: int = 8
    ):
        super().__init__(
            embed_dim=embed_dim,
            n_escalas=n_escalas,
            threshold_alto=0.90,  # Más estricto para la final
            threshold_bajo=0.40,
            nombre="ReflexionFinal"
        )
        
        self.vocab_size = vocab_size
        
        # Penalización de repetición
        self.repetition_penalty = 1.2
        
    def aplicar_penalizacion_repeticion(
        self,
        logits: torch.Tensor,
        historial: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Penaliza tokens que aparecen en el historial reciente."""
        if historial is None or historial.shape[1] == 0:
            return logits
            
        batch, vocab = logits.shape
        
        # Contar ocurrencias en historial reciente
        ultimos = historial[:, -min(30, historial.shape[1]):]
        
        for b in range(batch):
            tokens_recientes = ultimos[b].unique()
            for t in tokens_recientes:
                if t < vocab:
                    # Penalizar dividiendo el logit
                    logits[b, t] = logits[b, t] / self.repetition_penalty
        
        return logits
    
    def calcular_temperatura_adaptativa(self, score_confianza: torch.Tensor) -> torch.Tensor:
        """
        Temperatura adaptativa basada en confianza.
        Score bajo → más temperatura (explorar)
        Score alto → menos temperatura (explotar)
        """
        # Mapear score [0,1] a temperatura [1.5, 0.7]
        temperatura = 1.5 - score_confianza * 0.8
        return temperatura.clamp(0.5, 2.0)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Test ModuloReflexion")
    print("=" * 60)
    
    reflexion = ModuloReflexion(embed_dim=128)
    reflexion_final = ModuloReflexionFinal(embed_dim=128)
    
    # Simular input y output neuronal
    batch, seq_len, embed_dim = 4, 64, 128
    input_original = torch.randn(batch, seq_len, embed_dim)
    
    # Caso 1: Output muy similar al input (debería dar early exit)
    output_similar = input_original + 0.1 * torch.randn_like(input_original)
    resultado1 = reflexion(output_similar, input_original)
    print(f"\nCaso 1 (output similar):")
    print(f"  Score confianza: {resultado1.score_confianza.mean():.3f}")
    print(f"  Early exit: {resultado1.early_exit.sum()}/{batch}")
    
    # Caso 2: Output muy diferente (debería corregir)
    output_diferente = torch.randn_like(input_original) * 5
    resultado2 = reflexion(output_diferente, input_original)
    print(f"\nCaso 2 (output muy diferente):")
    print(f"  Score confianza: {resultado2.score_confianza.mean():.3f}")
    print(f"  Early exit: {resultado2.early_exit.sum()}/{batch}")
    
    # Caso 3: Output moderado
    output_moderado = input_original * 0.7 + torch.randn_like(input_original) * 0.3
    resultado3 = reflexion(output_moderado, input_original)
    print(f"\nCaso 3 (output moderado):")
    print(f"  Score confianza: {resultado3.score_confianza.mean():.3f}")
    print(f"  Early exit: {resultado3.early_exit.sum()}/{batch}")
    
    # Verificar parámetros
    params = sum(p.numel() for p in reflexion.parameters())
    print(f"\nParámetros Reflexion: {params:,}")
    
    params_final = sum(p.numel() for p in reflexion_final.parameters())
    print(f"Parámetros ReflexionFinal: {params_final:,}")
    
    print("\n✅ ModuloReflexion funcionando!")
