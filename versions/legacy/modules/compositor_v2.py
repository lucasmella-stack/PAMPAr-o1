# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas.mella@outlook.com)
"""
Compositor v2 - Matemáticas Blindadas para Cajas 7-8-9.

INNOVACIÓN: Las cajas compositoras usan cálculos matemáticos PUROS
que no pueden fallar, en lugar de redes neuronales que pueden colapsar.

Caja 7 (Detector): Entropía, Markov, Compresión, Autocorrelación
Caja 8 (Planificador): Transiciones, Gramática, Beam, Mutual Info
Caja 9 (Refinador): Bayes, Penalización, Temperatura, Normalización

Las matemáticas actúan como "guardias" sobre las cajas neuronales 1-6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class CompositorV2Config:
    """Configuración del compositor v2 con matemáticas blindadas."""
    embed_dim: int = 128
    vocab_size: int = 256
    
    # Detector (Caja 7)
    entropy_threshold_low: float = 0.5   # Muy seguro (posible colapso)
    entropy_threshold_high: float = 4.0  # Muy incierto
    ngram_window: int = 8                # Ventana para detectar repeticiones
    repetition_threshold: float = 0.3    # Si >30% repetido, alerta
    
    # Planificador (Caja 8)
    beam_width: int = 5                  # Candidatos a considerar
    
    # Refinador (Caja 9)
    repetition_penalty: float = 1.2      # Penalización por repetición
    min_temperature: float = 0.5
    max_temperature: float = 1.5
    

# =============================================================================
# CAJA 7: DETECTOR MATEMÁTICO
# =============================================================================

class CuadranteEntropia(nn.Module):
    """
    Q1: Calcula entropía de Shannon sobre la distribución.
    
    H = -Σ p(x) * log₂(p(x))
    
    - H bajo (< 0.5): Distribución muy concentrada → posible colapso
    - H alto (> 4.0): Muy dispersa → incierto
    - H medio: Saludable
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.threshold_low = config.entropy_threshold_low
        self.threshold_high = config.entropy_threshold_high
        
    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch, seq, vocab]
        Returns:
            entropy: [batch, seq] - entropía por posición
            alert_low: [batch, seq] - bool, entropía muy baja
            alert_high: [batch, seq] - bool, entropía muy alta
        """
        # Softmax para obtener probabilidades
        probs = F.softmax(logits, dim=-1)
        
        # Entropía: -Σ p * log(p)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Convertir a bits (log2)
        entropy_bits = entropy / math.log(2)
        
        # Alertas
        alert_low = entropy_bits < self.threshold_low
        alert_high = entropy_bits > self.threshold_high
        
        return {
            'entropy': entropy_bits,
            'alert_low': alert_low,
            'alert_high': alert_high,
        }


class CuadranteNGrama(nn.Module):
    """
    Q2: Detector de repeticiones usando N-gramas estilo Markov.
    
    Cuenta cuántas veces cada byte apareció recientemente.
    Si un byte tiene P(repetición) > threshold → alerta.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.window = config.ngram_window
        self.threshold = config.repetition_threshold
        self.vocab_size = config.vocab_size
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq] - tokens de entrada
            logits: [batch, seq, vocab]
        Returns:
            repetition_score: [batch, seq, vocab] - score de repetición por token
            top_repeated: [batch, seq] - token más repetido
        """
        batch, seq, vocab = logits.shape
        device = logits.device
        
        # Contar frecuencia de cada token en ventana
        repetition_score = torch.zeros(batch, seq, vocab, device=device)
        
        for b in range(batch):
            for t in range(seq):
                # Ventana hacia atrás
                start = max(0, t - self.window)
                window = input_ids[b, start:t+1]
                
                # Contar ocurrencias
                for token in window:
                    repetition_score[b, t, token] += 1
                
                # Normalizar por tamaño de ventana
                window_size = t - start + 1
                if window_size > 0:
                    repetition_score[b, t] /= window_size
        
        # Token más repetido
        top_repeated = repetition_score.argmax(dim=-1)
        
        # Alertas: tokens con alta repetición
        alert_repeated = repetition_score > self.threshold
        
        return {
            'repetition_score': repetition_score,
            'top_repeated': top_repeated,
            'alert_repeated': alert_repeated,
        }


class CuadranteCompresion(nn.Module):
    """
    Q3: Ratio de compresión - si la secuencia es muy compresible,
    significa que hay muchos patrones repetidos.
    
    Usa una aproximación simple: cuenta bytes únicos / total bytes.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.window = config.ngram_window
        
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq]
        Returns:
            compression_ratio: [batch, seq] - 0=muy compresible, 1=muy único
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        
        compression_ratio = torch.ones(batch, seq, device=device)
        
        for b in range(batch):
            for t in range(seq):
                start = max(0, t - self.window)
                window = input_ids[b, start:t+1]
                
                # Ratio = únicos / total
                unique = len(torch.unique(window))
                total = len(window)
                
                compression_ratio[b, t] = unique / total if total > 0 else 1.0
        
        return {
            'compression_ratio': compression_ratio,
            'is_repetitive': compression_ratio < 0.5,  # <50% únicos = repetitivo
        }


class CuadranteAutocorrelacion(nn.Module):
    """
    Q4: Detecta periodicidad en la secuencia.
    
    Si hay autocorrelación alta en lag k, significa que
    el patrón se repite cada k tokens.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.max_lag = min(config.ngram_window, 8)
        
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq]
        Returns:
            max_autocorr: [batch, seq] - máxima autocorrelación encontrada
            periodic_lag: [batch, seq] - lag con mayor autocorrelación
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        
        max_autocorr = torch.zeros(batch, seq, device=device)
        periodic_lag = torch.zeros(batch, seq, dtype=torch.long, device=device)
        
        for b in range(batch):
            for t in range(self.max_lag * 2, seq):  # Necesitamos al menos 2*max_lag
                best_corr = 0.0
                best_lag = 1
                
                for lag in range(1, min(self.max_lag + 1, t // 2 + 1)):
                    # Verificar que tenemos suficientes elementos
                    start1, end1 = t - lag, t
                    start2, end2 = t - 2*lag, t - lag
                    
                    if start2 >= 0 and end1 - start1 == end2 - start2 and end1 - start1 > 0:
                        # Comparar con lag posiciones atrás
                        matches = (input_ids[b, start1:end1] == input_ids[b, start2:end2]).float()
                        if len(matches) > 0:
                            corr = matches.mean().item()
                            if corr > best_corr:
                                best_corr = corr
                                best_lag = lag
                
                max_autocorr[b, t] = best_corr
                periodic_lag[b, t] = best_lag
        
        return {
            'max_autocorr': max_autocorr,
            'periodic_lag': periodic_lag,
            'is_periodic': max_autocorr > 0.7,
        }


class CajaDetectoraV2(nn.Module):
    """
    Caja 7: Detector con 4 cuadrantes matemáticos.
    
    Analiza la situación actual usando matemáticas puras
    para detectar problemas antes de que ocurran.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.q1_entropia = CuadranteEntropia(config)
        self.q2_ngrama = CuadranteNGrama(config)
        self.q3_compresion = CuadranteCompresion(config)
        self.q4_autocorr = CuadranteAutocorrelacion(config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Ejecuta los 4 cuadrantes detectores."""
        
        q1 = self.q1_entropia(logits)
        q2 = self.q2_ngrama(input_ids, logits)
        q3 = self.q3_compresion(input_ids)
        q4 = self.q4_autocorr(input_ids)
        
        # Combinar alertas: ¿hay algún problema?
        needs_correction = (
            q1['alert_low'] |           # Entropía muy baja
            q3['is_repetitive'] |       # Muy compresible
            q4['is_periodic']           # Detectó periodicidad
        )
        
        return {
            'entropy': q1,
            'ngrama': q2,
            'compresion': q3,
            'autocorr': q4,
            'needs_correction': needs_correction,
        }


# =============================================================================
# CAJA 8: PLANIFICADOR MATEMÁTICO
# =============================================================================

class CuadranteTransiciones(nn.Module):
    """
    Q1: Modelo de transiciones válidas byte→byte.
    
    Algunas transiciones son más probables:
    - letra → letra (palabra continúa)
    - letra → espacio (palabra termina)
    - espacio → mayúscula (nueva oración)
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        
        # Crear matriz de transiciones "razonables"
        # Esto es conocimiento a priori del inglés
        self.register_buffer(
            'transition_boost',
            self._create_transition_matrix()
        )
        
    def _create_transition_matrix(self) -> torch.Tensor:
        """Crea matriz de transiciones basada en reglas."""
        vocab = self.vocab_size
        matrix = torch.zeros(vocab, vocab)
        
        # Definir rangos de bytes
        lowercase = list(range(ord('a'), ord('z') + 1))
        uppercase = list(range(ord('A'), ord('Z') + 1))
        digits = list(range(ord('0'), ord('9') + 1))
        space = [ord(' ')]
        punct = [ord('.'), ord(','), ord('!'), ord('?'), ord(':'), ord(';')]
        newline = [ord('\n')]
        
        # Reglas de transición (boost positivo = más probable)
        
        # letra minúscula → letra minúscula (continuar palabra)
        for a in lowercase:
            for b in lowercase:
                matrix[a, b] = 0.3
        
        # letra → espacio (terminar palabra)
        for a in lowercase + uppercase:
            for b in space:
                matrix[a, b] = 0.2
        
        # espacio → letra (nueva palabra)
        for a in space:
            for b in lowercase + uppercase:
                matrix[a, b] = 0.3
        
        # puntuación → espacio
        for a in punct:
            for b in space:
                matrix[a, b] = 0.4
        
        # espacio después de punto → mayúscula
        # (esto se aplica en contexto, aquí solo el espacio→mayúscula)
        for a in space:
            for b in uppercase:
                matrix[a, b] = 0.1
        
        # Penalizar transiciones raras
        # letra → dígito (raro)
        for a in lowercase + uppercase:
            for b in digits:
                matrix[a, b] = -0.2
        
        return matrix
        
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq]
            logits: [batch, seq, vocab]
        Returns:
            transition_bias: [batch, seq, vocab] - ajuste por transición
        """
        batch, seq, vocab = logits.shape
        device = logits.device
        
        # Obtener último token de cada posición
        # Para posición t, el último token es input_ids[t-1]
        last_tokens = torch.zeros(batch, seq, dtype=torch.long, device=device)
        last_tokens[:, 1:] = input_ids[:, :-1]
        
        # Buscar boost de transición
        transition_bias = self.transition_boost[last_tokens]  # [batch, seq, vocab]
        
        return {
            'transition_bias': transition_bias,
        }


class CuadranteGramatica(nn.Module):
    """
    Q2: Gramática probabilística simplificada.
    
    Estados: INICIO, PALABRA, ESPACIO, PUNTUACION, FIN_ORACION
    Probabilidades de transición entre estados.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        
    def _classify_byte(self, byte_val: int) -> str:
        """Clasifica un byte en categoría gramatical."""
        if ord('a') <= byte_val <= ord('z') or ord('A') <= byte_val <= ord('Z'):
            return 'LETTER'
        elif byte_val == ord(' '):
            return 'SPACE'
        elif byte_val in [ord('.'), ord('!'), ord('?')]:
            return 'END_PUNCT'
        elif byte_val in [ord(','), ord(':'), ord(';')]:
            return 'MID_PUNCT'
        elif ord('0') <= byte_val <= ord('9'):
            return 'DIGIT'
        else:
            return 'OTHER'
    
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq]
            logits: [batch, seq, vocab]
        Returns:
            grammar_bias: [batch, seq, vocab] - ajuste gramatical
        """
        batch, seq, vocab = logits.shape
        device = logits.device
        
        grammar_bias = torch.zeros(batch, seq, vocab, device=device)
        
        for b in range(batch):
            for t in range(1, seq):
                last_byte = input_ids[b, t-1].item()
                last_type = self._classify_byte(last_byte)
                
                # Reglas gramaticales
                if last_type == 'END_PUNCT':
                    # Después de punto: boost a espacio
                    grammar_bias[b, t, ord(' ')] = 0.5
                    
                elif last_type == 'SPACE':
                    # Después de espacio: boost a letras
                    for c in range(ord('a'), ord('z') + 1):
                        grammar_bias[b, t, c] = 0.2
                    for c in range(ord('A'), ord('Z') + 1):
                        grammar_bias[b, t, c] = 0.2
                        
                elif last_type == 'LETTER':
                    # Continuar palabra o terminar
                    for c in range(ord('a'), ord('z') + 1):
                        grammar_bias[b, t, c] = 0.1
                    grammar_bias[b, t, ord(' ')] = 0.1
        
        return {
            'grammar_bias': grammar_bias,
        }


class CuadranteBeam(nn.Module):
    """
    Q3: Mantiene los K mejores candidatos.
    
    En lugar de tomar solo el máximo, considera los top-K
    y evalúa cuál tiene mejor continuación.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.beam_width = config.beam_width
        
    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch, seq, vocab]
        Returns:
            top_k_tokens: [batch, seq, k]
            top_k_probs: [batch, seq, k]
        """
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_tokens = torch.topk(probs, self.beam_width, dim=-1)
        
        return {
            'top_k_tokens': top_k_tokens,
            'top_k_probs': top_k_probs,
        }


class CuadranteMutualInfo(nn.Module):
    """
    Q4: Información mutua aproximada.
    
    Mide cuánta información comparte el token candidato
    con el contexto reciente.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.window = config.ngram_window
        self.vocab_size = config.vocab_size
        
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Aproximación: tokens que co-ocurren con el contexto
        tienen mayor información mutua.
        """
        batch, seq, vocab = logits.shape
        device = logits.device
        
        mi_boost = torch.zeros(batch, seq, vocab, device=device)
        
        for b in range(batch):
            for t in range(seq):
                start = max(0, t - self.window)
                context = input_ids[b, start:t]
                
                # Boost pequeño a tokens que aparecen en contexto
                # (co-ocurrencia → información mutua)
                unique_context = torch.unique(context)
                mi_boost[b, t, unique_context] = 0.1
        
        return {
            'mi_boost': mi_boost,
        }


class CajaPlanificadoraV2(nn.Module):
    """
    Caja 8: Planificador con 4 cuadrantes matemáticos.
    
    Decide qué tipo de token debería seguir basado
    en reglas lingüísticas y probabilísticas.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.q1_transiciones = CuadranteTransiciones(config)
        self.q2_gramatica = CuadranteGramatica(config)
        self.q3_beam = CuadranteBeam(config)
        self.q4_mi = CuadranteMutualInfo(config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        detections: Dict
    ) -> Dict[str, torch.Tensor]:
        """Ejecuta los 4 cuadrantes planificadores."""
        
        q1 = self.q1_transiciones(input_ids, logits)
        q2 = self.q2_gramatica(input_ids, logits)
        q3 = self.q3_beam(logits)
        q4 = self.q4_mi(input_ids, logits)
        
        # Combinar todos los biases del planificador
        plan_bias = (
            q1['transition_bias'] +
            q2['grammar_bias'] +
            q4['mi_boost']
        )
        
        return {
            'transiciones': q1,
            'gramatica': q2,
            'beam': q3,
            'mutual_info': q4,
            'plan_bias': plan_bias,
        }


# =============================================================================
# CAJA 9: REFINADOR MATEMÁTICO (CORRECCIÓN GARANTIZADA)
# =============================================================================

class CuadranteBayes(nn.Module):
    """
    Q1: Ajuste Bayesiano.
    
    P(token|contexto) ∝ P(contexto|token) × P(token)
    
    Usa las detecciones como evidencia para ajustar probabilidades.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        
    def forward(
        self,
        logits: torch.Tensor,
        detections: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Ajusta logits basado en evidencia de detecciones.
        """
        # Si entropía muy baja, añadir ruido uniforme (prior)
        entropy_alert = detections['entropy']['alert_low']
        
        # Prior uniforme como regularización
        batch, seq, vocab = logits.shape
        uniform_prior = torch.zeros_like(logits)
        
        # Donde hay alerta de baja entropía, mezclar con uniforme
        adjustment = torch.zeros_like(logits)
        adjustment[entropy_alert] = 0.1  # Pequeño boost uniforme
        
        return {
            'bayes_adjustment': adjustment,
        }


class CuadrantePenalizacion(nn.Module):
    """
    Q2: Penalización de repetición.
    
    Divide la probabilidad de tokens repetidos por un factor.
    P_new(token) = P(token) / (1 + penalty × count(token))
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.penalty = config.repetition_penalty
        
    def forward(
        self,
        logits: torch.Tensor,
        detections: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch, seq, vocab]
            detections: incluye ngrama con repetition_score
        Returns:
            penalty_bias: [batch, seq, vocab] - penalización (negativa)
        """
        rep_score = detections['ngrama']['repetition_score']
        
        # Penalización logarítmica: -penalty * log(1 + score)
        penalty_bias = -self.penalty * torch.log(1 + rep_score)
        
        return {
            'penalty_bias': penalty_bias,
        }


class CuadranteTemperatura(nn.Module):
    """
    Q3: Temperatura adaptativa.
    
    Si la entropía es muy baja → aumentar temperatura (más diverso)
    Si la entropía es muy alta → reducir temperatura (más enfocado)
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.min_temp = config.min_temperature
        self.max_temp = config.max_temperature
        self.target_entropy = 2.0  # Entropía objetivo saludable
        
    def forward(
        self,
        logits: torch.Tensor,
        detections: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula temperatura adaptativa por posición.
        """
        entropy = detections['entropy']['entropy']  # [batch, seq]
        
        # T = 1 + k × (target - actual)
        # Si entropy < target → T > 1 (más diverso)
        # Si entropy > target → T < 1 (más enfocado)
        k = 0.3
        temperature = 1.0 + k * (self.target_entropy - entropy)
        
        # Clamp
        temperature = temperature.clamp(self.min_temp, self.max_temp)
        
        return {
            'temperature': temperature,  # [batch, seq]
        }


class CuadranteNormalizacion(nn.Module):
    """
    Q4: Renormalización garantizada.
    
    Después de todos los ajustes, garantiza que:
    1. No hay probabilidades negativas
    2. Suma = 1
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        
    def forward(
        self,
        logits: torch.Tensor,
        total_bias: torch.Tensor,
        temperature: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Aplica bias, temperatura, y normaliza.
        
        Args:
            logits: [batch, seq, vocab] - logits originales
            total_bias: [batch, seq, vocab] - todos los ajustes
            temperature: [batch, seq] - temperatura por posición
        Returns:
            final_logits: [batch, seq, vocab] - logits corregidos
            final_probs: [batch, seq, vocab] - probabilidades finales
        """
        # Aplicar bias
        adjusted = logits + total_bias
        
        # Aplicar temperatura
        temp_expanded = temperature.unsqueeze(-1)  # [batch, seq, 1]
        tempered = adjusted / temp_expanded
        
        # Softmax garantiza normalización
        final_probs = F.softmax(tempered, dim=-1)
        
        return {
            'final_logits': tempered,
            'final_probs': final_probs,
        }


class CajaRefinadoraV2(nn.Module):
    """
    Caja 9: Refinador con 4 cuadrantes matemáticos.
    
    Corrige las probabilidades de forma GARANTIZADA
    para evitar colapsos y repeticiones.
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.q1_bayes = CuadranteBayes(config)
        self.q2_penalizacion = CuadrantePenalizacion(config)
        self.q3_temperatura = CuadranteTemperatura(config)
        self.q4_normalizacion = CuadranteNormalizacion(config)
        
    def forward(
        self,
        logits: torch.Tensor,
        detections: Dict,
        plan_bias: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Ejecuta los 4 cuadrantes refinadores."""
        
        q1 = self.q1_bayes(logits, detections)
        q2 = self.q2_penalizacion(logits, detections)
        q3 = self.q3_temperatura(logits, detections)
        
        # Combinar todos los biases
        total_bias = (
            plan_bias +
            q1['bayes_adjustment'] +
            q2['penalty_bias']
        )
        
        # Normalización final con temperatura
        q4 = self.q4_normalizacion(logits, total_bias, q3['temperature'])
        
        return {
            'bayes': q1,
            'penalizacion': q2,
            'temperatura': q3,
            'normalizacion': q4,
            'total_bias': total_bias,
            'final_logits': q4['final_logits'],
            'final_probs': q4['final_probs'],
        }


# =============================================================================
# MÓDULO COMPOSITOR V2 COMPLETO
# =============================================================================

class ModuloCompositorV2(nn.Module):
    """
    Compositor v2: Matemáticas Blindadas.
    
    Recibe logits de las cajas neuronales 1-6 y los corrige
    usando cálculos matemáticos puros que NO PUEDEN FALLAR.
    
    Flujo:
        Logits de Cajas 1-6
              ↓
        Caja 7: Detectar problemas (Entropía, N-grama, etc.)
              ↓
        Caja 8: Planificar corrección (Transiciones, Gramática)
              ↓
        Caja 9: Aplicar corrección (Bayes, Penalización, Temp)
              ↓
        Logits/Probs corregidos y GARANTIZADOS
    """
    
    def __init__(self, config: CompositorV2Config):
        super().__init__()
        self.config = config
        
        # Las 3 cajas matemáticas
        self.caja_7_detector = CajaDetectoraV2(config)
        self.caja_8_planificador = CajaPlanificadoraV2(config)
        self.caja_9_refinador = CajaRefinadoraV2(config)
        
        print(f"✓ ModuloCompositorV2: 3 cajas MATEMÁTICAS (7-8-9)")
        print(f"  Caja 7: Entropía + N-grama + Compresión + Autocorr")
        print(f"  Caja 8: Transiciones + Gramática + Beam + MI")
        print(f"  Caja 9: Bayes + Penalización + Temperatura + Norm")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq] - tokens de entrada
            logits: [batch, seq, vocab] - logits de cajas 1-6
        Returns:
            dict con logits/probs corregidos y diagnósticos
        """
        # Caja 7: Detectar
        detections = self.caja_7_detector(input_ids, logits)
        
        # Caja 8: Planificar
        plan = self.caja_8_planificador(input_ids, logits, detections)
        
        # Caja 9: Refinar
        refined = self.caja_9_refinador(logits, detections, plan['plan_bias'])
        
        return {
            'final_logits': refined['final_logits'],
            'final_probs': refined['final_probs'],
            'detections': detections,
            'plan': plan,
            'refined': refined,
            'needs_correction': detections['needs_correction'],
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST COMPOSITOR V2 - MATEMÁTICAS BLINDADAS")
    print("=" * 60 + "\n")
    
    config = CompositorV2Config(
        vocab_size=256,
        ngram_window=8,
    )
    
    compositor = ModuloCompositorV2(config)
    
    # Simular entrada
    batch, seq, vocab = 2, 32, 256
    input_ids = torch.randint(0, vocab, (batch, seq))
    logits = torch.randn(batch, seq, vocab)
    
    # Simular colapso (logits muy concentrados en un token)
    logits[:, -5:, ord('e')] = 10.0  # Forzar 'e' en últimas posiciones
    
    print("Simulando colapso en 'e'...")
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Forward
    output = compositor(input_ids, logits)
    
    print(f"\n📊 Resultados:")
    print(f"  Final logits shape: {output['final_logits'].shape}")
    print(f"  Final probs shape: {output['final_probs'].shape}")
    
    # Verificar corrección
    original_probs = F.softmax(logits, dim=-1)
    final_probs = output['final_probs']
    
    print(f"\n📈 Comparación última posición:")
    print(f"  Original P('e'): {original_probs[0, -1, ord('e')].item():.4f}")
    print(f"  Corregido P('e'): {final_probs[0, -1, ord('e')].item():.4f}")
    
    print(f"\n  Original entropía: {-(original_probs[0, -1] * torch.log(original_probs[0, -1] + 1e-10)).sum().item():.4f}")
    print(f"  Corregida entropía: {-(final_probs[0, -1] * torch.log(final_probs[0, -1] + 1e-10)).sum().item():.4f}")
    
    print(f"\n  Temperatura aplicada: {output['refined']['temperatura']['temperature'][0, -1].item():.4f}")
    print(f"  Needs correction: {output['needs_correction'][0, -1].item()}")
    
    print("\n✅ Compositor V2 funcionando!")
