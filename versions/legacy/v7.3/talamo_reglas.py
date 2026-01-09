# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Tálamo con Detección Explícita y Reglas Claras

El Tálamo real es el "portero" del cerebro - dirige información a las áreas correctas.
Aquí implementamos DETECTORES EXPLÍCITOS que identifican el tipo de contenido:

- Matemáticas: números, operadores, ecuaciones
- Lenguaje: artículos, pronombres, estructuras gramaticales
- Lógica: condicionales, causalidad, deducciones
- Patrones: secuencias, repeticiones
- Contexto: referencias, relaciones temporales
- Creatividad: metáforas, expresiones inusuales

Cada módulo tiene su "LLAVE" - patrones que lo activan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import re


class DetectorDeContenido(nn.Module):
    """
    Detecta el tipo de contenido usando REGLAS + APRENDIZAJE.
    
    Combina:
    1. Detectores basados en reglas (patrones conocidos)
    2. Detectores aprendidos (embeddings)
    """
    
    # === LLAVES: Patrones que identifican cada dominio ===
    
    # Tokens que indican MATEMÁTICAS (IDs comunes en BPE)
    TOKENS_MATEMATICAS = {
        # Números
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '10', '100', '1000',
        # Operadores
        '+', '-', '*', '/', '=', '<', '>', '%',
        # Palabras matemáticas
        'plus', 'minus', 'times', 'divided', 'equals', 'sum', 'total',
        'percent', 'number', 'calculate', 'equation', 'formula',
        'million', 'billion', 'thousand', 'hundred',
    }
    
    # Tokens que indican LÓGICA
    TOKENS_LOGICA = {
        'if', 'then', 'else', 'therefore', 'because', 'since', 'thus',
        'however', 'although', 'unless', 'either', 'neither', 'both',
        'implies', 'conclude', 'reason', 'logic', 'proof', 'assume',
        'true', 'false', 'valid', 'invalid', 'premise', 'argument',
    }
    
    # Tokens que indican LENGUAJE/GRAMÁTICA
    TOKENS_LENGUAJE = {
        # Artículos y determinantes
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        # Pronombres
        'he', 'she', 'it', 'they', 'we', 'you', 'i',
        # Verbos auxiliares
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        # Preposiciones comunes
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    }
    
    # Tokens que indican CONTEXTO/NARRATIVA
    TOKENS_CONTEXTO = {
        # Temporales
        'when', 'while', 'before', 'after', 'during', 'until', 'since',
        'now', 'then', 'later', 'earlier', 'soon', 'always', 'never',
        'today', 'yesterday', 'tomorrow', 'year', 'month', 'day',
        # Referencias
        'here', 'there', 'where', 'which', 'who', 'whom', 'whose',
        # Narrativa
        'said', 'told', 'asked', 'replied', 'story', 'once',
    }
    
    # Tokens que indican CREATIVIDAD
    TOKENS_CREATIVIDAD = {
        'like', 'as', 'imagine', 'dream', 'wonder', 'perhaps', 'maybe',
        'beautiful', 'strange', 'magical', 'mysterious', 'amazing',
        'create', 'invent', 'design', 'art', 'music', 'poetry',
        'fantasy', 'fiction', 'novel', 'tale', 'legend', 'myth',
    }
    
    # Tokens que indican PATRONES/SECUENCIAS
    TOKENS_PATRONES = {
        'first', 'second', 'third', 'next', 'last', 'final',
        'step', 'stage', 'phase', 'level', 'order', 'sequence',
        'repeat', 'again', 'pattern', 'cycle', 'series', 'list',
        '1.', '2.', '3.', 'a)', 'b)', 'c)',
    }
    
    def __init__(self, dim: int, vocab_size: int = 8000):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_dominios = 6
        
        # === DETECTORES BASADOS EN REGLAS ===
        # Matriz de "llaves": vocab_size x n_dominios
        # Cada token tiene un peso por dominio basado en las reglas
        self.llaves_base = nn.Parameter(
            torch.zeros(vocab_size, self.n_dominios),
            requires_grad=False  # Las reglas no cambian
        )
        
        # === DETECTORES APRENDIDOS ===
        # Complementan las reglas con patrones aprendidos
        self.detector_aprendido = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_dominios),
        )
        
        # Peso entre reglas y aprendido
        self.mix_reglas_aprendido = nn.Parameter(torch.tensor(0.7))  # 70% reglas, 30% aprendido
        
        # Nombres de dominios
        self.dominios = ['matematicas', 'logica', 'lenguaje', 'contexto', 'creatividad', 'patrones']
        
    def inicializar_llaves(self, tokenizer):
        """
        Inicializa las llaves basándose en el vocabulario del tokenizer.
        Debe llamarse después de cargar el tokenizer.
        """
        # Mapeo de conjuntos de tokens a índices de dominio
        token_sets = [
            self.TOKENS_MATEMATICAS,  # 0
            self.TOKENS_LOGICA,       # 1
            self.TOKENS_LENGUAJE,     # 2
            self.TOKENS_CONTEXTO,     # 3
            self.TOKENS_CREATIVIDAD,  # 4
            self.TOKENS_PATRONES,     # 5
        ]
        
        # Para cada token en el vocabulario
        for token_id in range(self.vocab_size):
            try:
                token_str = tokenizer.id_to_piece(token_id).lower().replace('▁', '')
            except:
                continue
                
            # Verificar en qué dominios cae este token
            for dominio_idx, token_set in enumerate(token_sets):
                if token_str in token_set:
                    self.llaves_base.data[token_id, dominio_idx] = 1.0
                # También verificar si el token CONTIENE alguna palabra clave
                for keyword in token_set:
                    if len(keyword) > 2 and keyword in token_str:
                        self.llaves_base.data[token_id, dominio_idx] += 0.5
                        break
        
        # Normalizar
        row_sums = self.llaves_base.data.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1.0)  # Evitar división por 0
        self.llaves_base.data = self.llaves_base.data / row_sums
        
        print(f"✅ Llaves inicializadas para {self.vocab_size} tokens")
        
    def forward(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Detecta el dominio del input.
        
        Args:
            input_ids: (batch, seq) IDs de tokens
            embeddings: (batch, seq, dim) Embeddings del input
            
        Returns:
            scores: (batch, n_dominios) Puntuación por dominio
        """
        batch, seq = input_ids.shape
        
        # === 1. DETECCIÓN POR REGLAS ===
        # Buscar las llaves para cada token
        llaves = self.llaves_base[input_ids]  # (batch, seq, n_dominios)
        
        # Promediar sobre la secuencia
        scores_reglas = llaves.mean(dim=1)  # (batch, n_dominios)
        
        # === 2. DETECCIÓN APRENDIDA ===
        # Usar el embedding promedio
        emb_mean = embeddings.mean(dim=1)  # (batch, dim)
        scores_aprendido = self.detector_aprendido(emb_mean)  # (batch, n_dominios)
        
        # === 3. COMBINAR ===
        mix = torch.sigmoid(self.mix_reglas_aprendido)
        scores = mix * scores_reglas + (1 - mix) * scores_aprendido
        
        return scores


class TalamoConReglas(nn.Module):
    """
    Tálamo que usa REGLAS CLARAS para asignar liderazgo.
    
    Principios:
    1. El Tálamo es el ORQUESTADOR - decide quién lidera
    2. Cada módulo tiene su LLAVE - patrones que lo activan
    3. La decisión es CLARA - no hay ambigüedad
    """
    
    def __init__(
        self, 
        dim: int,
        vocab_size: int = 8000,
        n_modulos: int = 6,
        umbral_liderazgo: float = 0.3,  # Mínimo para ser líder
        actividad_basal: float = 0.15,   # Actividad mínima de cada módulo
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        self.umbral_liderazgo = umbral_liderazgo
        self.actividad_basal = actividad_basal
        
        # Detector de contenido con reglas
        self.detector = DetectorDeContenido(dim, vocab_size)
        
        # Nombres de módulos (en orden)
        self.nombres = ['matematicas', 'logica', 'lenguaje', 'contexto', 'creatividad', 'patrones']
        
        # Modulador de intensidad (contextual)
        self.modulador = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Estado
        self.ultimo_lider = None
        self.ultimos_scores = None
        
    def inicializar(self, tokenizer):
        """Inicializa los detectores con el tokenizer."""
        self.detector.inicializar_llaves(tokenizer)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Determina quién lidera y cómo se modulan los módulos.
        
        Returns:
            liderazgo: (batch, n_modulos) - Probabilidad de liderazgo
            modulacion: (batch, n_modulos) - Nivel de activación de cada módulo
            stats: Dict con información de debug
        """
        batch = input_ids.shape[0]
        device = input_ids.device
        
        # === 1. DETECTAR TIPO DE CONTENIDO ===
        scores = self.detector(input_ids, embeddings)  # (batch, n_modulos)
        
        # === 2. DETERMINAR LÍDER ===
        # Softmax con temperatura baja = decisión más clara
        liderazgo = F.softmax(scores * 3.0, dim=-1)  # Temperatura 0.33
        
        # Encontrar el líder (máximo score)
        lider_idx = scores.argmax(dim=-1)  # (batch,)
        lider_score = scores.gather(1, lider_idx.unsqueeze(1)).squeeze(1)  # (batch,)
        
        # === 3. CALCULAR MODULACIÓN ===
        # El líder tiene activación alta, los otros tienen actividad basal + algo proporcional
        modulacion = torch.ones(batch, self.n_modulos, device=device) * self.actividad_basal
        
        # El líder tiene activación completa
        modulacion.scatter_(1, lider_idx.unsqueeze(1), 1.0)
        
        # Los demás se activan proporcionalmente a su score
        for i in range(self.n_modulos):
            mask = (lider_idx != i)  # No es el líder
            if mask.any():
                # Actividad basal + proporción del score
                actividad = self.actividad_basal + scores[:, i] * (1 - self.actividad_basal)
                modulacion[:, i] = torch.where(mask, actividad, modulacion[:, i])
        
        # === 4. STATS ===
        # Nombre del líder más común en el batch
        lider_comun = lider_idx.mode().values.item()
        nombre_lider = self.nombres[lider_comun]
        
        stats = {
            'lider': nombre_lider,
            'lider_idx': lider_comun,
            'lider_score': lider_score.mean().item(),
        }
        
        # Scores por módulo
        for i, nombre in enumerate(self.nombres):
            stats[f'score_{nombre}'] = scores[:, i].mean().item()
            stats[f'mod_{nombre}'] = modulacion[:, i].mean().item()
        
        # Guardar estado
        self.ultimo_lider = nombre_lider
        self.ultimos_scores = scores.detach()
        
        return liderazgo, modulacion, stats
    
    def get_matriz_acoplamiento(self) -> torch.Tensor:
        """Retorna cómo los módulos se acoplan según el líder actual."""
        # Matriz de acoplamiento natural
        # Filas = líder, Columnas = seguidor
        # Valor = qué tan fuerte se acopla el seguidor al líder
        acoplamiento = torch.tensor([
            # mat   log   len   ctx   cre   pat
            [1.0,  0.7,  0.3,  0.2,  0.1,  0.8],  # matemáticas lidera
            [0.6,  1.0,  0.5,  0.4,  0.2,  0.6],  # lógica lidera
            [0.2,  0.4,  1.0,  0.7,  0.6,  0.3],  # lenguaje lidera
            [0.3,  0.5,  0.8,  1.0,  0.5,  0.4],  # contexto lidera
            [0.2,  0.3,  0.7,  0.6,  1.0,  0.4],  # creatividad lidera
            [0.7,  0.6,  0.3,  0.4,  0.3,  1.0],  # patrones lidera
        ])
        return acoplamiento
    
    def reset_estado(self):
        """Resetea el estado interno."""
        self.ultimo_lider = None
        self.ultimos_scores = None
