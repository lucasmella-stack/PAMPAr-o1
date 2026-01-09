# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Módulo de Lenguaje - Especializado en procesamiento lingüístico

Dominio:
- Gramática y sintaxis
- Artículos, preposiciones, conectores
- Fluidez del texto
- Coherencia semántica
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from ..neurona import Neurona


class NeuronaLenguaje(Neurona):
    """
    Neurona especializada en procesamiento del lenguaje.
    
    Características:
    - Mayor atención a patrones secuenciales (gramática)
    - Sensible a tokens de alta frecuencia (artículos, etc.)
    - Promueve fluidez y coherencia
    """
    
    NOMBRE = 'lenguaje'
    
    # Tokens que activan fuertemente este módulo
    LLAVES = {
        'articulos': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'],
        'preposiciones': ['de', 'en', 'a', 'por', 'para', 'con', 'sin', 'sobre'],
        'conectores': ['y', 'o', 'pero', 'aunque', 'porque', 'que', 'como'],
        'verbos_auxiliares': ['es', 'son', 'era', 'fue', 'ser', 'estar', 'haber'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Capa adicional para detectar patrones gramaticales
        self.detector_gramatical = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
        )
        
        # Bias hacia tokens frecuentes (se aprende)
        self.bias_frecuencia = nn.Parameter(torch.zeros(1, 1, dim))
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Determina qué tan relevante es este módulo para cada posición.
        
        Para lenguaje: alta relevancia en tokens de función (artículos, etc.)
        """
        batch, seq = token_ids.shape
        
        # Base: todos los tokens tienen algo de relevancia lingüística
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.3
        
        # Aumentar para tokens específicos (se hará con tokenizer registrado)
        # Por ahora, usar heurística basada en embeddings
        # Tokens de función tienden a tener embeddings más "centrales"
        norma_embed = embeddings.norm(dim=-1)
        es_frecuente = norma_embed < norma_embed.mean()
        relevancia = relevancia + es_frecuente.float() * 0.4
        
        return relevancia.clamp(0, 1)
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesamiento especializado en lenguaje.
        
        1. Atención base (hereda de Neurona)
        2. Detección de patrones gramaticales
        3. Bias de frecuencia
        """
        # Atención base
        atendido = self.attention(self.norm1(x), mask)
        x = x + atendido
        
        # Detección gramatical
        gramatical = self.detector_gramatical(x)
        x = x + 0.3 * gramatical  # Contribución moderada
        
        # FFN con bias de frecuencia
        x = x + self.ffn(self.norm2(x)) + self.bias_frecuencia
        
        return x


class NeuronaLogica(Neurona):
    """
    Neurona especializada en razonamiento lógico.
    
    Características:
    - Detecta conectores lógicos (si, entonces, porque)
    - Prepara información para sistema de axiomas
    - Evalúa consistencia
    """
    
    NOMBRE = 'logica'
    
    LLAVES = {
        'condicionales': ['si', 'entonces', 'cuando', 'mientras'],
        'causales': ['porque', 'por lo tanto', 'ya que', 'debido a'],
        'adversativos': ['pero', 'sin embargo', 'aunque', 'no obstante'],
        'valores': ['verdadero', 'falso', 'correcto', 'incorrecto'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Proyección para análisis de premisas
        self.analisis_premisa = nn.Linear(dim, dim)
        
        # Detector de estructura lógica
        self.detector_estructura = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        # Lógica tiene relevancia base moderada
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.2
        return relevancia
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesamiento lógico: busca estructuras de razonamiento.
        """
        # Atención base
        atendido = self.attention(self.norm1(x), mask)
        x_atendido = x + atendido
        
        # Análisis de premisas (qué tokens son premisas vs conclusiones)
        premisas = self.analisis_premisa(x_atendido)
        
        # Combinar entrada con premisas para detectar estructura
        combinado = torch.cat([x_atendido, premisas], dim=-1)
        estructura = self.detector_estructura(combinado)
        
        # FFN final
        x = x_atendido + 0.5 * estructura
        x = x + self.ffn(self.norm2(x))
        
        return x


class NeuronaMatematicas(Neurona):
    """
    Neurona especializada en procesamiento matemático/numérico.
    
    Características:
    - Alta sensibilidad a dígitos y operadores
    - Procesa secuencias numéricas
    - Detecta operaciones
    """
    
    NOMBRE = 'matematicas'
    
    LLAVES = {
        'digitos': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'operadores': ['+', '-', '*', '/', '=', '<', '>', '%'],
        'terminos': ['suma', 'resta', 'multiplica', 'divide', 'igual'],
        'magnitudes': ['número', 'cantidad', 'valor', 'resultado', 'total'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Codificador posicional para secuencias numéricas
        self.pos_numerica = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        
        # Detector de operaciones
        self.detector_operacion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        # Matemáticas tiene relevancia base baja (muy específico)
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.1
        return relevancia
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesamiento matemático: énfasis en orden y operaciones.
        """
        batch, seq, _ = x.shape
        
        # Añadir codificación posicional numérica
        if seq <= self.pos_numerica.shape[1]:
            x = x + self.pos_numerica[:, :seq, :]
        
        # Atención
        atendido = self.attention(self.norm1(x), mask)
        x = x + atendido
        
        # Detectar operaciones
        operacion = self.detector_operacion(x)
        x = x + 0.4 * operacion
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class NeuronaPatrones(Neurona):
    """
    Neurona especializada en detección de patrones y secuencias.
    
    Características:
    - Detecta repeticiones
    - Identifica estructuras recurrentes
    - Agrupa elementos similares
    """
    
    NOMBRE = 'patrones'
    
    LLAVES = {
        'secuencia': ['secuencia', 'serie', 'patrón', 'repetir', 'siguiente'],
        'orden': ['primero', 'segundo', 'último', 'anterior', 'posterior'],
        'cuantificadores': ['todos', 'ninguno', 'algunos', 'cada', 'siempre'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Detector de similitud entre posiciones
        self.detector_similitud = nn.Bilinear(dim, dim, 1)
        
        # Agregador de patrones
        self.agregador = nn.Linear(dim, dim)
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.15
        return relevancia
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Detecta patrones comparando posiciones.
        """
        # Atención base
        atendido = self.attention(self.norm1(x), mask)
        x = x + atendido
        
        # Agregar información de patrones
        patrones = self.agregador(x)
        x = x + 0.3 * patrones
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class NeuronaContexto(Neurona):
    """
    Neurona especializada en mantener y usar contexto.
    
    Características:
    - Resuelve referencias (él, ella, esto)
    - Mantiene coherencia temporal
    - Conecta con memoria (hipocampo)
    """
    
    NOMBRE = 'contexto'
    
    LLAVES = {
        'pronombres': ['él', 'ella', 'ellos', 'su', 'sus', 'este', 'ese'],
        'temporales': ['antes', 'después', 'ahora', 'luego', 'mientras'],
        'espaciales': ['aquí', 'allí', 'donde', 'cerca', 'lejos'],
        'referencias': ['anterior', 'previo', 'mencionado', 'dicho'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Estado contextual acumulado
        self.estado_ctx = nn.GRU(dim, dim, batch_first=True)
        
        # Proyección para contexto
        self.proyeccion_ctx = nn.Linear(dim * 2, dim)
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        # Contexto siempre tiene relevancia moderada
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.25
        return relevancia
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesa manteniendo estado contextual.
        """
        # Atención base
        atendido = self.attention(self.norm1(x), mask)
        x_atendido = x + atendido
        
        # Acumular contexto con GRU
        contexto, _ = self.estado_ctx(x_atendido)
        
        # Combinar entrada con contexto
        combinado = torch.cat([x_atendido, contexto], dim=-1)
        x = self.proyeccion_ctx(combinado)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class NeuronaCreatividad(Neurona):
    """
    Neurona especializada en generación creativa.
    
    Características:
    - Introduce variabilidad controlada
    - Combina conceptos distantes
    - Genera alternativas
    """
    
    NOMBRE = 'creatividad'
    
    LLAVES = {
        'creacion': ['imagina', 'inventa', 'crea', 'diseña', 'nuevo'],
        'hipoteticos': ['qué pasaría', 'y si', 'podría', 'sería'],
        'metaforas': ['como', 'parece', 'similar', 'diferente'],
    }
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=dim, n_heads=n_heads, dropout=dropout)
        
        # Perturbación creativa (ruido controlado)
        self.escala_ruido = nn.Parameter(torch.tensor(0.1))
        
        # Combinador de conceptos
        self.combinador = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def es_mi_dominio(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        relevancia = torch.ones(batch, seq, device=token_ids.device) * 0.15
        return relevancia
    
    def procesar(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesamiento creativo con variabilidad controlada.
        """
        # Atención base
        atendido = self.attention(self.norm1(x), mask)
        x = x + atendido
        
        # Combinar conceptos
        combinado = self.combinador(x)
        
        # Añadir ruido creativo (solo en training)
        if self.training:
            ruido = torch.randn_like(combinado) * self.escala_ruido
            combinado = combinado + ruido
        
        x = x + 0.4 * combinado
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
