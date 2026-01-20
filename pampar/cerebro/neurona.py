# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Neurona Base - Unidad fundamental de procesamiento

Cada neurona:
- Tiene un dominio de especialización
- Se conecta con otras via sinapsis
- Puede activarse o inhibirse
- Procesa información en su dominio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod


class Neurona(nn.Module, ABC):
    """
    Clase base abstracta para todas las neuronas/módulos.
    
    Cada neurona tiene:
    - Dominio: su especialización (lenguaje, lógica, etc.)
    - Umbral: nivel mínimo de activación
    - Conexiones: sinapsis a otras neuronas
    """
    
    dominio: str = "base"
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Procesamiento interno
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Estado de activación
        self.activacion = 0.0
        self.umbral = 0.15  # Actividad basal mínima
        
        # Receptor de señales sinápticas
        self.receptor = nn.Linear(dim, dim)
        
    def procesar(self, x: torch.Tensor) -> torch.Tensor:
        """Procesamiento interno de la neurona."""
        # Atención
        residual = x
        x_norm = self.norm1(x)
        x_att, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + x_att
        
        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_ff = self.ffn(x_norm)
        x = residual + x_ff
        
        return x
    
    def recibir_senal(self, senal: torch.Tensor, peso: float = 1.0) -> torch.Tensor:
        """Recibe señal de otra neurona via sinapsis."""
        return self.receptor(senal) * peso
    
    @abstractmethod
    def es_mi_dominio(self, tokens: List[str]) -> float:
        """
        Determina si el input pertenece al dominio de esta neurona.
        Retorna un score de 0 a 1.
        """
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        senal_externa: Optional[torch.Tensor] = None,
        intensidad: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass de la neurona.
        
        Args:
            x: Input (batch, seq, dim)
            senal_externa: Señal de otras neuronas
            intensidad: Nivel de activación (del Tálamo)
            
        Returns:
            output: Salida procesada
            activacion: Nivel de activación actual
        """
        # Aplicar intensidad (modulación del Tálamo)
        if intensidad < self.umbral:
            intensidad = self.umbral  # Nunca se apaga completamente
        
        # Procesar
        output = self.procesar(x)
        
        # Si hay señal externa, integrarla
        if senal_externa is not None:
            senal_procesada = self.recibir_senal(senal_externa)
            output = output + senal_procesada * 0.3  # 30% de influencia externa
        
        # Escalar por intensidad
        output = output * intensidad
        
        # Guardar nivel de activación
        self.activacion = intensidad
        
        return output, self.activacion
