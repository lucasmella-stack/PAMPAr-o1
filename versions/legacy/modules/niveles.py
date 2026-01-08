# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Procesamiento por niveles fractales.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from llarri_o1.modules.cache import CacheBinario


class ProcesoNivel(nn.Module):
    """
    Procesa UN nivel específico del pipeline fractal.
    
    Arquitectura:
        Linear(dim → dim*2) → LayerNorm → GELU → Dropout → Linear(dim*2 → dim)
        + Conexión residual
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Args:
            dim: Dimensión del nivel
            dropout: Probabilidad de dropout
        """
        super().__init__()
        self.proceso = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Procesa con conexión residual."""
        return self.norm(x + self.proceso(x))


class CuadranteProgresivo(nn.Module):
    """
    Cuadrante que procesa SECUENCIALMENTE desde nivel 2 hasta 256.
    
    Flujo:
        entrada(quad_dim) → comprimir(2) → nivel_2 → nivel_4 → ... → nivel_256 → expandir(quad_dim)
    
    Cada nivel usa ProcesoNivel con su dimensión específica.
    El nivel 2 (binario) utiliza CacheBinario para lookup instantáneo.
    """
    
    def __init__(
        self,
        dim_cuadrante: int,
        niveles: List[int],
        dropout: float = 0.1,
        cache_binario: Optional[CacheBinario] = None
    ):
        """
        Args:
            dim_cuadrante: Dimensión del cuadrante (hidden_dim // 4)
            niveles: Lista de dimensiones por nivel [2, 4, 8, 16, 32, 64, 128, 256]
            dropout: Probabilidad de dropout
            cache_binario: Cache opcional para nivel 2
        """
        super().__init__()
        self.dim = dim_cuadrante
        self.niveles = niveles
        self.cache = cache_binario
        self.active_levels: Optional[List[int]] = None  # None = todos
        
        # Comprimir entrada al nivel binario
        self.comprimir = nn.Linear(dim_cuadrante, 2)
        
        # Fusión de cache binario (2 valores + 7 operaciones = 9)
        self.fusion_cache = nn.Linear(2 + 7, 2)
        
        # Procesos para cada nivel
        self.procesos = nn.ModuleDict()
        for nivel in self.niveles:
            self.procesos[str(nivel)] = ProcesoNivel(nivel, dropout)
        
        # Transiciones entre niveles (subir: 2→4→8→...)
        self.subir = nn.ModuleDict()
        for i in range(len(self.niveles) - 1):
            din, dout = self.niveles[i], self.niveles[i + 1]
            self.subir[f'{din}_{dout}'] = nn.Linear(din, dout)
        
        # Expandir al tamaño original
        self.expandir = nn.Linear(self.niveles[-1], dim_cuadrante)
        self.norm_final = nn.LayerNorm(dim_cuadrante)
    
    def set_active_levels(self, levels: Optional[List[int]]):
        """
        Establece qué niveles están activos para el forward.
        
        Args:
            levels: Lista de niveles activos (ej: [2, 4, 8]) o None para todos
        """
        self.active_levels = levels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa secuencialmente: 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256
        Solo procesa los niveles activos (o todos si active_levels es None).
        
        Args:
            x: Tensor de shape (batch, dim_cuadrante)
            
        Returns:
            Tensor de shape (batch, dim_cuadrante)
        """
        residual = x
        
        # Determinar qué niveles procesar
        if self.active_levels is None:
            niveles_activos = self.niveles
        else:
            niveles_activos = [n for n in self.niveles if n in self.active_levels]
            if not niveles_activos:
                niveles_activos = self.niveles[:1]  # Al menos nivel 2
        
        # 1. Comprimir al nivel binario
        h = self.comprimir(x)  # (batch, 2)
        
        # 2. Usar cache binario (lookup instantáneo)
        if self.cache is not None:
            cache_out = self.cache.lookup(h)  # (batch, 7)
            h = self.fusion_cache(torch.cat([h, cache_out], dim=-1))
        
        # 3. Procesar nivel 2 (siempre se procesa)
        h = self.procesos['2'](h)
        
        # 4. Subir SECUENCIALMENTE solo por niveles activos
        ultimo_nivel = 2
        for i in range(len(self.niveles) - 1):
            din, dout = self.niveles[i], self.niveles[i + 1]
            
            # Solo procesar si el nivel de destino está activo
            if dout in niveles_activos:
                # Si necesitamos saltar niveles, hacerlo con las transiciones
                if din != ultimo_nivel:
                    # Transición directa desde ultimo_nivel a din
                    idx_from = self.niveles.index(ultimo_nivel)
                    idx_to = self.niveles.index(din)
                    for j in range(idx_from, idx_to):
                        d1, d2 = self.niveles[j], self.niveles[j + 1]
                        h = self.subir[f'{d1}_{d2}'](h)
                
                h = self.subir[f'{din}_{dout}'](h)
                h = self.procesos[str(dout)](h)
                ultimo_nivel = dout
        
        # 5. Si no llegamos al nivel máximo, expandir desde donde estamos
        max_nivel = self.niveles[-1]
        if ultimo_nivel != max_nivel:
            # Subir hasta el máximo para poder expandir correctamente
            idx_from = self.niveles.index(ultimo_nivel)
            for j in range(idx_from, len(self.niveles) - 1):
                d1, d2 = self.niveles[j], self.niveles[j + 1]
                h = self.subir[f'{d1}_{d2}'](h)
        
        # 6. Expandir y residual
        h = self.expandir(h)
        return self.norm_final(h + residual)
