# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Cajas de procesamiento: CajaDatos y CajaCalculos.
"""

import torch
import torch.nn as nn
from typing import Optional

from llarri_o1.modules.niveles import CuadranteProgresivo
from llarri_o1.modules.relaciones import RelacionesCuadrantes, AutoCalculos


class CajaDatos(nn.Module):
    """
    Caja de DATOS - procesa entrada dividida en 4 cuadrantes.
    
    Flujo:
        1. Proyectar entrada a hidden_dim (si es necesario)
        2. Dividir en 4 cuadrantes
        3. Procesar cada cuadrante con pipeline fractal
        4. Relacionar cuadrantes entre sí
        5. Fusionar y conexión residual
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cuadrante: CuadranteProgresivo
    ):
        """
        Args:
            input_dim: Dimensión de entrada (784 para MNIST)
            hidden_dim: Dimensión oculta
            cuadrante: Módulo CuadranteProgresivo compartido
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        quad_dim = hidden_dim // 4
        
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.cuadrante = cuadrante
        self.relaciones = RelacionesCuadrantes(quad_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa entrada a través de 4 cuadrantes.
        
        Args:
            x: Tensor de shape (batch, input_dim) o (batch, hidden_dim)
            
        Returns:
            Tensor de shape (batch, hidden_dim)
        """
        # Proyectar si viene de input original
        if x.shape[-1] == self.input_dim:
            x = self.proj_in(x)
        
        quad_dim = self.hidden_dim // 4
        
        # Dividir en 4 cuadrantes
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        # Procesar SECUENCIALMENTE cada cuadrante
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relacionar
        a, b, c, d = self.relaciones(a, b, c, d)
        
        # Fusionar con residual
        out = torch.cat([a, b, c, d], dim=-1)
        return self.fusion(out) + x


class CajaCalculos(nn.Module):
    """
    Caja de CÁLCULOS - opera sobre datos + otros cálculos.
    
    INCLUYE: Auto-cálculos internos donde los valores intermedios
    también se calculan entre sí.
    
    Flujo:
        1. Combinar dos entradas de datos
        2. Integrar cálculos previos (si existen)
        3. Dividir en 4 cuadrantes
        4. Procesar cada cuadrante
        5. Relacionar cuadrantes
        6. Auto-cálculos internos
        7. Conexión residual
    """
    
    def __init__(self, hidden_dim: int, cuadrante: CuadranteProgresivo):
        """
        Args:
            hidden_dim: Dimensión oculta
            cuadrante: Módulo CuadranteProgresivo compartido
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        quad_dim = hidden_dim // 4
        
        self.op_combinar = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cuadrante = cuadrante
        self.relaciones = RelacionesCuadrantes(quad_dim)
        self.auto_calculos = AutoCalculos(quad_dim)
        self.integrar = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        datos1: torch.Tensor,
        datos2: torch.Tensor,
        otros_calc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Procesa combinando datos y cálculos previos.
        
        Args:
            datos1: Primera entrada de datos (batch, hidden_dim)
            datos2: Segunda entrada de datos (batch, hidden_dim)
            otros_calc: Cálculos previos opcionales (batch, hidden_dim)
            
        Returns:
            Tensor de shape (batch, hidden_dim)
        """
        # Combinar datos
        x = self.op_combinar(torch.cat([datos1, datos2], dim=-1))
        
        # Integrar cálculos previos si existen
        if otros_calc is not None:
            x = self.norm(x + self.integrar(torch.cat([x, otros_calc], dim=-1)) * 0.5)
        
        quad_dim = self.hidden_dim // 4
        
        # Dividir en 4 cuadrantes
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        # Procesar cuadrantes
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relaciones entre cuadrantes
        a, b, c, d = self.relaciones(a, b, c, d)
        
        # AUTO-CÁLCULOS: los valores intermedios se calculan entre sí
        a, b, c, d = self.auto_calculos(a, b, c, d)
        
        return torch.cat([a, b, c, d], dim=-1) + x
