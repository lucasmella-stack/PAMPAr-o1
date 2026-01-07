# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Relaciones entre cuadrantes y auto-cálculos internos.
"""

import torch
import torch.nn as nn


class RelacionesCuadrantes(nn.Module):
    """
    Conecta 4 cuadrantes entre sí mediante relaciones cruzadas.
    
    Relaciones:
        - A↔B, C↔D (horizontales)
        - A↔C, B↔D (verticales)
    
    Cada cuadrante recibe información de sus vecinos.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimensión de cada cuadrante
        """
        super().__init__()
        self.rel = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor
    ) -> tuple:
        """
        Aplica relaciones cruzadas entre los 4 cuadrantes.
        
        Args:
            a, b, c, d: Tensores de shape (batch, dim)
            
        Returns:
            Tupla (a', b', c', d') con los cuadrantes actualizados
        """
        # Relaciones cruzadas
        ab = self.rel(torch.cat([a, b], dim=-1))
        cd = self.rel(torch.cat([c, d], dim=-1))
        ac = self.rel(torch.cat([a, c], dim=-1))
        bd = self.rel(torch.cat([b, d], dim=-1))
        
        # Actualizar cada cuadrante con sus relaciones
        a = self.norm(a + ab + ac)
        b = self.norm(b + ab + bd)
        c = self.norm(c + cd + ac)
        d = self.norm(d + cd + bd)
        
        return a, b, c, d


class AutoCalculos(nn.Module):
    """
    Auto-cálculos internos: los valores intermedios se calculan entre sí.
    
    Si tengo [v1, v2, v3, v4], calculo:
        - v1 ⊗ v2, v2 ⊗ v3, v3 ⊗ v4 (adyacentes)
        - v1 ⊗ v3, v2 ⊗ v4 (cruzados)
        - v1 ⊗ v4 (diagonales)
    
    Total: 6 operaciones que se fusionan y redistribuyen.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimensión de cada valor
        """
        super().__init__()
        
        # 6 operaciones entre pares
        self.op_adyacente_1 = nn.Linear(dim * 2, dim)  # v1 ⊗ v2
        self.op_adyacente_2 = nn.Linear(dim * 2, dim)  # v2 ⊗ v3
        self.op_adyacente_3 = nn.Linear(dim * 2, dim)  # v3 ⊗ v4
        self.op_cruzado_1 = nn.Linear(dim * 2, dim)    # v1 ⊗ v3
        self.op_cruzado_2 = nn.Linear(dim * 2, dim)    # v2 ⊗ v4
        self.op_diagonal = nn.Linear(dim * 2, dim)     # v1 ⊗ v4
        
        # Fusión de todos los cálculos
        self.fusion = nn.Sequential(
            nn.Linear(dim * 6, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor
    ) -> tuple:
        """
        Calcula todas las combinaciones entre valores internos.
        
        Args:
            a, b, c, d: Tensores de shape (batch, dim)
            
        Returns:
            Tupla (a', b', c', d') con los valores actualizados
        """
        # Cálculos adyacentes
        ab = self.op_adyacente_1(torch.cat([a, b], dim=-1))
        bc = self.op_adyacente_2(torch.cat([b, c], dim=-1))
        cd = self.op_adyacente_3(torch.cat([c, d], dim=-1))
        
        # Cálculos cruzados
        ac = self.op_cruzado_1(torch.cat([a, c], dim=-1))
        bd = self.op_cruzado_2(torch.cat([b, d], dim=-1))
        
        # Cálculo diagonal
        ad = self.op_diagonal(torch.cat([a, d], dim=-1))
        
        # Fusionar todos los cálculos
        todos = torch.cat([ab, bc, cd, ac, bd, ad], dim=-1)
        fusionado = self.fusion(todos)
        
        # Distribuir de vuelta a los 4 valores con residuales
        dim = a.shape[-1]
        a_new = self.norm(a + fusionado[..., :dim] * 0.5)
        b_new = self.norm(b + fusionado[..., dim:dim*2] * 0.5)
        c_new = self.norm(c + fusionado[..., dim*2:dim*3] * 0.5)
        d_new = self.norm(d + fusionado[..., dim*3:] * 0.5)
        
        return a_new, b_new, c_new, d_new
