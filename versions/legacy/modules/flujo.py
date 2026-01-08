# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Sistema de flujo entre cajas: llaves de conexión y flujo completo.
"""

import torch
import torch.nn as nn
from typing import List


class LlaveConexion(nn.Module):
    """
    Conecta cajas entre sí (unidireccional).
    
    Transfiere información de origen a destino con conexión residual.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimensión de las cajas
        """
        super().__init__()
        self.llave = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, origen: torch.Tensor, destino: torch.Tensor) -> torch.Tensor:
        """
        Transfiere información de origen a destino.
        
        Args:
            origen: Tensor fuente (batch, dim)
            destino: Tensor destino (batch, dim)
            
        Returns:
            Destino actualizado con información del origen
        """
        return self.norm(destino + self.llave(origen) * 0.5)


class LlaveBidireccional(nn.Module):
    """
    Llave BIDIRECCIONAL - intercambio mutuo entre dos cajas.
    
    A y B se comunican en ambas direcciones simultáneamente:
        A' = A + info_de_B
        B' = B + info_de_A
    
    Economiza memoria: usa UN solo set de pesos para ambas direcciones
    con una transformación simétrica y gate adaptativo.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimensión de las cajas
        """
        super().__init__()
        # Pesos compartidos para ambas direcciones (economía de memoria)
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, 2)  # Gate para balancear mezcla
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple:
        """
        Intercambio bidireccional entre A y B.
        
        Args:
            a: Primera caja (batch, dim)
            b: Segunda caja (batch, dim)
            
        Returns:
            Tupla (a_actualizado, b_actualizado)
        """
        # Transformaciones (pesos compartidos)
        a_to_b = self.transform(a)
        b_to_a = self.transform(b)
        
        # Gate adaptativo para balancear la mezcla
        concat = torch.cat([a, b], dim=-1)
        gates = torch.sigmoid(self.gate(concat))  # [batch, 2]
        g_a, g_b = gates[..., 0:1], gates[..., 1:2]
        
        # Intercambio bidireccional
        a_new = self.norm(a + b_to_a * g_a * 0.5)
        b_new = self.norm(b + a_to_b * g_b * 0.5)
        
        return a_new, b_new


class SistemaFlujoCompleto(nn.Module):
    """
    Sistema de flujo COMPLETO entre 6 cajas.
    
    FASE 1 - IDA:    A → B → C → D → E → F
    FASE 2 - VUELTA: F → E → D → C → B → A  
    FASE 3 - BIDI:   A↔B, B↔C, C↔D, D↔E, E↔F (simultáneo)
    
    Economiza memoria:
        - Llaves separadas para IDA y VUELTA (aprenden patrones diferentes)
        - Llaves bidireccionales con pesos compartidos internamente
        - Solo 1 LayerNorm por llave
    """
    
    def __init__(self, dim: int, num_cajas: int = 6):
        """
        Args:
            dim: Dimensión de cada caja
            num_cajas: Número de cajas (default: 6)
        """
        super().__init__()
        self.dim = dim
        self.num_cajas = num_cajas
        
        # Llaves de IDA: A→B, B→C, C→D, D→E, E→F (5 llaves)
        self.llaves_ida = nn.ModuleList([
            LlaveConexion(dim) for _ in range(num_cajas - 1)
        ])
        
        # Llaves de VUELTA: F→E, E→D, D→C, C→B, B→A (5 llaves)
        self.llaves_vuelta = nn.ModuleList([
            LlaveConexion(dim) for _ in range(num_cajas - 1)
        ])
        
        # Llaves BIDIRECCIONALES: A↔B, B↔C, C↔D, D↔E, E↔F (5 llaves)
        self.llaves_bidi = nn.ModuleList([
            LlaveBidireccional(dim) for _ in range(num_cajas - 1)
        ])
        
        # Fusion final después de las 3 fases
        self.fusion_fases = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
    
    def forward(self, cajas: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Procesa lista de 6 tensores [A, B, C, D, E, F].
        
        Args:
            cajas: Lista de 6 tensores de shape (batch, dim)
            
        Returns:
            Lista actualizada después de las 3 fases de flujo
        """
        assert len(cajas) == self.num_cajas, f"Esperaba {self.num_cajas} cajas"
        
        # Guardar residuales originales
        residuales = [c.clone() for c in cajas]
        
        # ========== FASE 1: IDA (A → B → C → D → E → F) ==========
        for i in range(self.num_cajas - 1):
            cajas[i + 1] = self.llaves_ida[i](cajas[i], cajas[i + 1])
        
        # ========== FASE 2: VUELTA (F → E → D → C → B → A) ==========
        for i in range(self.num_cajas - 1, 0, -1):
            idx_llave = self.num_cajas - 1 - i
            cajas[i - 1] = self.llaves_vuelta[idx_llave](cajas[i], cajas[i - 1])
        
        # ========== FASE 3: BIDIRECCIONAL (todos los pares simultáneo) ==========
        nuevas = list(cajas)  # Copiar para actualizar simultáneamente
        for i in range(self.num_cajas - 1):
            a_new, b_new = self.llaves_bidi[i](cajas[i], cajas[i + 1])
            nuevas[i] = a_new
            nuevas[i + 1] = b_new
        cajas = nuevas
        
        # ========== FUSION FINAL ==========
        # Aplicar fusion y añadir residuales
        cajas = [self.fusion_fases(c) + r for c, r in zip(cajas, residuales)]
        
        return cajas
