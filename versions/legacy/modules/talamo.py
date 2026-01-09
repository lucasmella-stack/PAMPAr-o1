# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
"""
Tálamo - El router central del cerebro

En el cerebro real:
- Recibe información sensorial
- Distribuye a las áreas corticales apropiadas
- Modula el nivel de activación de cada área
- NO apaga áreas, solo modula su "volumen"

En LLARRI v7:
- Recibe el input
- Calcula qué tan relevante es cada módulo
- Propaga activación a módulos vecinos
- Mantiene actividad basal en todos los módulos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Talamo(nn.Module):
    """
    Router central que modula (no apaga) los módulos especializados.
    
    Principios:
    1. Actividad basal siempre > 0 (nada se apaga)
    2. Modulación continua entre basal y máximo
    3. Propagación a módulos vecinos
    """
    
    def __init__(
        self, 
        dim: int,
        n_modulos: int = 6,
        actividad_basal: float = 0.15,
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        self.actividad_basal = actividad_basal
        
        # Proyección para calcular relevancia de cada módulo
        self.relevancia = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, n_modulos),
        )
        
        # Matriz de adyacencia entre módulos (aprendible)
        # Define qué tan "cerca" está cada módulo de otro
        # Inicialización: diagonal fuerte + conexiones suaves
        self.adyacencia = nn.Parameter(
            torch.eye(n_modulos) * 0.5 + torch.ones(n_modulos, n_modulos) * 0.1
        )
        
        # Factor de propagación (cuánto se propaga la activación)
        self.factor_propagacion = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la modulación para cada módulo.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            modulacion: (batch, n_modulos) valores entre actividad_basal y 1.0
        """
        # Contexto global (promedio de la secuencia)
        contexto = x.mean(dim=1)  # (batch, dim)
        
        # 1. Calcular relevancia primaria de cada módulo
        relevancia_primaria = torch.sigmoid(self.relevancia(contexto))  # (batch, n_modulos)
        
        # 2. Propagar activación a vecinos
        # Normalizar adyacencia para que sea estocástica por filas
        adyacencia_norm = F.softmax(self.adyacencia, dim=-1)
        
        # Propagación: si módulo i está activo, sus vecinos también suben
        propagacion = relevancia_primaria @ adyacencia_norm
        
        # Mezclar relevancia primaria con propagación
        relevancia_final = (
            relevancia_primaria + 
            self.factor_propagacion * propagacion
        ) / (1 + self.factor_propagacion)
        
        # 3. Escalar a rango [actividad_basal, 1.0]
        # Nunca menos que actividad basal (nada se apaga)
        modulacion = (
            self.actividad_basal + 
            relevancia_final * (1.0 - self.actividad_basal)
        )
        
        return modulacion
    
    def get_adyacencia(self) -> torch.Tensor:
        """Retorna la matriz de adyacencia normalizada (para visualización)."""
        return F.softmax(self.adyacencia, dim=-1).detach()


class TalamoConMemoria(Talamo):
    """
    Tálamo que también considera el estado anterior.
    Permite modulación temporal (inercia en la activación).
    """
    
    def __init__(
        self,
        dim: int,
        n_modulos: int = 6,
        actividad_basal: float = 0.15,
        inercia: float = 0.3,
    ):
        super().__init__(dim, n_modulos, actividad_basal)
        self.inercia = inercia
        self.estado_anterior = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcular modulación actual
        modulacion_actual = super().forward(x)
        
        # Si hay estado anterior, mezclar (inercia)
        if self.estado_anterior is not None:
            # Asegurar mismo tamaño de batch
            if self.estado_anterior.shape[0] == modulacion_actual.shape[0]:
                modulacion_actual = (
                    self.inercia * self.estado_anterior +
                    (1 - self.inercia) * modulacion_actual
                )
        
        # Guardar para siguiente paso
        self.estado_anterior = modulacion_actual.detach()
        
        return modulacion_actual
    
    def reset_estado(self):
        """Reinicia el estado para nueva secuencia."""
        self.estado_anterior = None
