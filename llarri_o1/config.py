# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Configuración del modelo LLARRI-O1.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """
    Configuración del modelo LLARRI-O1 v4.0 HyperComprimido.
    
    Attributes:
        input_dim: Dimensión de entrada (784 para MNIST 28x28)
        hidden_dim: Dimensión oculta (debe ser divisible por 4)
        output_dim: Dimensión de salida (10 para clasificación)
        num_cajas_datos: Número de cajas de datos (default: 3)
        num_cajas_calculos: Número de cajas de cálculos (default: 3)
        niveles_fractales: Lista de dimensiones para cada nivel fractal
        dropout: Probabilidad de dropout
    """
    input_dim: int = 784
    hidden_dim: int = 1024
    output_dim: int = 10
    num_cajas_datos: int = 3
    num_cajas_calculos: int = 3
    niveles_fractales: List[int] = None  # Se calcula en __post_init__
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validaciones y defaults de configuración."""
        assert self.hidden_dim % 4 == 0, "hidden_dim debe ser divisible por 4"
        
        quad_dim = self.hidden_dim // 4
        
        # Calcular niveles fractales por defecto basado en quad_dim
        if self.niveles_fractales is None:
            # Niveles: 2, 4, 8, ... hasta el máximo permitido por quad_dim
            self.niveles_fractales = []
            nivel = 2
            while nivel <= quad_dim:
                self.niveles_fractales.append(nivel)
                nivel *= 2
        
        # Validar que los niveles no excedan quad_dim
        max_nivel = max(self.niveles_fractales)
        assert quad_dim >= max_nivel, (
            f"quad_dim ({quad_dim}) debe ser >= nivel máximo ({max_nivel}). "
            f"Aumenta hidden_dim a {max_nivel * 4} o reduce niveles_fractales."
        )
