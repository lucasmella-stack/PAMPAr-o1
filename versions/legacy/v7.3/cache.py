# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

"""
Cache binario para el nivel 2 del procesamiento fractal.
Pre-computa TODAS las operaciones posibles para lookup instantáneo.
"""

import torch


class CacheBinario:
    """
    Cache en RAM para el nivel binario (dim=2).
    
    Pre-computa una tabla de lookup con todas las combinaciones posibles
    de valores binarios y sus operaciones, permitiendo cálculos instantáneos.
    
    Operaciones pre-computadas:
        - Suma
        - Producto
        - Diferencia absoluta
        - Media
        - Máximo
        - Mínimo
        - XOR suave
    
    Example:
        >>> cache = CacheBinario(torch.device('cuda'))
        >>> x = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        >>> result = cache.lookup(x)  # Shape: (2, 7)
    """
    
    def __init__(self, device: torch.device):
        """
        Inicializa el cache en el dispositivo especificado.
        
        Args:
            device: Dispositivo PyTorch (cpu o cuda)
        """
        self.device = device
        self._tabla = None
        self._inicializar()
    
    def _inicializar(self):
        """Pre-computa tabla de lookup para todas las combinaciones."""
        # 4 estados posibles × 7 operaciones
        self._tabla = torch.zeros(4, 7, device=self.device)
        
        combinaciones = [
            [0.0, 0.0],  # Estado 0
            [0.0, 1.0],  # Estado 1
            [1.0, 0.0],  # Estado 2
            [1.0, 1.0],  # Estado 3
        ]
        
        for i, (a, b) in enumerate(combinaciones):
            self._tabla[i] = torch.tensor([
                a + b,                       # suma
                a * b,                       # producto
                abs(a - b),                  # diferencia
                (a + b) / 2,                 # media
                max(a, b),                   # máximo
                min(a, b),                   # mínimo
                a * (1 - b) + (1 - a) * b,   # XOR suave
            ], device=self.device)
    
    def to(self, device: torch.device) -> "CacheBinario":
        """
        Mueve el cache a otro dispositivo.
        
        Args:
            device: Nuevo dispositivo PyTorch
            
        Returns:
            Self para encadenamiento
        """
        self.device = device
        self._inicializar()
        return self
    
    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lookup vectorizado en la tabla.
        
        Args:
            x: Tensor de shape (batch, 2) con valores a buscar
            
        Returns:
            Tensor de shape (batch, 7) con los resultados de las 7 operaciones
        """
        # Cuantizar a índice 0-3: [a,b] -> a*2 + b
        x_bin = (x > 0.5).float()
        idx = (x_bin[..., 0] * 2 + x_bin[..., 1]).long()
        return self._tabla[idx]
