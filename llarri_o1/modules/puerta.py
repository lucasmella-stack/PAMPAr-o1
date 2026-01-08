# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Módulo PUERTA (Cajas 1-3)

Gates rápidos con fórmulas O(n) para decidir cuánta información
pasa a cada escala (2→256).

Concepto: "¿Qué escalas son importantes para este input?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class CajaMedirEnergia(nn.Module):
    """
    CAJA 1: Mide energía por escala
    
    Para cada escala s ∈ [2,4,8,16,32,64,128,256]:
    - Calcula diferencias entre posiciones separadas por s
    - Mayor diferencia = más información a esa escala
    
    Fórmula: energia[s] = mean(|x[i] - x[i-s]|)
    Complejidad: O(n)
    """
    
    def __init__(self, escalas: List[int] = [2, 4, 8, 16, 32, 64, 128, 256]):
        super().__init__()
        self.escalas = escalas
        self.n_escalas = len(escalas)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            energias: [batch, n_escalas]
        """
        batch, seq_len, embed_dim = x.shape
        energias = []
        
        for s in self.escalas:
            if s < seq_len:
                # Diferencia entre posiciones separadas por s
                diff = x[:, s:, :] - x[:, :-s, :]  # [batch, seq_len-s, embed_dim]
                # Energía = magnitud promedio de las diferencias
                energia = diff.abs().mean(dim=(1, 2))  # [batch]
            else:
                # Escala mayor que secuencia = energía 0
                energia = torch.zeros(batch, device=x.device)
            energias.append(energia)
        
        return torch.stack(energias, dim=1)  # [batch, n_escalas]


class CajaLocalizarPicos(nn.Module):
    """
    CAJA 2: Localiza picos de información
    
    Calcula entropía local por ventana de cada escala.
    Alto contraste de entropía = información concentrada.
    
    Fórmula: varianza de energía local por escala
    Complejidad: O(n)
    """
    
    def __init__(self, escalas: List[int] = [2, 4, 8, 16, 32, 64, 128, 256]):
        super().__init__()
        self.escalas = escalas
        self.n_escalas = len(escalas)
        
    def forward(self, x: torch.Tensor, energias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            energias: [batch, n_escalas] de Caja 1
        Returns:
            picos: [batch, n_escalas] - score de concentración por escala
        """
        batch, seq_len, embed_dim = x.shape
        picos = []
        
        for i, s in enumerate(self.escalas):
            if s < seq_len:
                # Varianza de la norma por posición a escala s
                # Ventanas de tamaño s
                n_ventanas = seq_len // s
                if n_ventanas > 1:
                    x_reshape = x[:, :n_ventanas*s, :].view(batch, n_ventanas, s, embed_dim)
                    normas_ventana = x_reshape.norm(dim=(2, 3))  # [batch, n_ventanas]
                    varianza = normas_ventana.var(dim=1)  # [batch]
                    pico = varianza / (energias[:, i] + 1e-6)  # Normalizado
                else:
                    pico = torch.zeros(batch, device=x.device)
            else:
                pico = torch.zeros(batch, device=x.device)
            picos.append(pico)
        
        return torch.stack(picos, dim=1)  # [batch, n_escalas]


class CajaGenerarGates(nn.Module):
    """
    CAJA 3: Genera gates (pesos) para cada escala
    
    Combina energía y picos para decidir importancia de cada escala.
    
    Fórmula: gates = softmax(α·energias + β·picos + bias)
    Complejidad: O(1) después de recibir métricas
    """
    
    def __init__(self, n_escalas: int = 8, embed_dim: int = 128):
        super().__init__()
        self.n_escalas = n_escalas
        
        # Pesos aprendibles para combinar métricas
        self.alpha = nn.Parameter(torch.ones(n_escalas))  # Peso de energía
        self.beta = nn.Parameter(torch.ones(n_escalas) * 0.5)  # Peso de picos
        self.bias = nn.Parameter(torch.zeros(n_escalas))  # Bias por escala
        
        # Proyección opcional del embedding para refinar gates
        self.proj = nn.Linear(embed_dim, n_escalas)
        
    def forward(self, x: torch.Tensor, energias: torch.Tensor, picos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            energias: [batch, n_escalas]
            picos: [batch, n_escalas]
        Returns:
            gates: [batch, n_escalas] - suman 1.0
        """
        # Combinar métricas
        scores = self.alpha * energias + self.beta * picos + self.bias
        
        # Añadir información del embedding (promedio de la secuencia)
        x_mean = x.mean(dim=1)  # [batch, embed_dim]
        embed_scores = self.proj(x_mean)  # [batch, n_escalas]
        
        # Combinar
        final_scores = scores + 0.1 * embed_scores
        
        # Softmax para que sumen 1
        gates = F.softmax(final_scores, dim=-1)
        
        return gates


class ModuloPuerta(nn.Module):
    """
    CAJAS 1-3: Módulo Puerta completo
    
    Decide rápidamente qué escalas son importantes para el input.
    Fórmulas O(n) para velocidad.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        escalas: List[int] = [2, 4, 8, 16, 32, 64, 128, 256]
    ):
        super().__init__()
        self.escalas = escalas
        self.n_escalas = len(escalas)
        self.embed_dim = embed_dim
        
        # Las 3 cajas
        self.caja1_energia = CajaMedirEnergia(escalas)
        self.caja2_picos = CajaLocalizarPicos(escalas)
        self.caja3_gates = CajaGenerarGates(len(escalas), embed_dim)
        
        print(f"✓ ModuloPuerta: {len(escalas)} escalas {escalas}")
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            gates: [batch, n_escalas]
            metricas: dict con energias, picos para debug
        """
        # Caja 1: Medir energía
        energias = self.caja1_energia(x)
        
        # Caja 2: Localizar picos
        picos = self.caja2_picos(x, energias)
        
        # Caja 3: Generar gates
        gates = self.caja3_gates(x, energias, picos)
        
        metricas = {
            'energias': energias,
            'picos': picos,
            'gates': gates
        }
        
        return gates, metricas


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Test ModuloPuerta (Cajas 1-3)")
    print("=" * 60)
    
    puerta = ModuloPuerta(embed_dim=128)
    
    # Input de prueba
    x = torch.randn(4, 64, 128)  # batch=4, seq=64, embed=128
    
    gates, metricas = puerta(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Gates shape: {gates.shape}")
    print(f"Gates sum: {gates.sum(dim=1)}")  # Debe ser ~1.0
    print(f"\nEscalas: {puerta.escalas}")
    print(f"Gates ejemplo: {gates[0].tolist()}")
    
    # Verificar parámetros
    params = sum(p.numel() for p in puerta.parameters())
    print(f"\nParámetros: {params:,}")
    
    print("\n✅ ModuloPuerta funcionando!")
