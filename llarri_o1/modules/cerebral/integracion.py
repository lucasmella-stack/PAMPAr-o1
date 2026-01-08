# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
"""
Integración Cerebral - Cuerpo Calloso e Integrador

El Cuerpo Calloso conecta los hemisferios.
El Integrador combina todos los módulos especializados.

Principios:
- Comunicación ordenada (protocolos claros)
- Ponderación por modulación (no binario)
- Preservar especialización (cada módulo mantiene su identidad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CuerpoCalloso(nn.Module):
    """
    Conecta dos "hemisferios" o grupos de módulos.
    Permite que compartan información sin perder especialización.
    """
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        
        # Cross-attention bidireccional
        self.attn_izq_a_der = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.attn_der_a_izq = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # Gates para controlar cuánta información pasa
        self.gate_izq = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate_der = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        
        # Normalización
        self.norm_izq = nn.LayerNorm(dim)
        self.norm_der = nn.LayerNorm(dim)
        
    def forward(
        self, 
        izq: torch.Tensor, 
        der: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Intercambia información entre dos representaciones.
        
        Args:
            izq: (batch, seq, dim) representación "izquierda"
            der: (batch, seq, dim) representación "derecha"
            
        Returns:
            izq_actualizado, der_actualizado
        """
        # Izquierda mira a derecha
        info_de_der, _ = self.attn_izq_a_der(izq, der, der)
        
        # Derecha mira a izquierda
        info_de_izq, _ = self.attn_der_a_izq(der, izq, izq)
        
        # Gates: cuánta información nueva incorporar
        gate_i = self.gate_izq(torch.cat([izq, info_de_der], dim=-1))
        gate_d = self.gate_der(torch.cat([der, info_de_izq], dim=-1))
        
        # Actualizar con información cruzada (gated)
        izq_nuevo = self.norm_izq(izq + gate_i * info_de_der)
        der_nuevo = self.norm_der(der + gate_d * info_de_izq)
        
        return izq_nuevo, der_nuevo


class IntegradorCerebral(nn.Module):
    """
    Combina las salidas de todos los módulos especializados
    respetando la modulación del tálamo.
    
    NO es un simple promedio - cada módulo contribuye
    según su modulación Y su relevancia para el token.
    """
    
    def __init__(
        self, 
        dim: int, 
        n_modulos: int = 6,
        n_heads: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        
        # Attention para integración
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # Proyección final
        self.proyeccion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        # Normalización
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        outputs_modulos: List[torch.Tensor],
        modulaciones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integra las salidas de los módulos.
        
        Args:
            outputs_modulos: Lista de (batch, seq, dim) - una por módulo
            modulaciones: (batch, n_modulos) - del tálamo
            
        Returns:
            output integrado: (batch, seq, dim)
        """
        batch, seq, dim = outputs_modulos[0].shape
        
        # Apilar outputs: (batch, n_modulos, seq, dim)
        stacked = torch.stack(outputs_modulos, dim=1)
        
        # Expandir modulaciones: (batch, n_modulos, 1, 1)
        mod_expanded = modulaciones.unsqueeze(-1).unsqueeze(-1)
        
        # Ponderar por modulación
        ponderado = stacked * mod_expanded  # (batch, n_modulos, seq, dim)
        
        # Sumar módulos (cada token recibe contribución ponderada)
        integrado = ponderado.sum(dim=1)  # (batch, seq, dim)
        
        # Normalizar por suma de modulaciones (para mantener escala)
        suma_mod = modulaciones.sum(dim=-1, keepdim=True).unsqueeze(-1)
        integrado = integrado / (suma_mod + 1e-6)
        
        # Refinar con attention (permite interacción entre posiciones)
        integrado = self.norm(integrado)
        refinado, _ = self.attention(integrado, integrado, integrado)
        
        # Proyección final
        output = self.proyeccion(refinado + integrado)
        
        return output


class DetectorConsenso(nn.Module):
    """
    Detecta si los módulos están de acuerdo.
    
    Si hay alto consenso → respuesta confiable
    Si hay bajo consenso → necesita más procesamiento
    """
    
    def __init__(self, dim: int, n_modulos: int = 6):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        
        # Proyector para comparación
        self.proyector = nn.Linear(dim, dim // 2)
        
    def forward(
        self,
        outputs_modulos: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcula nivel de consenso entre módulos.
        
        Args:
            outputs_modulos: Lista de (batch, seq, dim)
            
        Returns:
            consenso: (batch,) valor 0-1
            stats: diccionario con estadísticas
        """
        batch = outputs_modulos[0].shape[0]
        
        # Proyectar cada módulo
        proyectados = [self.proyector(o.mean(dim=1)) for o in outputs_modulos]
        
        # Calcular similitud entre todos los pares
        similitudes = []
        for i in range(len(proyectados)):
            for j in range(i + 1, len(proyectados)):
                sim = F.cosine_similarity(proyectados[i], proyectados[j], dim=-1)
                similitudes.append(sim)
        
        # Consenso = promedio de similitudes
        if similitudes:
            similitudes = torch.stack(similitudes, dim=-1)
            consenso = similitudes.mean(dim=-1)
        else:
            consenso = torch.ones(batch, device=outputs_modulos[0].device)
        
        # Estadísticas
        stats = {
            'consenso_medio': consenso.mean().item(),
            'consenso_min': consenso.min().item(),
            'consenso_max': consenso.max().item(),
        }
        
        return consenso, stats
