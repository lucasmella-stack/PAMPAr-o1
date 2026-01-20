# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Integración Cerebral con Consenso Real

El integrador ahora:
1. Recibe VOTOS de cada módulo (no solo outputs)
2. Calcula CONSENSO real basado en los votos
3. El módulo LÍDER tiene más peso en la integración
4. Si hay CONFLICTO, puede solicitar más iteraciones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class DetectorConsensoReal(nn.Module):
    """
    Calcula consenso basado en los votos de cada módulo.
    
    El consenso es REAL: mide si los módulos están de acuerdo
    sobre qué token/dirección tomar.
    """
    
    def __init__(self, dim: int, n_modulos: int = 6):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        self.dim_voto = dim // 2
        
        # Umbral de consenso aprendible
        self.umbral_consenso = nn.Parameter(torch.tensor(0.7))
        
        # Para calcular "conflicto" (desacuerdo fuerte)
        self.detector_conflicto = nn.Sequential(
            nn.Linear(n_modulos * self.dim_voto, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        votos: List[torch.Tensor],  # Lista de (batch, dim_voto)
        liderazgo: torch.Tensor,    # (batch, n_modulos)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Calcula nivel de consenso entre módulos.
        
        Args:
            votos: Lista de tensores (batch, dim_voto), uno por módulo
            liderazgo: (batch, n_modulos) probabilidades de liderazgo
            
        Returns:
            consenso: (batch,) valor 0-1
            conflicto: (batch,) valor 0-1
            stats: diccionario con estadísticas
        """
        batch = votos[0].shape[0]
        device = votos[0].device
        
        # Stack votos
        votos_stacked = torch.stack(votos, dim=1)  # (batch, n_modulos, dim_voto)
        
        # === CONSENSO: Similitud entre votos ===
        # Calcular similitud coseno entre todos los pares
        votos_norm = F.normalize(votos_stacked, dim=-1)
        
        # Matriz de similitud: (batch, n_modulos, n_modulos)
        similitud = torch.bmm(votos_norm, votos_norm.transpose(1, 2))
        
        # Promediar similitudes (excluyendo diagonal)
        mask = 1 - torch.eye(self.n_modulos, device=device)
        mask = mask.unsqueeze(0).expand(batch, -1, -1)
        
        sim_sin_diag = similitud * mask
        consenso = sim_sin_diag.sum(dim=(1, 2)) / (self.n_modulos * (self.n_modulos - 1))
        
        # === CONSENSO PONDERADO POR LIDERAZGO ===
        # El voto del líder importa más
        # Calcular similitud de cada módulo con el líder
        lider_idx = liderazgo.argmax(dim=-1)  # (batch,)
        
        voto_lider = torch.gather(
            votos_stacked, 
            1, 
            lider_idx.view(-1, 1, 1).expand(-1, 1, self.dim_voto)
        ).squeeze(1)  # (batch, dim_voto)
        
        voto_lider_norm = F.normalize(voto_lider, dim=-1)
        
        # Similitud de cada módulo con el líder
        sim_con_lider = (votos_norm * voto_lider_norm.unsqueeze(1)).sum(dim=-1)  # (batch, n_modulos)
        
        # Consenso ponderado (más peso a acuerdo con líder)
        consenso_ponderado = (sim_con_lider * liderazgo).sum(dim=-1)
        
        # === CONFLICTO: Desacuerdo fuerte ===
        votos_flat = votos_stacked.view(batch, -1)
        conflicto = self.detector_conflicto(votos_flat).squeeze(-1)
        
        # Stats
        stats = {
            'consenso_mean': consenso.mean().item(),
            'consenso_ponderado_mean': consenso_ponderado.mean().item(),
            'conflicto_mean': conflicto.mean().item(),
        }
        
        # Agregar similitud con líder por módulo
        nombres = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
        for i, nombre in enumerate(nombres):
            stats[f'sim_{nombre}_lider'] = sim_con_lider[:, i].mean().item()
        
        return consenso_ponderado, conflicto, stats


class IntegradorConLiderazgo(nn.Module):
    """
    Integra outputs de módulos considerando el liderazgo.
    
    El módulo líder tiene MÁS PESO en la integración final.
    Los seguidores contribuyen según su acoplamiento.
    """
    
    def __init__(
        self, 
        dim: int, 
        n_modulos: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        
        # Attention para refinar integración
        self.attention = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )
        
        # Proyección final
        self.proyeccion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # Normalización
        self.norm = nn.LayerNorm(dim)
        
        # Factor de amplificación para el líder
        self.amplificacion_lider = nn.Parameter(torch.tensor(1.5))
        
    def forward(
        self,
        outputs_modulos: List[torch.Tensor],
        modulacion: torch.Tensor,
        liderazgo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integra las salidas de los módulos con sesgo hacia el líder.
        
        Args:
            outputs_modulos: Lista de (batch, seq, dim)
            modulacion: (batch, n_modulos) intensidades
            liderazgo: (batch, n_modulos) probabilidades de liderazgo
            
        Returns:
            output integrado: (batch, seq, dim)
        """
        batch, seq, dim = outputs_modulos[0].shape
        
        # Apilar outputs
        stacked = torch.stack(outputs_modulos, dim=1)  # (batch, n_modulos, seq, dim)
        
        # === PESOS DE INTEGRACIÓN ===
        # Combinar modulación con liderazgo
        # El líder tiene peso amplificado
        pesos = modulacion * (1 + liderazgo * self.amplificacion_lider)
        
        # Normalizar pesos
        pesos = pesos / (pesos.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Expandir para multiplicar
        pesos_expanded = pesos.view(batch, self.n_modulos, 1, 1)
        
        # Ponderar
        ponderado = stacked * pesos_expanded
        
        # Sumar módulos
        integrado = ponderado.sum(dim=1)  # (batch, seq, dim)
        
        # Refinar con attention
        integrado = self.norm(integrado)
        refinado, _ = self.attention(integrado, integrado, integrado)
        
        # Proyección final
        output = self.proyeccion(refinado + integrado)
        
        return output


class CoordinadorCerebral(nn.Module):
    """
    Coordina todo el procesamiento cerebral.
    
    Funciones:
    1. Gestiona el ciclo tálamo → módulos → integración
    2. Genera la señal de liderazgo
    3. Detecta si hay consenso suficiente
    4. Puede solicitar más iteraciones si hay conflicto
    """
    
    def __init__(
        self,
        dim: int,
        n_modulos: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        
        # Generador de señal de liderazgo
        self.generador_senal = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # Detector de consenso
        self.detector_consenso = DetectorConsensoReal(dim, n_modulos)
        
        # Integrador
        self.integrador = IntegradorConLiderazgo(dim, n_modulos, n_heads, dropout)
        
    def generar_senal_lider(
        self,
        outputs_modulos: List[torch.Tensor],
        liderazgo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Genera la señal que el líder envía a los seguidores.
        
        Args:
            outputs_modulos: Lista de outputs de cada módulo
            liderazgo: (batch, n_modulos) probabilidades
            
        Returns:
            senal: (batch, dim)
        """
        batch = outputs_modulos[0].shape[0]
        device = outputs_modulos[0].device
        
        # Combinar outputs ponderados por liderazgo
        senal = torch.zeros(batch, self.dim, device=device)
        for i, output in enumerate(outputs_modulos):
            senal += liderazgo[:, i:i+1] * output.mean(dim=1)
        
        # Procesar señal
        senal = self.generador_senal(senal)
        
        return senal
    
    def forward(
        self,
        outputs_modulos: List[torch.Tensor],
        votos: List[torch.Tensor],
        modulacion: torch.Tensor,
        liderazgo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Coordina la integración cerebral.
        
        Args:
            outputs_modulos: Lista de (batch, seq, dim)
            votos: Lista de (batch, dim_voto)
            modulacion: (batch, n_modulos)
            liderazgo: (batch, n_modulos)
            
        Returns:
            output: (batch, seq, dim)
            consenso: (batch,)
            conflicto: (batch,)
            stats: dict
        """
        # Detectar consenso
        consenso, conflicto, stats = self.detector_consenso(votos, liderazgo)
        
        # Integrar
        output = self.integrador(outputs_modulos, modulacion, liderazgo)
        
        return output, consenso, conflicto, stats
