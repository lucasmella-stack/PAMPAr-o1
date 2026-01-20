# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Módulos Especializados con Acoplamiento

Cada módulo ahora puede:
1. Procesar INDEPENDIENTE (como antes)
2. Recibir SEÑAL del líder y ajustar su procesamiento
3. CONTRIBUIR al consenso según su rol
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class ModuloAcoplable(nn.Module):
    """
    Clase base para módulos que pueden acoplarse a un líder.
    
    Cada módulo tiene:
    - Procesamiento propio (especializado)
    - Receptor de señal de liderazgo
    - Gate de acoplamiento (cuánto sigue al líder)
    """
    
    dominio: str = "base"
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Procesamiento propio
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # === RECEPTOR DE SEÑAL ===
        # Recibe la señal del líder y la adapta al dominio del módulo
        self.receptor_senal = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        # === GATE DE ACOPLAMIENTO ===
        # Decide cuánto del procesamiento propio vs. adaptado al líder
        self.gate_acoplamiento = nn.Sequential(
            nn.Linear(dim * 2, dim),  # [propio, señal]
            nn.Sigmoid(),
        )
        
        # === CONTRIBUCIÓN AL CONSENSO ===
        # Cómo este módulo "vota" en la decisión final
        self.proyector_voto = nn.Linear(dim, dim // 2)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        """Procesamiento independiente del módulo."""
        # Attention
        residual = x
        x_norm = self.norm1(x)
        x_att, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + x_att
        
        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_ff = self.ffn(x_norm)
        x = residual + x_ff
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        senal_lider: Optional[torch.Tensor] = None,
        intensidad_acoplamiento: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa input, opcionalmente acoplándose al líder.
        
        Args:
            x: (batch, seq, dim) input
            senal_lider: (batch, dim) señal del módulo líder (opcional)
            intensidad_acoplamiento: 0-1, qué tan fuerte es el acoplamiento
            
        Returns:
            output: (batch, seq, dim) resultado procesado
            voto: (batch, dim//2) contribución al consenso
        """
        # 1. Procesamiento propio
        output_propio = self.procesar_propio(x)
        
        # 2. Si hay señal de líder, ajustar
        if senal_lider is not None and intensidad_acoplamiento > 0:
            batch, seq, dim = output_propio.shape
            
            # Expandir señal a toda la secuencia
            senal_expanded = senal_lider.unsqueeze(1).expand(-1, seq, -1)
            
            # Procesar señal según el dominio del módulo
            senal_adaptada = self.receptor_senal(senal_expanded)
            
            # Gate: cuánto seguir al líder vs. mantener propio
            gate_input = torch.cat([
                output_propio.mean(dim=1, keepdim=True).expand(-1, seq, -1),
                senal_adaptada
            ], dim=-1)
            gate = self.gate_acoplamiento(gate_input)
            
            # Mezclar: propio + ajuste del líder
            output = output_propio + gate * intensidad_acoplamiento * senal_adaptada
        else:
            output = output_propio
            
        # 3. Calcular voto para consenso
        voto = self.proyector_voto(output.mean(dim=1))  # (batch, dim//2)
        
        return output, voto


class ModuloLenguajeAcoplable(ModuloAcoplable):
    """Módulo de Lenguaje con capacidad de acoplamiento."""
    
    dominio = "lenguaje"
    
    def __init__(self, dim: int, n_heads: int = 4, n_roles: int = 8, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        # Roles gramaticales
        self.roles = nn.Parameter(torch.randn(n_roles, dim) * 0.1)
        self.asignador_roles = nn.Linear(dim, n_roles)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x = super().procesar_propio(x)
        
        # Añadir información de roles gramaticales
        pesos_roles = F.softmax(self.asignador_roles(x), dim=-1)
        info_roles = pesos_roles @ self.roles
        
        return x + 0.1 * info_roles


class ModuloLogicaAcoplable(ModuloAcoplable):
    """Módulo de Lógica con capacidad de acoplamiento."""
    
    dominio = "logica"
    
    def __init__(self, dim: int, n_heads: int = 4, n_relaciones: int = 8, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        # Relaciones lógicas
        self.relaciones = nn.Parameter(torch.randn(n_relaciones, dim, dim) * 0.02)
        self.detector_relacion = nn.Linear(dim * 2, n_relaciones)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x = super().procesar_propio(x)
        
        batch, seq, dim = x.shape
        contexto = x.mean(dim=1, keepdim=True).expand(-1, seq, -1)
        pares = torch.cat([x, contexto], dim=-1)
        
        pesos_rel = F.softmax(self.detector_relacion(pares), dim=-1)
        
        transformado = torch.zeros_like(x)
        for i in range(self.relaciones.shape[0]):
            t = torch.einsum('bsd,de->bse', x, self.relaciones[i])
            transformado += pesos_rel[..., i:i+1] * t
            
        return x + 0.1 * transformado


class ModuloMatematicasAcoplable(ModuloAcoplable):
    """Módulo de Matemáticas con capacidad de acoplamiento."""
    
    dominio = "matematicas"
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        # Representación numérica
        self.linea_numerica = nn.Parameter(torch.randn(100, dim) * 0.1)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x = super().procesar_propio(x)
        
        similitud = x @ self.linea_numerica.T
        magnitud = similitud.softmax(dim=-1) @ self.linea_numerica
        
        return x + 0.1 * magnitud


class ModuloPatronesAcoplable(ModuloAcoplable):
    """Módulo de Patrones con capacidad de acoplamiento."""
    
    dominio = "patrones"
    
    def __init__(self, dim: int, n_heads: int = 4, n_escalas: int = 4, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        # Convoluciones multiescala
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=2**i, padding=2**(i-1))
            for i in range(1, n_escalas + 1)
        ])
        self.fusion = nn.Linear(dim * n_escalas, dim)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x_att = super().procesar_propio(x)
        
        x_conv = x.transpose(1, 2)
        escalas = []
        for conv in self.convs:
            e = conv(x_conv)
            if e.shape[2] != x.shape[1]:
                e = F.interpolate(e, size=x.shape[1])
            escalas.append(e)
        
        multi_escala = torch.cat(escalas, dim=1).transpose(1, 2)
        patrones = self.fusion(multi_escala)
        
        return x_att + 0.1 * patrones


class ModuloContextoAcoplable(ModuloAcoplable):
    """Módulo de Contexto con capacidad de acoplamiento."""
    
    dominio = "contexto"
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        self.query_global = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.attention_global = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x = super().procesar_propio(x)
        
        batch = x.shape[0]
        query = self.query_global.expand(batch, -1, -1)
        contexto_global, _ = self.attention_global(query, x, x)
        contexto_broadcast = contexto_global.expand(-1, x.shape[1], -1)
        
        return x + 0.1 * contexto_broadcast


class ModuloCreatividadAcoplable(ModuloAcoplable):
    """Módulo de Creatividad con capacidad de acoplamiento."""
    
    dominio = "creatividad"
    
    def __init__(self, dim: int, n_heads: int = 4, ruido: float = 0.1, dropout: float = 0.1):
        super().__init__(dim, n_heads, dropout)
        
        self.ruido_base = ruido
        self.proyeccion_lateral = nn.Linear(dim, dim)
        self.mezcla = nn.Linear(dim * 2, dim)
        
    def procesar_propio(self, x: torch.Tensor) -> torch.Tensor:
        x = super().procesar_propio(x)
        
        if self.training:
            ruido = torch.randn_like(x) * self.ruido_base
        else:
            ruido = 0
            
        lateral = torch.tanh(self.proyeccion_lateral(x + ruido))
        combinado = torch.cat([x, lateral], dim=-1)
        creativo = self.mezcla(combinado)
        
        return x + 0.1 * creativo
