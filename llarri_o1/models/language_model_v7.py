# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi <lucas@segundacabeza.com>
# Coordinator: Alvaro <alvaro@segundacabeza.com>
"""
LLARRI v7 - Modelo de Lenguaje con Arquitectura Cerebral

Arquitectura inspirada en el cerebro humano:

┌─────────────────────────────────────────────────────────────────┐
│                      LLARRI v7                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input → Embedding → Tálamo (modulador)                        │
│                         │                                       │
│          ┌──────────────┼──────────────┐                       │
│          ▼              ▼              ▼                        │
│     [Lenguaje]    [Lógica]    [Matemáticas]                    │
│     [Patrones]    [Contexto]  [Creatividad]                    │
│          │              │              │                        │
│          └──────────────┼──────────────┘                       │
│                         ▼                                       │
│              Integrador Cerebral                               │
│                         │                                       │
│                         ▼                                       │
│                   Hipocampo (memoria)                          │
│                         │                                       │
│                         ▼                                       │
│                  Output → Logits                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Principios:
1. Especialización: cada módulo tiene una tarea específica
2. No interferencia: los módulos no se meten en el dominio de otros
3. Modulación continua: nada se apaga, todo se modula (15%-100%)
4. Propagación: la activación se propaga a módulos vecinos
5. Memoria: el hipocampo enriquece con experiencias pasadas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..modules.cerebral.talamo import Talamo, TalamoConMemoria
from ..modules.cerebral.modulos_especializados import (
    ModuloLenguaje,
    ModuloLogica,
    ModuloMatematicas,
    ModuloPatrones,
    ModuloContexto,
    ModuloCreatividad,
)
from ..modules.cerebral.integracion import IntegradorCerebral, DetectorConsenso
from ..modules.cerebral.hipocampo import Hipocampo


class LLARRIv7Cerebral(nn.Module):
    """
    LLARRI v7 - Arquitectura Cerebral
    
    Modelo de lenguaje con módulos especializados inspirados
    en la organización del cerebro humano.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        n_heads: int = 4,
        actividad_basal: float = 0.15,
        usar_hipocampo: bool = True,
        capacidad_memoria: int = 5000,
        max_iteraciones: int = 2,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_modulos = 6
        self.max_iteraciones = max_iteraciones
        self.usar_hipocampo = usar_hipocampo
        
        # === EMBEDDING ===
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Embedding(4096, dim)
        
        # === TÁLAMO (Router/Modulador) ===
        self.talamo = TalamoConMemoria(
            dim=dim,
            n_modulos=self.n_modulos,
            actividad_basal=actividad_basal,
        )
        
        # === MÓDULOS ESPECIALIZADOS ===
        self.modulos = nn.ModuleDict({
            'lenguaje': ModuloLenguaje(dim, n_heads),
            'logica': ModuloLogica(dim, n_heads),
            'matematicas': ModuloMatematicas(dim, n_heads),
            'patrones': ModuloPatrones(dim, n_heads),
            'contexto': ModuloContexto(dim, n_heads),
            'creatividad': ModuloCreatividad(dim, n_heads),
        })
        self.nombres_modulos = list(self.modulos.keys())
        
        # === INTEGRADOR ===
        self.integrador = IntegradorCerebral(
            dim=dim,
            n_modulos=self.n_modulos,
            n_heads=n_heads,
        )
        
        # === DETECTOR DE CONSENSO ===
        self.detector_consenso = DetectorConsenso(dim, self.n_modulos)
        
        # === HIPOCAMPO (Memoria) ===
        if usar_hipocampo:
            self.hipocampo = Hipocampo(
                dim=dim,
                capacidad=capacidad_memoria,
                k_memorias=5,
            )
        else:
            self.hipocampo = None
            
        # === OUTPUT ===
        self.norm_final = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        
        # Inicialización
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización de pesos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo.
        
        Args:
            input_ids: (batch, seq_len) tokens de entrada
            targets: (batch, seq_len) tokens objetivo (para training)
            
        Returns:
            dict con 'logits', 'loss' (si targets), y stats
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # === 1. EMBEDDING ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        
        # === 2. PROCESAMIENTO CEREBRAL ===
        stats = {}
        
        for iteracion in range(self.max_iteraciones):
            # 2a. Tálamo calcula modulación
            modulacion = self.talamo(x)  # (batch, n_modulos)
            
            # 2b. Cada módulo procesa (todos, modulados)
            outputs_modulos = []
            for i, (nombre, modulo) in enumerate(self.modulos.items()):
                out = modulo(x)
                outputs_modulos.append(out)
                
                # Guardar stats de modulación
                stats[f'mod_{nombre}'] = modulacion[:, i].mean().item()
            
            # 2c. Detectar consenso
            consenso, consenso_stats = self.detector_consenso(outputs_modulos)
            stats.update(consenso_stats)
            
            # 2d. Integrar outputs
            x = self.integrador(outputs_modulos, modulacion)
            
            # 2e. Early exit si hay alto consenso
            if consenso.mean() > 0.9 and iteracion < self.max_iteraciones - 1:
                stats['iteraciones'] = iteracion + 1
                break
        else:
            stats['iteraciones'] = self.max_iteraciones
            
        # === 3. HIPOCAMPO (Memoria) ===
        if self.hipocampo is not None:
            x = self.hipocampo(x)
            
        # === 4. OUTPUT ===
        x = self.norm_final(x)
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        
        # === 5. LOSS (si hay targets) ===
        result = {'logits': logits, 'stats': stats}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss
            
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Genera texto autoregressivamente.
        
        Args:
            input_ids: (batch, seq) prompt inicial
            max_new_tokens: cuántos tokens generar
            temperature: control de creatividad
            top_k: filtrar a top-k tokens
            
        Returns:
            tokens generados (batch, seq + max_new_tokens)
        """
        self.eval()
        
        # Reset estado del tálamo para nueva generación
        self.talamo.reset_estado()
        
        for _ in range(max_new_tokens):
            # Truncar si excede contexto máximo
            context = input_ids[:, -4096:]
            
            # Forward
            result = self(context)
            logits = result['logits'][:, -1, :]  # Último token
            
            # Temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Muestrear
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Añadir al contexto
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
    
    def get_num_params(self) -> int:
        """Retorna número de parámetros."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict:
        """Retorna configuración del modelo."""
        return {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'n_modulos': self.n_modulos,
            'n_heads': 4,
            'usar_hipocampo': self.usar_hipocampo,
            'max_iteraciones': self.max_iteraciones,
            'total_params': self.get_num_params(),
        }


class LLARRIv7Mini(LLARRIv7Cerebral):
    """Versión mini para pruebas rápidas."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__(
            vocab_size=vocab_size,
            dim=64,
            n_heads=2,
            usar_hipocampo=False,
            max_iteraciones=1,
        )


class LLARRIv7Base(LLARRIv7Cerebral):
    """Versión base balanceada."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__(
            vocab_size=vocab_size,
            dim=128,
            n_heads=4,
            usar_hipocampo=True,
            capacidad_memoria=5000,
            max_iteraciones=2,
        )


class LLARRIv7Large(LLARRIv7Cerebral):
    """Versión grande para mejor rendimiento."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__(
            vocab_size=vocab_size,
            dim=256,
            n_heads=8,
            usar_hipocampo=True,
            capacidad_memoria=10000,
            max_iteraciones=3,
        )


def crear_modelo_v7(
    size: str = 'base',
    vocab_size: int = 256,
) -> LLARRIv7Cerebral:
    """Factory function para crear modelos v7."""
    modelos = {
        'mini': LLARRIv7Mini,
        'base': LLARRIv7Base,
        'large': LLARRIv7Large,
    }
    
    if size not in modelos:
        raise ValueError(f"Size debe ser uno de: {list(modelos.keys())}")
        
    return modelos[size](vocab_size=vocab_size)
