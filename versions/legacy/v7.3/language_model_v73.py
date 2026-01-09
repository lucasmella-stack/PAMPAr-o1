# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.3 - Arquitectura Cerebral con Liderazgo Dinámico

Mejoras sobre v7.2:
1. Sistema de LIDERAZGO: un módulo domina según la tarea
2. ACOPLAMIENTO: los módulos se ajustan al líder
3. CONSENSO REAL: los módulos "votan" y se mide el acuerdo
4. Señal de coordinación: el líder guía a los seguidores

Arquitectura:

    Input → Embedding → Tálamo (selecciona líder)
                            │
                ┌───────────┴───────────┐
                │     SEÑAL LÍDER       │
                ▼           ▼           ▼
           [Lenguaje]  [Lógica]   [Matemáticas]
           [Patrones]  [Contexto] [Creatividad]
                │           │           │
                └───────────┼───────────┘
                            │ VOTOS
                            ▼
              Coordinador (consenso + integración)
                            │
                            ▼
                      Hipocampo
                            │
                            ▼
                        Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from ..modules.cerebral.talamo_liderazgo import TalamoConLiderazgo
from ..modules.cerebral.modulos_acoplables import (
    ModuloLenguajeAcoplable,
    ModuloLogicaAcoplable,
    ModuloMatematicasAcoplable,
    ModuloPatronesAcoplable,
    ModuloContextoAcoplable,
    ModuloCreatividadAcoplable,
)
from ..modules.cerebral.integracion_liderazgo import CoordinadorCerebral
from ..modules.cerebral.hipocampo import Hipocampo


class LLARRIv73Liderazgo(nn.Module):
    """
    LLARRI v7.3 - Arquitectura con Liderazgo Dinámico
    
    Cuando el modelo procesa "2 + 2", el módulo de Matemáticas LIDERA
    y los demás módulos se ACOPLAN a él.
    
    Cuando procesa "The cat sat on the mat", Lenguaje LIDERA
    y Contexto/Patrones lo SIGUEN.
    """
    
    def __init__(
        self,
        vocab_size: int = 8000,
        dim: int = 128,
        n_heads: int = 4,
        actividad_basal: float = 0.15,
        temperatura_liderazgo: float = 0.5,
        usar_hipocampo: bool = True,
        capacidad_memoria: int = 2000,
        max_iteraciones: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_modulos = 6
        self.max_iteraciones = max_iteraciones
        self.usar_hipocampo = usar_hipocampo
        self.dropout_rate = dropout
        
        # === EMBEDDING ===
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Embedding(4096, dim)
        self.dropout = nn.Dropout(dropout)
        
        # === TÁLAMO CON LIDERAZGO ===
        self.talamo = TalamoConLiderazgo(
            dim=dim,
            n_modulos=self.n_modulos,
            actividad_basal=actividad_basal,
            temperatura_seleccion=temperatura_liderazgo,
        )
        
        # === MÓDULOS ESPECIALIZADOS ACOPLABLES ===
        self.modulos = nn.ModuleDict({
            'lenguaje': ModuloLenguajeAcoplable(dim, n_heads, dropout=dropout),
            'logica': ModuloLogicaAcoplable(dim, n_heads, dropout=dropout),
            'matematicas': ModuloMatematicasAcoplable(dim, n_heads, dropout=dropout),
            'patrones': ModuloPatronesAcoplable(dim, n_heads, dropout=dropout),
            'contexto': ModuloContextoAcoplable(dim, n_heads, dropout=dropout),
            'creatividad': ModuloCreatividadAcoplable(dim, n_heads, dropout=dropout),
        })
        self.nombres_modulos = list(self.modulos.keys())
        
        # === COORDINADOR CEREBRAL ===
        self.coordinador = CoordinadorCerebral(
            dim=dim,
            n_modulos=self.n_modulos,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # === HIPOCAMPO ===
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
        Forward pass con liderazgo dinámico.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # === 1. EMBEDDING ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        x = self.dropout(x)
        
        # === 2. PROCESAMIENTO CEREBRAL CON LIDERAZGO ===
        stats = {}
        
        for iteracion in range(self.max_iteraciones):
            # 2a. Tálamo determina LÍDER y MODULACIÓN
            liderazgo, modulacion, talamo_stats = self.talamo(x)
            stats.update(talamo_stats)
            
            # 2b. Generar señal del líder (primera iteración sin señal)
            if iteracion == 0:
                senal_lider = None
            else:
                senal_lider = self.coordinador.generar_senal_lider(
                    outputs_modulos, liderazgo
                )
            
            # 2c. Cada módulo procesa CON ACOPLAMIENTO
            outputs_modulos = []
            votos = []
            
            for i, (nombre, modulo) in enumerate(self.modulos.items()):
                intensidad = modulacion[:, i].mean().item()
                
                output, voto = modulo(
                    x, 
                    senal_lider=senal_lider,
                    intensidad_acoplamiento=intensidad,
                )
                outputs_modulos.append(output)
                votos.append(voto)
                
                stats[f'mod_{nombre}'] = modulacion[:, i].mean().item()
            
            # 2d. Coordinar (consenso + integración)
            x, consenso, conflicto, coord_stats = self.coordinador(
                outputs_modulos, votos, modulacion, liderazgo
            )
            stats.update(coord_stats)
            
            # 2e. Early exit si hay alto consenso
            consenso_medio = consenso.mean().item()
            stats[f'consenso_iter_{iteracion}'] = consenso_medio
            stats[f'conflicto_iter_{iteracion}'] = conflicto.mean().item()
            
            if consenso_medio > 0.85 and iteracion < self.max_iteraciones - 1:
                stats['iteraciones'] = iteracion + 1
                stats['early_exit'] = True
                break
        else:
            stats['iteraciones'] = self.max_iteraciones
            stats['early_exit'] = False
            
        # === 3. HIPOCAMPO ===
        if self.hipocampo is not None:
            x = self.hipocampo(x)
            
        # === 4. OUTPUT ===
        x = self.norm_final(x)
        logits = self.output(x)
        
        # === 5. LOSS ===
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
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """
        Genera texto con liderazgo dinámico.
        """
        self.eval()
        self.talamo.reset_estado()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            if generated.shape[1] > 512:
                context = generated[:, -512:]
            else:
                context = generated
                
            # Forward
            result = self(context)
            logits = result['logits'][:, -1, :]  # Último token
            
            # Aplicar repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Temperature
            logits = logits / temperature
            
            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop en EOS (si existe)
            if next_token.item() == 0:  # Asumiendo 0 = EOS
                break
                
        return generated
    
    def get_estado_cerebral(self) -> Dict:
        """
        Retorna el estado actual del "cerebro" para visualización/debug.
        """
        return {
            'ultimo_lider': self.talamo.ultimo_lider,
            'matriz_acoplamiento': self.talamo.get_matriz_acoplamiento(),
        }
