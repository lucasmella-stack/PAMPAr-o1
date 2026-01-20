# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
#
# ⚠️ LEGACY CODE - For historical reference only
# Imports reference modules that no longer exist in this structure
#
"""
LLARRI v7.4 - Arquitectura Cerebral con Tálamo Orquestador

El Tálamo ahora es un ORQUESTADOR con REGLAS CLARAS:
- Cada módulo tiene su LLAVE (patrones que lo activan)
- El Tálamo detecta el tipo de contenido y asigna liderazgo
- No hay ambigüedad: está claro quién se encarga de qué

Arquitectura:

    Input IDs → Embedding
                    │
                    ▼
             ┌─────────────┐
             │   TÁLAMO    │  ← Detecta tipo con REGLAS
             │ (Orquestador)│  ← Asigna LÍDER
             └─────────────┘
                    │
         ┌──────────┼──────────┐
         │ LÍDER    │ SEGUIDORES│
         ▼          ▼          ▼
    [Matemáticas] [Lógica] [Lenguaje]
    [Patrones]  [Contexto] [Creatividad]
         │          │          │
         └──────────┼──────────┘
                    │
                    ▼
              Integrador
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

from ..modules.cerebral.talamo_reglas import TalamoConReglas
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


class LLARRIv74Orquestado(nn.Module):
    """
    LLARRI v7.4 - El Tálamo es el Orquestador
    
    Principios:
    1. REGLAS CLARAS: cada módulo tiene patrones que lo activan
    2. LIDERAZGO DEFINIDO: el Tálamo decide quién lidera antes de procesar
    3. ACOPLAMIENTO: los seguidores reciben señal del líder
    4. CONSENSO: la salida es la integración coordinada
    """
    
    # Orden de los módulos (importante para el Tálamo)
    ORDEN_MODULOS = ['matematicas', 'logica', 'lenguaje', 'contexto', 'creatividad', 'patrones']
    
    def __init__(
        self,
        vocab_size: int = 8000,
        dim: int = 128,
        n_heads: int = 4,
        actividad_basal: float = 0.15,
        usar_hipocampo: bool = True,
        capacidad_memoria: int = 2000,
        dropout: float = 0.15,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_modulos = 6
        self.usar_hipocampo = usar_hipocampo
        self.dropout_rate = dropout
        
        # === EMBEDDING ===
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Embedding(4096, dim)
        self.dropout = nn.Dropout(dropout)
        
        # === TÁLAMO ORQUESTADOR ===
        self.talamo = TalamoConReglas(
            dim=dim,
            vocab_size=vocab_size,
            n_modulos=self.n_modulos,
            actividad_basal=actividad_basal,
        )
        
        # === MÓDULOS ESPECIALIZADOS ===
        # Orden: matematicas, logica, lenguaje, contexto, creatividad, patrones
        self.modulos = nn.ModuleDict({
            'matematicas': ModuloMatematicasAcoplable(dim, n_heads, dropout=dropout),
            'logica': ModuloLogicaAcoplable(dim, n_heads, dropout=dropout),
            'lenguaje': ModuloLenguajeAcoplable(dim, n_heads, dropout=dropout),
            'contexto': ModuloContextoAcoplable(dim, n_heads, dropout=dropout),
            'creatividad': ModuloCreatividadAcoplable(dim, n_heads, dropout=dropout),
            'patrones': ModuloPatronesAcoplable(dim, n_heads, dropout=dropout),
        })
        
        # === INTEGRADOR ===
        self.integrador = nn.Linear(dim, dim)
        self.norm_final = nn.LayerNorm(dim)
        
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
        self.output = nn.Linear(dim, vocab_size)
        
        # Estado
        self._tokenizer = None
        self._inicializado = False
        
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
                
    def inicializar_talamo(self, tokenizer):
        """
        Inicializa el Tálamo con el tokenizer.
        DEBE llamarse antes de entrenar/inferir.
        """
        self.talamo.inicializar(tokenizer)
        self._tokenizer = tokenizer
        self._inicializado = True
        
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con orquestación del Tálamo.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # === 1. EMBEDDING ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        x = self.dropout(x)
        
        # === 2. TÁLAMO DECIDE LÍDER ===
        liderazgo, modulacion, stats = self.talamo(input_ids, x)
        
        # Obtener índice y nombre del líder
        lider_idx = stats['lider_idx']
        nombre_lider = stats['lider']
        
        # === 3. LÍDER PROCESA PRIMERO ===
        modulo_lider = self.modulos[nombre_lider]
        salida_lider, voto_lider = modulo_lider(x, senal_lider=None, intensidad_acoplamiento=1.0)
        
        # Señal del líder = promedio de su salida (batch, dim)
        senal_lider = salida_lider.mean(dim=1)
        
        # === 4. SEGUIDORES PROCESAN CON SEÑAL DEL LÍDER ===
        outputs = {nombre_lider: salida_lider}
        votos = [voto_lider]
        
        # Matriz de acoplamiento
        acoplamiento = self.talamo.get_matriz_acoplamiento().to(device)
        
        for i, nombre in enumerate(self.ORDEN_MODULOS):
            if nombre != nombre_lider:
                # Intensidad de acoplamiento desde la matriz
                intensidad = acoplamiento[lider_idx, i].item()
                intensidad = max(intensidad, modulacion[0, i].item())
                
                modulo = self.modulos[nombre]
                output, voto = modulo(x, senal_lider=senal_lider, intensidad_acoplamiento=intensidad)
                outputs[nombre] = output
                votos.append(voto)
        
        # === 5. INTEGRAR SALIDAS ===
        # Ponderación según modulación del Tálamo
        h_integrado = torch.zeros_like(x)
        for i, nombre in enumerate(self.ORDEN_MODULOS):
            peso = modulacion[:, i].view(-1, 1, 1)  # (batch, 1, 1)
            h_integrado = h_integrado + outputs[nombre] * peso
        
        # Normalizar por suma de pesos
        suma_pesos = modulacion.sum(dim=1, keepdim=True).unsqueeze(-1)  # (batch, 1, 1)
        h_integrado = h_integrado / (suma_pesos + 1e-6)
        
        # Proyección final
        h_final = self.norm_final(self.integrador(h_integrado))
        
        # === 6. HIPOCAMPO ===
        if self.hipocampo is not None:
            h_final = self.hipocampo(h_final)
            
        # === 7. OUTPUT ===
        logits = self.output(h_final)
        
        # === 8. LOSS ===
        result = {'logits': logits, 'stats': stats}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
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
        """Genera texto."""
        self.eval()
        self.talamo.reset_estado()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = generated[:, -512:] if generated.shape[1] > 512 else generated
            
            # Forward
            result = self(context)
            logits = result['logits'][:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p
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
            
            # Stop en EOS
            if next_token.item() == 0:
                break
                
        return generated
