# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
PAMPAr-o1 v9 - Modelo de Lenguaje Cerebral con Territorios

Arquitectura cerebral que procesa tokens usando:
- TÁLAMO: Orquestador central con LLAVES (70% reglas + 30% aprendido)
- 4 TERRITORIOS: Agrupaciones funcionales de módulos
- 6 FRONTERAS: Conexiones bidireccionales entre territorios
- AXIOMAS: Razonamiento deductivo formal (opcional)
- MEMORIA: Experiencia acumulada (opcional)

Territorios:
- EXPRESIVO: Lenguaje + Creatividad
- CONTEXTUAL: Contexto (memoria de trabajo)
- FORMAL: Lógica
- ESTRUCTURAL: Patrones + Matemáticas

Flujo:
1. Token → Embedding
2. Tálamo decide pesos por territorio/módulo
3. Territorios procesan (módulos internos + buffer compartido)
4. Fronteras intercambian señales entre territorios activos
5. Axiomas aplican razonamiento (si habilitado)
6. LM Head genera siguiente token

En una frase:
"PampaR es un cerebro artificial donde el tálamo dirige tokens a territorios 
especializados que colaboran via fronteras bidireccionales, combinando 
reglas explícitas (LLAVES) con aprendizaje profundo para generar lenguaje."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Optional

from .talamo import TalamoTerritorial
from .territorio import GestorTerritorios, CONFIGURACION_TERRITORIOS
from .frontera import GestorFronteras
from .razonamiento.axiomas import MotorAxiomas
from .memoria.experiencia import MemoriaExperiencia

from pampar.config import ConfigPampaR


class BloqueTerrritorial(nn.Module):
    """
    Bloque de procesamiento territorial.
    
    Cada bloque:
    1. Tálamo decide activación de territorios
    2. Territorios procesan en paralelo
    3. Fronteras intercambian señales entre territorios activos
    4. Combina salidas pesadas por tálamo
    """
    
    def __init__(self, config: ConfigPampaR, block_idx: int = 0):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        
        # Tálamo que maneja territorios
        self.talamo = TalamoTerritorial(
            dim=config.dim,
            vocab_size=config.vocab_size,
            peso_llaves=config.peso_llaves,
        )
        
        # Gestor de territorios (contiene los 4 territorios)
        self.territorios = GestorTerritorios(
            dim=config.dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        # Gestor de fronteras (6 conexiones bidireccionales)
        self.fronteras = GestorFronteras(
            dim=config.dim,
            umbral_activacion=0.3,
        )
        
        # Normalización final del bloque
        self.norm = nn.LayerNorm(config.dim)
        
        # Proyección de combinación
        self.combinar = nn.Linear(config.dim, config.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Procesa entrada a través de territorios y fronteras.
        
        Args:
            x: (batch, seq, dim) embeddings
            token_ids: (batch, seq) IDs para LLAVES
            mask: máscara de atención
            
        Returns:
            output: (batch, seq, dim)
            info: dict con estadísticas
        """
        batch, seq, dim = x.shape
        residual = x
        
        # 1. Tálamo decide activación de territorios y módulos
        pesos_territorios, pesos_modulos = self.talamo(x, token_ids)
        
        # 2. Cada territorio procesa
        estados_territorios = {}
        salidas_territorios = {}
        
        for nombre_terr in self.territorios.nombres_territorios:
            territorio = self.territorios.obtener_territorio(nombre_terr)
            peso_terr = pesos_territorios[nombre_terr]
            
            # Si el territorio tiene peso bajo, procesamiento basal
            if peso_terr.max() < 0.2:
                salida = territorio.procesar_basal(x)
                estado = salida
            else:
                # Procesamiento completo
                salida, estado = territorio(x, pesos_modulos, mask)
            
            salidas_territorios[nombre_terr] = salida
            estados_territorios[nombre_terr] = estado
        
        # 3. Fronteras intercambian señales entre territorios activos
        senales_fronteras = self.fronteras.intercambiar(
            estados_territorios,
            pesos_territorios
        )
        
        # 4. Integrar señales de fronteras en salidas
        for nombre_terr, senal in senales_fronteras.items():
            if senal is not None:
                salidas_territorios[nombre_terr] = salidas_territorios[nombre_terr] + senal * 0.3
        
        # 5. Combinar salidas pesadas por tálamo
        output = torch.zeros_like(x)
        for nombre_terr in self.territorios.nombres_territorios:
            peso = pesos_territorios[nombre_terr]
            output = output + peso * salidas_territorios[nombre_terr]
        
        # 6. Residual + normalización
        output = self.norm(residual + self.combinar(output))
        
        # Info para debug (vacío durante training)
        info = {}
        
        return output, info


class PampaR(nn.Module):
    """
    PAMPAr-o1 v9 - Modelo de Lenguaje con Arquitectura Cerebral Territorial
    
    Innovaciones:
    - Territorios: Agrupaciones funcionales de módulos que comparten buffer
    - Fronteras: Conexiones bidireccionales entre territorios (6 vs 18 sinapsis)
    - LLAVES: Routing híbrido (70% reglas explícitas + 30% atención aprendida)
    - Axiomas: Razonamiento deductivo diferenciable (modus ponens, silogismo)
    """
    
    def __init__(self, config: ConfigPampaR):
        super().__init__()
        self.config = config
        
        # ============================================
        # Embeddings
        # ============================================
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # ============================================
        # Bloques Territoriales
        # ============================================
        self.bloques = nn.ModuleList([
            BloqueTerrritorial(config, block_idx=i)
            for i in range(config.n_capas)
        ])
        
        # Gradient checkpointing flag
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # ============================================
        # Razonamiento Axiomático (opcional)
        # ============================================
        if config.usar_axiomas:
            self.motor_axiomas = MotorAxiomas(config.dim)
        else:
            self.motor_axiomas = None
        
        # ============================================
        # Memoria de Experiencia (opcional)
        # ============================================
        if config.usar_memoria:
            self.memoria = MemoriaExperiencia(
                dim=config.dim,
                capacidad=config.capacidad_memoria,
            )
        else:
            self.memoria = None
        
        # ============================================
        # LM Head
        # ============================================
        self.norm_final = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights (embedding y lm_head comparten pesos)
        self.lm_head.weight = self.token_embed.weight
        
        # Inicialización
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicialización de pesos estándar."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def registrar_tokenizer(self, tokenizer):
        """Registra el tokenizer para las LLAVES del tálamo."""
        for bloque in self.bloques:
            bloque.talamo.registrar_tokenizer(tokenizer, self.config.vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Dict:
        """
        Forward pass del modelo.
        
        Args:
            input_ids: (batch, seq) IDs de tokens
            labels: (batch, seq) labels para loss
            return_info: si retornar info de debug
            
        Returns:
            dict con 'logits', 'loss' (si labels), 'info' (si return_info)
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        
        # 1. Embeddings
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)
        
        # Máscara causal
        mask = torch.triu(
            torch.ones(seq, seq, device=device),
            diagonal=1
        ).bool()
        
        # 2. Bloques Territoriales
        info_bloques = []
        for bloque in self.bloques:
            if self.use_gradient_checkpointing and self.training:
                x, info = checkpoint(
                    bloque, x, input_ids, mask,
                    use_reentrant=False
                )
            else:
                x, info = bloque(x, input_ids, mask)
            info_bloques.append(info)
        
        # 3. Razonamiento Axiomático
        stats_axiomas = {}
        if self.motor_axiomas is not None:
            x, stats_axiomas = self.motor_axiomas(x, mask)
        
        # 4. Memoria de Experiencia
        if self.memoria is not None:
            x = self.memoria(x)
        
        # 5. LM Head
        x = self.norm_final(x)
        logits = self.lm_head(x)
        
        # 6. Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
        
        # Resultado
        result = {'logits': logits}
        if loss is not None:
            result['loss'] = loss
        if return_info:
            result['info'] = {
                'bloques': info_bloques,
                'axiomas': stats_axiomas,
            }
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Genera tokens autoregressivamente.
        """
        self.eval()
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Truncar si excede max_seq_len
            if generated.shape[1] >= self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated
            
            # Forward
            outputs = self(context)
            logits = outputs['logits'][:, -1, :]
            
            # Repetition penalty
            for i in range(generated.shape[0]):
                for token_id in set(generated[i].tolist()):
                    logits[i, token_id] /= repetition_penalty
            
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
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si EOS
            if (next_token == 3).all():
                break
        
        return generated
    
    def contar_parametros(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        counts = {
            'embeddings': 0,
            'bloques_territoriales': 0,
            'axiomas': 0,
            'memoria': 0,
            'lm_head': 0,
        }
        
        for name, param in self.named_parameters():
            n = param.numel()
            if 'embed' in name:
                counts['embeddings'] += n
            elif 'bloque' in name:
                counts['bloques_territoriales'] += n
            elif 'axioma' in name:
                counts['axiomas'] += n
            elif 'memoria' in name:
                counts['memoria'] += n
            elif 'lm_head' in name:
                counts['lm_head'] += n
            else:
                counts['bloques_territoriales'] += n
        
        counts['total'] = sum(counts.values())
        return counts
    
    def estado_fronteras(self) -> Dict[str, Dict[str, float]]:
        """Obtiene el estado de las fronteras de cada bloque."""
        return {
            f'bloque_{i}': bloque.fronteras.estado_fronteras()
            for i, bloque in enumerate(self.bloques)
        }
