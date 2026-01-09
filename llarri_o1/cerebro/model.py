# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v8 - Modelo Principal

Integra todos los componentes del cerebro:
- Tálamo: orquestador con LLAVES
- Módulos especializados: 6 neuronas
- Sinapsis: conexiones entre módulos
- Axiomas: razonamiento deductivo
- Memoria de experiencia: aprendizaje práctico

Arquitectura:
1. Embedding de tokens
2. Tálamo distribuye a módulos según LLAVES
3. Módulos procesan en paralelo
4. Sinapsis comunican entre módulos
5. Axiomas aplican razonamiento lógico
6. Memoria refina basado en experiencias pasadas
7. LM Head genera tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .talamo import Talamo
from .sinapsis import Sinapsis
from .modulos.especializados import (
    NeuronaLenguaje, NeuronaLogica, NeuronaMatematicas,
    NeuronaPatrones, NeuronaContexto, NeuronaCreatividad
)
from .razonamiento.axiomas import MotorAxiomas
from .memoria.experiencia import MemoriaExperiencia


@dataclass
class ConfigLLARRI:
    """Configuración del modelo LLARRI v8."""
    
    # Dimensiones
    vocab_size: int = 8000
    dim: int = 256
    n_heads: int = 4
    n_capas: int = 4
    
    # Regularización
    dropout: float = 0.1
    
    # Tálamo
    peso_llaves: float = 0.7  # 70% reglas, 30% aprendido
    
    # Razonamiento
    usar_axiomas: bool = True
    
    # Memoria
    usar_memoria: bool = True
    capacidad_memoria: int = 500
    
    # Generación
    max_seq_len: int = 512
    repetition_penalty: float = 1.2


class CerebralBlock(nn.Module):
    """
    Un bloque cerebral: procesa con todos los módulos + sinapsis.
    
    Cada bloque:
    1. Recibe entrada
    2. Tálamo decide distribución a módulos
    3. Módulos procesan en paralelo
    4. Sinapsis comunican entre módulos
    5. Combina salidas pesadas
    """
    
    def __init__(self, config: ConfigLLARRI):
        super().__init__()
        self.config = config
        
        # Nombres de módulos
        self.nombres_modulos = [
            'lenguaje', 'logica', 'matematicas',
            'patrones', 'contexto', 'creatividad'
        ]
        
        # Tálamo
        self.talamo = Talamo(
            dim=config.dim,
            n_modulos=6,
            nombres_modulos=self.nombres_modulos,
            peso_llaves=config.peso_llaves,
        )
        
        # Módulos especializados
        self.modulos = nn.ModuleDict({
            'lenguaje': NeuronaLenguaje(config.dim, config.n_heads, config.dropout),
            'logica': NeuronaLogica(config.dim, config.n_heads, config.dropout),
            'matematicas': NeuronaMatematicas(config.dim, config.n_heads, config.dropout),
            'patrones': NeuronaPatrones(config.dim, config.n_heads, config.dropout),
            'contexto': NeuronaContexto(config.dim, config.n_heads, config.dropout),
            'creatividad': NeuronaCreatividad(config.dim, config.n_heads, config.dropout),
        })
        
        # Sinapsis
        self.sinapsis = Sinapsis(config.dim, self.nombres_modulos)
        
        # Normalización final
        self.norm = nn.LayerNorm(config.dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Procesa entrada a través de los módulos cerebrales.
        
        Args:
            x: (batch, seq, dim) embeddings
            token_ids: (batch, seq) IDs de tokens para LLAVES
            mask: máscara de atención
            
        Returns:
            output: (batch, seq, dim) salida procesada
            info: diccionario con estadísticas
        """
        batch, seq, dim = x.shape
        
        # 1. Tálamo decide distribución
        pesos = self.talamo(x, token_ids)  # (batch, seq, n_modulos)
        
        # 2. Cada módulo procesa la entrada
        salidas_modulos = {}
        for i, nombre in enumerate(self.nombres_modulos):
            modulo = self.modulos[nombre]
            salida = modulo.procesar(x, mask)
            salidas_modulos[nombre] = salida
        
        # 3. Sinapsis: comunicación entre módulos
        for origen in self.nombres_modulos:
            for destino in self.sinapsis.obtener_conexiones_de(origen):
                senal = self.sinapsis.transmitir(
                    origen, destino, 
                    salidas_modulos[origen]
                )
                if senal is not None:
                    # La señal sináptica modifica la salida del destino
                    salidas_modulos[destino] = salidas_modulos[destino] + senal
        
        # 4. Combinar salidas pesadas por tálamo
        output = torch.zeros_like(x)
        for i, nombre in enumerate(self.nombres_modulos):
            peso = pesos[:, :, i:i+1]  # (batch, seq, 1)
            output = output + peso * salidas_modulos[nombre]
        
        # 5. Residual + normalización
        output = self.norm(x + output)
        
        # Info para análisis
        info = {
            'pesos_modulos': {
                nombre: pesos[:, :, i].mean().item()
                for i, nombre in enumerate(self.nombres_modulos)
            }
        }
        
        return output, info


class LLARRIv8(nn.Module):
    """
    LLARRI v8 - Modelo de Lenguaje Cerebral
    
    Características:
    - Arquitectura modular inspirada en neurociencia
    - Tálamo como orquestador con LLAVES (reglas explícitas)
    - Sinapsis para comunicación inter-módulo
    - Axiomas para razonamiento deductivo
    - Memoria de experiencia para aprendizaje práctico
    """
    
    def __init__(self, config: ConfigLLARRI):
        super().__init__()
        self.config = config
        
        # ============================================
        # Embeddings
        # ============================================
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # ============================================
        # Bloques Cerebrales
        # ============================================
        self.bloques = nn.ModuleList([
            CerebralBlock(config) for _ in range(config.n_capas)
        ])
        
        # ============================================
        # Razonamiento Axiomático
        # ============================================
        if config.usar_axiomas:
            self.motor_axiomas = MotorAxiomas(config.dim)
        else:
            self.motor_axiomas = None
        
        # ============================================
        # Memoria de Experiencia
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
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
        
        # Inicialización
        self.apply(self._init_weights)
        
        # Estadísticas
        self.stats_acumuladas: Dict[str, List] = {
            'pesos_modulos': [],
            'axiomas_aplicados': [],
        }
    
    def _init_weights(self, module):
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
            labels: (batch, seq) labels para calcular loss
            return_info: si retornar info de debug
            
        Returns:
            dict con 'logits', 'loss' (si labels), 'info' (si return_info)
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        
        # ============================================
        # 1. Embeddings
        # ============================================
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)
        
        # Máscara causal
        mask = torch.triu(
            torch.ones(seq, seq, device=device), 
            diagonal=1
        ).bool()
        
        # ============================================
        # 2. Bloques Cerebrales
        # ============================================
        info_bloques = []
        for bloque in self.bloques:
            x, info = bloque(x, input_ids, mask)
            info_bloques.append(info)
        
        # ============================================
        # 3. Razonamiento Axiomático
        # ============================================
        stats_axiomas = {}
        if self.motor_axiomas is not None:
            x, stats_axiomas = self.motor_axiomas(x, mask)
        
        # ============================================
        # 4. Memoria de Experiencia
        # ============================================
        if self.memoria is not None:
            x = self.memoria(x)
        
        # ============================================
        # 5. LM Head
        # ============================================
        x = self.norm_final(x)
        logits = self.lm_head(x)
        
        # ============================================
        # 6. Loss (si hay labels)
        # ============================================
        loss = None
        if labels is not None:
            # Shift para predecir siguiente token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
        
        # ============================================
        # Resultado
        # ============================================
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
        
        Args:
            input_ids: (batch, seq) tokens iniciales
            max_new_tokens: cuántos tokens generar
            temperature: temperatura para sampling
            top_k: top-k filtering
            top_p: nucleus sampling
            repetition_penalty: penalidad por repetición
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
            logits = outputs['logits'][:, -1, :]  # (batch, vocab)
            
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
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si todos generaron EOS (asumiendo EOS = 3)
            if (next_token == 3).all():
                break
        
        return generated
    
    def obtener_estadisticas_modulos(self) -> Dict[str, float]:
        """Obtiene estadísticas promedio de activación de módulos."""
        stats = {}
        for bloque in self.bloques:
            talamo_stats = bloque.talamo.obtener_estadisticas()
            for nombre, valor in talamo_stats.items():
                if nombre not in stats:
                    stats[nombre] = []
                stats[nombre].append(valor)
        
        return {
            nombre: sum(valores) / len(valores) if valores else 0
            for nombre, valores in stats.items()
        }
    
    def reset_estadisticas(self):
        """Reinicia estadísticas de módulos."""
        for bloque in self.bloques:
            bloque.talamo.reset_estadisticas()
    
    def contar_parametros(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        counts = {
            'embeddings': 0,
            'bloques': 0,
            'axiomas': 0,
            'memoria': 0,
            'lm_head': 0,
        }
        
        for name, param in self.named_parameters():
            n = param.numel()
            if 'embed' in name:
                counts['embeddings'] += n
            elif 'bloque' in name:
                counts['bloques'] += n
            elif 'axioma' in name:
                counts['axiomas'] += n
            elif 'memoria' in name:
                counts['memoria'] += n
            elif 'lm_head' in name:
                counts['lm_head'] += n
            else:
                counts['bloques'] += n  # Default
        
        counts['total'] = sum(counts.values())
        return counts
