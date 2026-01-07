# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
LLARRI Language Model v4 - Arquitectura de 9 Cajas.

Innovación principal: Separación de PROCESAMIENTO y COMPOSICIÓN.

Cajas 1-6: Procesamiento Fractal (percepción)
    - Análisis multiescala
    - Extracción de features
    - Patrones a diferentes resoluciones

Cajas 7-9: Composición (cognición)
    - Caja 7: Detector de patrones/contexto
    - Caja 8: Planificador de siguiente token
    - Caja 9: Refinador de probabilidades

Es como un cerebro con áreas de procesamiento sensorial
Y áreas de planificación/decisión.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from llarri_o1.modules.multiescala import (
    ProcesadorMultiescala,
    MultiescalaConfig
)
from llarri_o1.modules.bloque_fractal import (
    BloqueFractal,
    BloqueFractalConfig,
)
from llarri_o1.modules.compositor import (
    ModuloCompositor,
    CompositorConfig
)


@dataclass 
class LLARRIv4Config:
    """Configuración LLARRI v4 - 9 Cajas."""
    
    # Dimensiones
    embed_dim: int = 128
    vocab_size: int = 256
    
    # Niveles multiescala
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    fusion_type: str = "concat_project"
    
    # Bloque Fractal (cajas 1-6)
    num_heads: int = 4
    ffn_expansion: float = 2.0
    num_vecinos: int = 3
    umbral_confianza: float = 0.7
    
    # Compositor (cajas 7-9)
    compositor_heads: int = 4
    compositor_dropout: float = 0.1
    pattern_window: int = 16
    
    # Generación
    max_length: int = 256
    temperatura_default: float = 0.8


class LLARRIv4(nn.Module):
    """
    LLARRI v4 - Modelo de Lenguaje con 9 Cajas.
    
    Arquitectura:
        Input (bytes)
            ↓
        Multiescala (niveles 2,4,8,16)
            ↓
        Cajas 1-6: Bloque Fractal (procesamiento)
            ↓
        Cajas 7-9: Compositor (razonamiento)
            ↓
        LM Head + Logit Bias
            ↓
        Output (probabilidades)
    """
    
    def __init__(self, config: LLARRIv4Config):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("INICIALIZANDO LLARRI v4 - 9 CAJAS")
        print("=" * 60)
        
        # 1. Procesador Multiescala
        multiescala_config = MultiescalaConfig(
            niveles=config.niveles,
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            fusion_type=config.fusion_type
        )
        self.multiescala = ProcesadorMultiescala(multiescala_config)
        
        # 2. Bloque Fractal (Cajas 1-6)
        fractal_config = BloqueFractalConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_expansion=config.ffn_expansion,
            num_vecinos=config.num_vecinos,
            umbral_confianza=config.umbral_confianza
        )
        self.bloque_fractal = BloqueFractal(fractal_config)
        print(f"✓ Cajas 1-6: Bloque Fractal")
        
        # 3. Compositor (Cajas 7-9) - NUEVO
        compositor_config = CompositorConfig(
            embed_dim=config.embed_dim,
            num_heads=config.compositor_heads,
            dropout=config.compositor_dropout,
            vocab_size=config.vocab_size,
            pattern_window=config.pattern_window
        )
        self.compositor = ModuloCompositor(compositor_config)
        print(f"✓ Cajas 7-9: Compositor")
        
        # 4. LM Head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        print(f"✓ LMHead: {config.embed_dim} → {config.vocab_size}")
        
        # Contar parámetros
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Total parámetros: {n_params:,} ({n_params/1e6:.2f}M)")
        print("=" * 60)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo.
        
        Args:
            input_ids: [batch, seq] tokens de entrada (bytes 0-255)
            labels: [batch, seq] tokens objetivo para loss
            
        Returns:
            dict con logits, loss (si labels), y info de debug
        """
        # 1. Multiescala: tokenizar a múltiples niveles
        multi_out = self.multiescala(input_ids)
        x = multi_out['fused']  # [batch, seq, embed_dim]
        
        # 2. Bloque Fractal (Cajas 1-6): procesamiento
        fractal_out = self.bloque_fractal(x)
        x = fractal_out['output']
        
        # 3. Compositor (Cajas 7-9): razonamiento
        comp_out = self.compositor(x)
        x = comp_out['features']
        logit_bias = comp_out['logit_bias']  # [batch, seq, vocab]
        
        # 4. LM Head + bias del compositor
        logits = self.lm_head(x) + logit_bias
        
        # Loss si hay labels
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'multiescala_info': multi_out,
            'fractal_info': fractal_out,
            'compositor_info': {
                'detections': comp_out['detections'],
                'dominant_plan': comp_out['dominant_plan'],
                'refiner_weights': comp_out['refiner_weights'],
            }
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperatura: float = None,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> str:
        """
        Genera texto con las 9 cajas.
        
        El compositor (cajas 7-9) ajusta automáticamente las
        probabilidades para mejorar coherencia.
        """
        temperatura = temperatura or self.config.temperatura_default
        device = next(self.parameters()).device
        self.eval()
        
        # Convertir prompt a bytes
        tokens = list(prompt.encode('utf-8', errors='ignore'))
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = tokens[-self.config.max_length:]
            
            # Forward completo (incluye compositor)
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            output = self.forward(input_ids)
            
            # Logits del último token (ya tienen bias del compositor)
            logits = output['logits'][:, -1, :] / temperatura
            
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
            
            # Sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
            
            # Stop en newline o EOS
            if next_token in [0, 10, 13]:
                break
        
        # Decodificar
        try:
            return bytes(tokens).decode('utf-8', errors='ignore')
        except:
            return prompt + "[error decoding]"
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Cuenta parámetros."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.multiescala.embedding.emb_base.weight.numel()
        return n_params


# Test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST LLARRI v4 - 9 CAJAS")
    print("=" * 60 + "\n")
    
    config = LLARRIv4Config(
        embed_dim=128,
        niveles=[2, 4, 8, 16],
        max_length=128
    )
    
    model = LLARRIv4(config)
    
    # Test forward
    batch = torch.randint(0, 256, (2, 64))
    labels = batch.clone()
    
    output = model(batch, labels=labels)
    
    print(f"\nTest forward:")
    print(f"  Input: {batch.shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Info del compositor
    print(f"\nInfo Compositor:")
    print(f"  Dominant plan shape: {output['compositor_info']['dominant_plan'].shape}")
    print(f"  Refiner weights shape: {output['compositor_info']['refiner_weights'].shape}")
    
    # Test generación
    print(f"\nTest generación (sin entrenar):")
    text = model.generate("Hello", max_new_tokens=20)
    print(f"  'Hello' → '{text}'")
