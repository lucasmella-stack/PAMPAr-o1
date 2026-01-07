# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
LLARRI Language Model v3 - Arquitectura Multiescala.

Procesamiento simultáneo en múltiples resoluciones:
- Nivel 2: caracteres (alta frecuencia, detalles)
- Nivel 4: bigramas (patrones)
- Nivel 8: cuadrantes (estructura)
- Nivel 16: contexto (semántica)

La misma entrada de 256 bytes se ve a TODAS las escalas,
con embeddings COMPARTIDOS para economía de recursos.
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


@dataclass 
class LLARRIv3Config:
    """Configuración LLARRI v3 Multiescala."""
    
    # Dimensiones
    embed_dim: int = 64
    
    # Niveles (multiescala)
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    fusion_type: str = "concat_project"  # concat_project, attention, sum
    
    # Bloque Fractal
    num_heads: int = 4
    ffn_expansion: float = 2.0
    num_vecinos: int = 3
    umbral_confianza: float = 0.7
    
    # General
    dropout: float = 0.1
    vocab_size: int = 256
    max_length: int = 256
    
    # Generación
    temperatura_default: float = 0.8


class LLARRIv3(nn.Module):
    """
    LLARRI Language Model v3 - Multiescala.
    
    Arquitectura:
    ```
    256 bytes entrada
          │
    ┌─────┼─────┐─────┐
    │     │     │     │
    ▼     ▼     ▼     ▼
   N2    N4    N8   N16   (tokenización paralela)
   256   64    16    4    tokens
    │     │     │     │
    ▼     ▼     ▼     ▼
   Embed (COMPARTIDO)     (economía)
    │     │     │     │
    └──┬──┴──┬──┴──┬──┘
       │     │     │
       ▼     ▼     ▼
      FUSION MULTIESCALA
              │
              ▼
      BLOQUE FRACTAL (6 cajas)
              │
              ▼
         LM HEAD → 256 vocab
    ```
    """
    
    def __init__(self, config: LLARRIv3Config = None):
        super().__init__()
        self.config = config or LLARRIv3Config()
        
        print("=" * 60)
        print("INICIALIZANDO LLARRI v3 - MULTIESCALA")
        print("=" * 60)
        
        # 1. Procesador Multiescala (tokeniza + embeds + fusiona)
        multiescala_config = MultiescalaConfig(
            niveles=self.config.niveles,
            embed_dim=self.config.embed_dim,
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_length,
            fusion_type=self.config.fusion_type
        )
        self.multiescala = ProcesadorMultiescala(multiescala_config)
        
        # 2. Bloque Fractal (6 cajas)
        bloque_config = BloqueFractalConfig(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            ffn_expansion=self.config.ffn_expansion,
            num_vecinos=self.config.num_vecinos,
            umbral_confianza=self.config.umbral_confianza,
            niveles=self.config.niveles
        )
        self.bloque_fractal = BloqueFractal(bloque_config)
        print(f"✓ BloqueFractal: 6 cajas, {self.config.num_vecinos} vecinos")
        
        # 3. LM Head (simple, 256 vocab)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.vocab_size)
        )
        print(f"✓ LMHead: {self.config.embed_dim} → {self.config.vocab_size}")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Total parámetros: {total_params:,} ({total_params/1e6:.2f}M)")
        print("=" * 60)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass multiescala.
        
        Args:
            input_ids: (batch, seq_len) bytes [0-255]
            labels: (batch, seq_len) para loss
            
        Returns:
            Dict con logits, loss (si labels), info multiescala
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Procesar multiescala (tokeniza + embeds + fusiona)
        fused, multiescala_info = self.multiescala(input_ids)
        
        # 2. Bloque Fractal (6 cajas)
        hidden, bloque_info = self.bloque_fractal(fused)
        
        # 3. LM Head
        logits = self.lm_head(hidden)
        
        output = {
            'logits': logits,
            'hidden_states': hidden,
            'multiescala_info': multiescala_info,
            'bloque_info': bloque_info
        }
        
        # Loss si hay labels
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperatura: float = None,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> str:
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt: texto inicial
            max_new_tokens: máximo de tokens a generar
            temperatura: para sampling (None = usar default)
            top_k: filtrar a top k tokens
            top_p: nucleus sampling
            
        Returns:
            texto generado
        """
        temperatura = temperatura or self.config.temperatura_default
        device = next(self.parameters()).device
        self.eval()
        
        # Convertir prompt a bytes
        tokens = list(prompt.encode('utf-8', errors='ignore'))
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = tokens[-self.config.max_length:]
            
            # Forward
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            output = self.forward(input_ids)
            
            # Logits del último token
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
            if next_token in [0, 10, 13]:  # NULL, \n, \r
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
    print("TEST LLARRI v3 MULTIESCALA")
    print("=" * 60 + "\n")
    
    config = LLARRIv3Config(
        embed_dim=64,
        niveles=[2, 4, 8, 16],
        max_length=128
    )
    
    model = LLARRIv3(config)
    
    # Test forward
    batch = torch.randint(0, 256, (2, 64))
    labels = batch.clone()
    
    output = model(batch, labels=labels)
    
    print(f"\nTest forward:")
    print(f"  Input: {batch.shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Tokens por nivel: {output['multiescala_info']['tokens_por_nivel']}")
    
    # Test generación (CPU)
    print(f"\nTest generación:")
    text = model.generate("Hello", max_new_tokens=20)
    print(f"  'Hello' → '{text[:50]}...'")
