# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
Módulo de Procesamiento Multiescala para LLARRI-O1.

Implementa la tokenización y procesamiento en múltiples resoluciones
simultáneas, similar a wavelets o FPN.

256 bytes → [Nivel 2: 256 tok] + [Nivel 4: 64 tok] + [Nivel 8: 16 tok] + [Nivel 16: 4 tok]
          → Fusión multiescala → Bloque Fractal → Output

Esto permite:
- Ver patrones a TODAS las escalas simultáneamente
- Economizar recursos con embeddings compartidos
- Mejor generalización por representación jerárquica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MultiescalaConfig:
    """Configuración del procesador multiescala."""
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    embed_dim: int = 64
    vocab_size: int = 256  # Bytes
    max_length: int = 256
    fusion_type: str = "concat_project"  # "concat_project", "attention", "sum"
    

class TokenizadorMultiescala(nn.Module):
    """
    Tokeniza entrada en múltiples niveles simultáneamente.
    
    Nivel 2: cada byte = 1 token
    Nivel 4: cada 4 bytes = 1 token (promedio/pool)
    Nivel 8: cada 8 bytes = 1 token
    etc.
    """
    
    def __init__(self, config: MultiescalaConfig):
        super().__init__()
        self.config = config
        self.niveles = config.niveles
        
        # Pooling para comprimir niveles superiores
        self.pools = nn.ModuleDict()
        for nivel in self.niveles:
            if nivel > 2:
                # Pool de tamaño nivel/2 para comprimir
                pool_size = nivel // 2
                self.pools[str(nivel)] = nn.AvgPool1d(
                    kernel_size=pool_size, 
                    stride=pool_size,
                    ceil_mode=True
                )
    
    def forward(self, bytes_input: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Tokeniza bytes en múltiples niveles.
        
        Args:
            bytes_input: (batch, seq_len) tensor de bytes [0-255]
            
        Returns:
            Dict[nivel, tokens] donde cada tokens tiene diferente longitud
        """
        batch_size, seq_len = bytes_input.shape
        tokens_por_nivel = {}
        
        # Nivel 2: directo (cada byte es un token)
        tokens_por_nivel[2] = bytes_input
        
        # Niveles superiores: comprimir secuencialmente
        for nivel in self.niveles:
            if nivel > 2:
                # Pool sobre la dimensión de secuencia
                # Necesitamos pasar por float para pooling
                prev_nivel = nivel // 2
                prev_tokens = tokens_por_nivel[prev_nivel].float()
                
                # (batch, seq) -> (batch, 1, seq) para pool1d
                prev_tokens = prev_tokens.unsqueeze(1)
                pooled = self.pools[str(nivel)](prev_tokens)
                
                # (batch, 1, new_seq) -> (batch, new_seq)
                tokens_por_nivel[nivel] = pooled.squeeze(1).long().clamp(0, 255)
        
        return tokens_por_nivel


class EmbeddingMultiescala(nn.Module):
    """
    Embeddings COMPARTIDOS entre niveles para economía.
    
    Todos los niveles usan el mismo embedding base de 256 tokens,
    pero proyectan a representaciones específicas por nivel.
    """
    
    def __init__(self, config: MultiescalaConfig):
        super().__init__()
        self.config = config
        
        # Embedding base COMPARTIDO (256 bytes)
        self.emb_base = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Proyecciones por nivel (ligeras, no duplican params)
        self.proyecciones = nn.ModuleDict()
        for nivel in config.niveles:
            self.proyecciones[str(nivel)] = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.GELU()
            )
        
        # Positional embeddings por nivel
        self.pos_embs = nn.ModuleDict()
        for nivel in config.niveles:
            max_len_nivel = config.max_length // (nivel // 2) if nivel > 2 else config.max_length
            self.pos_embs[str(nivel)] = nn.Embedding(max_len_nivel + 1, config.embed_dim)
    
    def forward(
        self, 
        tokens_por_nivel: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Genera embeddings para cada nivel.
        
        Args:
            tokens_por_nivel: Dict[nivel, (batch, seq_nivel)]
            
        Returns:
            Dict[nivel, (batch, seq_nivel, embed_dim)]
        """
        embeddings = {}
        
        for nivel, tokens in tokens_por_nivel.items():
            batch_size, seq_len = tokens.shape
            
            # Embedding base compartido
            emb = self.emb_base(tokens.clamp(0, 255))
            
            # Proyección específica del nivel
            emb = self.proyecciones[str(nivel)](emb)
            
            # Positional embedding
            positions = torch.arange(seq_len, device=tokens.device)
            pos_emb = self.pos_embs[str(nivel)](positions)
            
            # Sumar posicional
            embeddings[nivel] = emb + pos_emb.unsqueeze(0)
        
        return embeddings


class FusionMultiescala(nn.Module):
    """
    Fusiona representaciones de múltiples niveles.
    
    Estrategias:
    - concat_project: Concatena y proyecta a dim fijo
    - attention: Cross-attention entre niveles
    - sum: Suma ponderada (más simple)
    """
    
    def __init__(self, config: MultiescalaConfig):
        super().__init__()
        self.config = config
        self.fusion_type = config.fusion_type
        
        num_niveles = len(config.niveles)
        
        if self.fusion_type == "concat_project":
            # Concatenamos todos los niveles (upsampled al nivel 2)
            # y proyectamos
            self.fusion_proj = nn.Sequential(
                nn.Linear(config.embed_dim * num_niveles, config.embed_dim * 2),
                nn.GELU(),
                nn.Linear(config.embed_dim * 2, config.embed_dim),
                nn.LayerNorm(config.embed_dim)
            )
        
        elif self.fusion_type == "attention":
            # Cross-attention entre niveles
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=4,
                batch_first=True
            )
            self.norm = nn.LayerNorm(config.embed_dim)
        
        elif self.fusion_type == "sum":
            # Pesos aprendibles para cada nivel
            self.level_weights = nn.Parameter(torch.ones(num_niveles) / num_niveles)
        
        # Upsampling para niveles comprimidos
        self.upsamplers = nn.ModuleDict()
        for nivel in config.niveles:
            if nivel > 2:
                factor = nivel // 2
                self.upsamplers[str(nivel)] = nn.Upsample(
                    scale_factor=factor, 
                    mode='nearest'
                )
    
    def _upsample_to_base(
        self, 
        embeddings: Dict[int, torch.Tensor],
        target_len: int
    ) -> List[torch.Tensor]:
        """Upsamplea todos los niveles al tamaño del nivel 2."""
        upsampled = []
        
        for nivel in sorted(embeddings.keys()):
            emb = embeddings[nivel]  # (batch, seq_nivel, dim)
            
            if nivel == 2:
                upsampled.append(emb)
            else:
                # Transponer para upsample: (batch, dim, seq)
                emb_t = emb.transpose(1, 2)
                
                # Upsample
                up = self.upsamplers[str(nivel)](emb_t)
                
                # Ajustar tamaño exacto
                if up.shape[2] != target_len:
                    up = F.interpolate(up, size=target_len, mode='nearest')
                
                # Transponer de vuelta: (batch, seq, dim)
                upsampled.append(up.transpose(1, 2))
        
        return upsampled
    
    def forward(
        self, 
        embeddings: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fusiona embeddings multiescala.
        
        Args:
            embeddings: Dict[nivel, (batch, seq_nivel, embed_dim)]
            
        Returns:
            fused: (batch, seq_base, embed_dim) representación fusionada
        """
        # Obtener tamaño base (nivel 2)
        base_len = embeddings[2].shape[1]
        
        # Upsamplear todos al tamaño base
        upsampled = self._upsample_to_base(embeddings, base_len)
        
        if self.fusion_type == "concat_project":
            # Concatenar en dimensión de features
            concat = torch.cat(upsampled, dim=-1)  # (batch, seq, dim * num_niveles)
            fused = self.fusion_proj(concat)
        
        elif self.fusion_type == "attention":
            # El nivel 2 (detalle) es query, otros son key/value
            query = upsampled[0]
            kv = torch.stack(upsampled[1:], dim=2).mean(dim=2)  # Promedio de otros niveles
            
            attn_out, _ = self.cross_attn(query, kv, kv)
            fused = self.norm(query + attn_out)
        
        elif self.fusion_type == "sum":
            # Suma ponderada
            weights = F.softmax(self.level_weights, dim=0)
            stacked = torch.stack(upsampled, dim=0)  # (num_niveles, batch, seq, dim)
            fused = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        return fused


class ProcesadorMultiescala(nn.Module):
    """
    Procesador completo multiescala.
    
    Flujo:
    1. Tokeniza en múltiples niveles
    2. Genera embeddings (compartidos)
    3. Fusiona representaciones
    4. Output listo para Bloque Fractal
    """
    
    def __init__(self, config: MultiescalaConfig = None):
        super().__init__()
        self.config = config or MultiescalaConfig()
        
        self.tokenizador = TokenizadorMultiescala(self.config)
        self.embedding = EmbeddingMultiescala(self.config)
        self.fusion = FusionMultiescala(self.config)
        
        print(f"✓ ProcesadorMultiescala: niveles={self.config.niveles}")
        print(f"  Fusión: {self.config.fusion_type}")
        print(f"  Embed compartido: {self.config.vocab_size}→{self.config.embed_dim}")
    
    def forward(
        self, 
        bytes_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Procesa bytes en representación multiescala fusionada.
        
        Args:
            bytes_input: (batch, seq_len) tensor de bytes [0-255]
            
        Returns:
            output: (batch, seq_len, embed_dim) representación fusionada
            info: estadísticas de cada nivel
        """
        # 1. Tokenizar en múltiples niveles
        tokens_por_nivel = self.tokenizador(bytes_input)
        
        # 2. Embeddings por nivel (compartidos)
        embeddings = self.embedding(tokens_por_nivel)
        
        # 3. Fusionar
        fused = self.fusion(embeddings)
        
        # Info para debugging
        info = {
            'tokens_por_nivel': {n: t.shape[1] for n, t in tokens_por_nivel.items()},
            'embeddings_shapes': {n: e.shape for n, e in embeddings.items()}
        }
        
        return fused, info
    
    def get_memoria_estimada(self) -> Dict[str, float]:
        """Estima uso de memoria por nivel."""
        memoria = {}
        
        # Embedding base (compartido)
        params_emb = self.config.vocab_size * self.config.embed_dim
        memoria['embedding_compartido'] = params_emb * 4 / 1024**2  # MB
        
        # Por nivel
        for nivel in self.config.niveles:
            params_proy = self.config.embed_dim * self.config.embed_dim * 2  # Linear + LN
            max_len = self.config.max_length // (nivel // 2) if nivel > 2 else self.config.max_length
            params_pos = max_len * self.config.embed_dim
            memoria[f'nivel_{nivel}'] = (params_proy + params_pos) * 4 / 1024**2
        
        memoria['total'] = sum(memoria.values())
        return memoria


# Test rápido
if __name__ == "__main__":
    config = MultiescalaConfig(
        niveles=[2, 4, 8, 16],
        embed_dim=64,
        max_length=128
    )
    
    procesador = ProcesadorMultiescala(config)
    
    # Simular entrada
    batch = torch.randint(0, 256, (2, 128))
    
    output, info = procesador(batch)
    
    print(f"\nEntrada: {batch.shape}")
    print(f"Tokens por nivel: {info['tokens_por_nivel']}")
    print(f"Output fusionado: {output.shape}")
    print(f"\nMemoria estimada: {procesador.get_memoria_estimada()}")
