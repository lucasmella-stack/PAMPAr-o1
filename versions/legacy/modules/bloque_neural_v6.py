# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Bloque Neural Multi-Escala (9 cajas)

Un bloque de 9 cajas neurales que procesan TODAS las escalas 2→256.
Estructura interna fractal con gates por escala.

Concepto: "Procesar información a todas las escalas simultáneamente"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class AttentionMultiEscala(nn.Module):
    """
    CAJA NEURAL: Attention que opera en todas las escalas.
    
    Usa los gates de la puerta para ponderar qué escalas importan más.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [seq_len, seq_len] causal mask
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.shape
        
        # Proyecciones Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Aplicar máscara causal
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aplicar a valores
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        
        return self.out_proj(out)


class FFNEscalaAware(nn.Module):
    """
    CAJA NEURAL: FFN que procesa con conciencia de escala.
    
    Tiene sub-redes para diferentes rangos de escala.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        expansion: float = 2.0,
        n_escalas: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = int(embed_dim * expansion)
        self.n_escalas = n_escalas
        
        # Red principal
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        
        # Modulación por escala (lightweight)
        self.scale_modulation = nn.Linear(n_escalas, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            gates: [batch, n_escalas] - pesos por escala
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        # FFN principal
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        out = self.fc2(h)
        
        # Modular por escala si hay gates
        if gates is not None:
            modulation = self.scale_modulation(gates)  # [batch, embed_dim]
            modulation = modulation.unsqueeze(1)  # [batch, 1, embed_dim]
            out = out * (1 + 0.1 * modulation)  # Modulación suave
        
        return self.dropout(out)


class CombinadorEscalas(nn.Module):
    """
    CAJA NEURAL: Combina información de diferentes escalas.
    """
    
    def __init__(self, embed_dim: int = 128, n_escalas: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Proyección para combinar
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Gate learnable para balance
        self.balance_gate = nn.Linear(n_escalas, 1)
        
    def forward(self, x: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            gates: [batch, n_escalas]
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        combined = self.combiner(x)
        
        if gates is not None:
            # Usar gates para ajustar balance
            balance = torch.sigmoid(self.balance_gate(gates))  # [batch, 1]
            balance = balance.unsqueeze(1)  # [batch, 1, 1]
            combined = balance * combined + (1 - balance) * x
        
        return combined


class GateEscala(nn.Module):
    """
    CAJA NEURAL: Decide importancia de cada escala para el output.
    """
    
    def __init__(self, embed_dim: int = 128, n_escalas: int = 8):
        super().__init__()
        
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, n_escalas),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            gates_refined: [batch, n_escalas]
        """
        # Promediar sobre secuencia
        x_pooled = x.mean(dim=1)  # [batch, embed_dim]
        return self.gate_proj(x_pooled)


class BloqueNeuralV6(nn.Module):
    """
    Bloque Neural completo de 9 cajas para LLARRI v6.
    
    Estructura:
    - Caja 1: Attention multi-escala
    - Caja 2-4: FFN pequeño, medio, grande
    - Caja 5-6: Combinadores (escalas bajas y altas)
    - Caja 7: Fusión
    - Caja 8: Gate refinado
    - Caja 9: Output con residual
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_escalas: int = 8,
        dropout: float = 0.1,
        nombre: str = "BloqueNeural"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_escalas = n_escalas
        self.nombre = nombre
        
        # Caja 1: Attention
        self.caja1_attention = AttentionMultiEscala(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Caja 2: FFN pequeño (0.5x)
        self.caja2_ffn_small = FFNEscalaAware(embed_dim, 0.5, n_escalas, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Caja 3: FFN medio (1.0x)
        self.caja3_ffn_med = FFNEscalaAware(embed_dim, 1.0, n_escalas, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Caja 4: FFN grande (2.0x)
        self.caja4_ffn_large = FFNEscalaAware(embed_dim, 2.0, n_escalas, dropout)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # Caja 5: Combinador escalas bajas (2,4,8,16)
        self.caja5_combine_low = CombinadorEscalas(embed_dim, n_escalas)
        self.norm5 = nn.LayerNorm(embed_dim)
        
        # Caja 6: Combinador escalas altas (32,64,128,256)
        self.caja6_combine_high = CombinadorEscalas(embed_dim, n_escalas)
        self.norm6 = nn.LayerNorm(embed_dim)
        
        # Caja 7: Fusión de bajo y alto
        self.caja7_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm7 = nn.LayerNorm(embed_dim)
        
        # Caja 8: Gate refinado
        self.caja8_gate = GateEscala(embed_dim, n_escalas)
        
        # Caja 9: Output
        self.caja9_output = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"✓ {nombre}: 9 cajas, {embed_dim}d, {n_heads} heads, {n_escalas} escalas")
        
    def forward(
        self,
        x: torch.Tensor,
        gates: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            gates: [batch, n_escalas] de la puerta
            mask: [seq_len, seq_len] causal mask
        Returns:
            output: [batch, seq_len, embed_dim]
            gates_refined: [batch, n_escalas]
        """
        residual = x
        
        # Caja 1: Attention
        h = self.caja1_attention(self.norm1(x), mask)
        h = self.dropout(h) + residual
        
        # Caja 2: FFN pequeño
        h2 = self.caja2_ffn_small(self.norm2(h), gates)
        h2 = h2 + h
        
        # Caja 3: FFN medio
        h3 = self.caja3_ffn_med(self.norm3(h2), gates)
        h3 = h3 + h2
        
        # Caja 4: FFN grande
        h4 = self.caja4_ffn_large(self.norm4(h3), gates)
        h4 = h4 + h3
        
        # Separar para escalas bajas y altas
        gates_low = gates[:, :4]  # Escalas 2,4,8,16
        gates_high = gates[:, 4:]  # Escalas 32,64,128,256
        
        # Caja 5: Combinar escalas bajas
        h_low = self.caja5_combine_low(self.norm5(h4), gates)
        
        # Caja 6: Combinar escalas altas
        h_high = self.caja6_combine_high(self.norm6(h4), gates)
        
        # Caja 7: Fusión
        h_concat = torch.cat([h_low, h_high], dim=-1)  # [batch, seq, embed*2]
        h_fused = self.caja7_fusion(h_concat)
        h_fused = self.norm7(h_fused + h4)  # Residual desde h4
        
        # Caja 8: Gate refinado
        gates_refined = self.caja8_gate(h_fused)
        
        # Caja 9: Output
        output = self.caja9_output(h_fused)
        output = self.norm_out(output + residual)  # Residual desde input original
        
        return output, gates_refined


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Test BloqueNeuralV6")
    print("=" * 60)
    
    bloque = BloqueNeuralV6(
        embed_dim=128,
        n_heads=4,
        n_escalas=8,
        nombre="Bloque1"
    )
    
    # Input de prueba
    batch, seq_len, embed_dim = 4, 64, 128
    x = torch.randn(batch, seq_len, embed_dim)
    gates = F.softmax(torch.randn(batch, 8), dim=-1)
    
    # Máscara causal
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    output, gates_refined = bloque(x, gates, mask)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Gates shape: {gates.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Gates refined shape: {gates_refined.shape}")
    
    # Verificar parámetros
    params = sum(p.numel() for p in bloque.parameters())
    print(f"\nParámetros: {params:,}")
    
    print("\n✅ BloqueNeuralV6 funcionando!")
