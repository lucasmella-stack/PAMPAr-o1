# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas.mella@outlook.com)
"""
Bloque Fractal Distribuido para LLARRI-O1.

Implementa el flujo de 6 cajas:
- Caja 1: MEZCLA (Attention)
- Cajas 2,3,4: PROCESAN (FFN distribuido entre vecinos)
- Caja 5: EVALÚA (early exit check)
- Caja 6: OUTPUT (resultado)

El procesamiento va de menos a más:
- Primero cuadrantes cercanos
- Si no alcanza, expande a más lejanos
- Early exit cuando es suficiente
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BloqueFractalConfig:
    """Configuración para Bloque Fractal."""
    
    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    
    # FFN distribuido
    ffn_expansion: float = 2.0  # Expansión más modesta que 4x tradicional
    num_vecinos: int = 3  # Cajas 2,3,4 que procesan
    
    # Niveles fractales
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    
    # Early exit
    umbral_confianza: float = 0.7


class CajaMezcla(nn.Module):
    """
    Caja 1: MEZCLA (Attention local).
    
    Decide qué tokens son relevantes para cada posición.
    Usa attention causal (solo ve pasado).
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            causal: si True, máscara causal
            
        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: para análisis
        """
        batch, seq_len, _ = x.shape
        
        # Normalización pre-attention
        x_norm = self.norm(x)
        
        # Q, K, V
        Q = self.q_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Máscara causal
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Aplicar a V
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        # Residual
        output = x + out
        
        return output, attn_weights.mean(dim=1)  # Promedio sobre heads


class CajaProcesa(nn.Module):
    """
    Cajas 2,3,4: PROCESAN (FFN distribuido).
    
    Cada caja procesa una parte del "entendimiento".
    Van de menos a más complejidad.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        expansion: float = 2.0,
        nivel: int = 1,  # 1, 2 o 3 (cercanía)
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nivel = nivel
        
        # Expansión progresiva: vecino 1 expande menos, vecino 3 expande más
        # nivel 1: expansion * 0.5
        # nivel 2: expansion * 0.75
        # nivel 3: expansion * 1.0
        factor = 0.5 + (nivel - 1) * 0.25
        hidden_dim = int(embed_dim * expansion * factor)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Gate para controlar cuánto aporta esta caja
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            
        Returns:
            output: (batch, seq_len, embed_dim)
            contribucion: cuánto aportó esta caja (0-1)
        """
        # FFN
        ffn_out = self.ffn(x)
        
        # Gate: cuánto contribuye esta caja
        gate_val = self.gate(x.mean(dim=1, keepdim=True))  # (batch, 1, 1)
        gate_val = gate_val.expand_as(ffn_out)
        
        # Residual con gate
        output = x + gate_val * ffn_out
        
        # Contribución promedio
        contribucion = gate_val.mean().item()
        
        return output, contribucion


class CajaEvalua(nn.Module):
    """
    Caja 5: EVALÚA.
    
    Decide si el procesamiento es suficiente o necesita subir de nivel.
    """
    
    def __init__(self, embed_dim: int, umbral: float = 0.7):
        super().__init__()
        self.umbral = umbral
        
        # Red para estimar "confianza" del resultado
        self.evaluador = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[float, bool]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            
        Returns:
            confianza: float entre 0 y 1
            suficiente: bool, si supera umbral
        """
        # Evaluar sobre representación promedio
        x_pool = x.mean(dim=1)  # (batch, embed_dim)
        confianza = self.evaluador(x_pool).mean().item()
        
        suficiente = confianza >= self.umbral
        
        return confianza, suficiente


class CajaOutput(nn.Module):
    """
    Caja 6: OUTPUT.
    
    Prepara el resultado final para el LM Head.
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            
        Returns:
            output: (batch, seq_len, embed_dim) listo para LM Head
        """
        x = self.norm(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class BloqueFractal(nn.Module):
    """
    Bloque Fractal completo con 6 cajas.
    
    Flujo:
    1. Caja 1 (Mezcla): Attention - ¿qué es relevante?
    2. Caja 2 (Procesa 1): FFN cercano - primera transformación
    3. Caja 3 (Procesa 2): FFN medio - segunda transformación
    4. Caja 4 (Procesa 3): FFN lejano - tercera transformación
    5. Caja 5 (Evalúa): ¿Es suficiente?
    6. Caja 6 (Output): Resultado final
    
    Early exit: Si Caja 5 dice "suficiente" después de Caja 2 o 3,
                no ejecuta las cajas siguientes.
    """
    
    def __init__(self, config: BloqueFractalConfig = None):
        super().__init__()
        self.config = config or BloqueFractalConfig()
        
        # Caja 1: MEZCLA
        self.caja_mezcla = CajaMezcla(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        )
        
        # Cajas 2,3,4: PROCESAN (vecinos)
        self.cajas_procesan = nn.ModuleList([
            CajaProcesa(
                embed_dim=self.config.embed_dim,
                expansion=self.config.ffn_expansion,
                nivel=i+1,
                dropout=self.config.dropout
            )
            for i in range(self.config.num_vecinos)
        ])
        
        # Caja 5: EVALÚA
        self.caja_evalua = CajaEvalua(
            embed_dim=self.config.embed_dim,
            umbral=self.config.umbral_confianza
        )
        
        # Caja 6: OUTPUT
        self.caja_output = CajaOutput(
            embed_dim=self.config.embed_dim,
            dropout=self.config.dropout
        )
        
        # Stats
        self.stats = {
            'total_calls': 0,
            'early_exits': {1: 0, 2: 0, 3: 0},  # Después de qué caja procesa
            'full_passes': 0
        }
    
    def forward(
        self, 
        x: torch.Tensor,
        forzar_completo: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass por las 6 cajas.
        
        Args:
            x: (batch, seq_len, embed_dim)
            forzar_completo: si True, no hace early exit
            
        Returns:
            output: (batch, seq_len, embed_dim)
            info: estadísticas del paso
        """
        self.stats['total_calls'] += 1
        
        info = {
            'cajas_ejecutadas': [],
            'contribuciones': {},
            'confianzas': {},
            'early_exit': False,
            'caja_salida': None
        }
        
        # CAJA 1: MEZCLA
        x, attn_weights = self.caja_mezcla(x)
        info['cajas_ejecutadas'].append('mezcla')
        info['attn_weights'] = attn_weights
        
        # CAJAS 2,3,4: PROCESAN (con posible early exit)
        for i, caja in enumerate(self.cajas_procesan):
            x, contribucion = caja(x)
            info['cajas_ejecutadas'].append(f'procesa_{i+1}')
            info['contribuciones'][i+1] = contribucion
            
            # CAJA 5: EVALÚA (después de cada proceso)
            if not forzar_completo and i < len(self.cajas_procesan) - 1:
                confianza, suficiente = self.caja_evalua(x)
                info['confianzas'][i+1] = confianza
                
                if suficiente:
                    info['early_exit'] = True
                    info['caja_salida'] = i + 1
                    self.stats['early_exits'][i+1] += 1
                    
                    # CAJA 6: OUTPUT
                    output = self.caja_output(x)
                    info['cajas_ejecutadas'].append('output')
                    
                    return output, info
        
        # Pasó por todas las cajas
        self.stats['full_passes'] += 1
        info['caja_salida'] = self.config.num_vecinos
        
        # CAJA 6: OUTPUT
        output = self.caja_output(x)
        info['cajas_ejecutadas'].append('output')
        
        return output, info
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso."""
        total = self.stats['total_calls']
        if total == 0:
            return self.stats
        
        stats = dict(self.stats)
        stats['tasa_early_exit'] = sum(self.stats['early_exits'].values()) / total
        stats['tasa_full'] = self.stats['full_passes'] / total
        
        return stats
    
    def print_stats(self):
        """Imprime estadísticas."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("BLOQUE FRACTAL - ESTADÍSTICAS")
        print("="*50)
        print(f"Total llamadas: {stats['total_calls']}")
        print(f"Tasa early exit: {stats.get('tasa_early_exit', 0):.1%}")
        print(f"Tasa full pass: {stats.get('tasa_full', 0):.1%}")
        print(f"\nEarly exits por caja:")
        for caja, count in stats['early_exits'].items():
            pct = count / stats['total_calls'] * 100 if stats['total_calls'] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  Caja {caja+1}: {count:4d} ({pct:5.1f}%) {bar}")
        print("="*50)
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria por caja."""
        memoria = {
            'caja_mezcla': sum(p.numel() * p.element_size() for p in self.caja_mezcla.parameters()),
            'cajas_procesan': sum(
                sum(p.numel() * p.element_size() for p in caja.parameters())
                for caja in self.cajas_procesan
            ),
            'caja_evalua': sum(p.numel() * p.element_size() for p in self.caja_evalua.parameters()),
            'caja_output': sum(p.numel() * p.element_size() for p in self.caja_output.parameters()),
        }
        memoria['total'] = sum(memoria.values())
        return memoria
    
    def print_arquitectura(self):
        """Imprime detalles de la arquitectura."""
        print("\n" + "="*60)
        print("BLOQUE FRACTAL - 6 CAJAS")
        print("="*60)
        
        print("""
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CAJA 1: MEZCLA (Attention)                          │   │
│  │ "¿Qué tokens son relevantes?"                       │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CAJA 2: PROCESA (Vecino cercano)                    │   │
│  │ FFN pequeño - primera transformación                │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                    ┌─────┴─────┐                           │
│                    │ CAJA 5:   │                           │
│                    │ EVALÚA    │──► ¿Suficiente? → SALIR   │
│                    └─────┬─────┘                           │
│                          │ NO                               │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CAJA 3: PROCESA (Vecino medio)                      │   │
│  │ FFN medio - segunda transformación                  │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                    ┌─────┴─────┐                           │
│                    │ CAJA 5:   │                           │
│                    │ EVALÚA    │──► ¿Suficiente? → SALIR   │
│                    └─────┬─────┘                           │
│                          │ NO                               │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CAJA 4: PROCESA (Vecino lejano)                     │   │
│  │ FFN grande - tercera transformación                 │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CAJA 6: OUTPUT                                      │   │
│  │ Preparar resultado final                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")
        
        print(f"\nConfiguración:")
        print(f"  embed_dim: {self.config.embed_dim}")
        print(f"  num_heads: {self.config.num_heads}")
        print(f"  ffn_expansion: {self.config.ffn_expansion}x")
        print(f"  umbral_confianza: {self.config.umbral_confianza}")
        
        memoria = self.get_memoria_usada()
        print(f"\nMemoria por caja:")
        print(f"  Caja 1 (Mezcla): {memoria['caja_mezcla'] / 1024:.2f} KB")
        print(f"  Cajas 2-4 (Procesan): {memoria['cajas_procesan'] / 1024:.2f} KB")
        print(f"  Caja 5 (Evalúa): {memoria['caja_evalua'] / 1024:.2f} KB")
        print(f"  Caja 6 (Output): {memoria['caja_output'] / 1024:.2f} KB")
        print(f"  TOTAL: {memoria['total'] / 1024:.2f} KB")
        
        print("="*60)


class BloqueFractalMultinivel(nn.Module):
    """
    Múltiples bloques fractales, uno por nivel.
    
    Implementa el flujo completo:
    Nivel 2 → Nivel 4 → Nivel 8 → ...
    
    Con early exit tanto dentro de cada bloque como entre niveles.
    """
    
    def __init__(self, config: BloqueFractalConfig = None):
        super().__init__()
        self.config = config or BloqueFractalConfig()
        
        # Un bloque por nivel
        self.bloques = nn.ModuleDict()
        for nivel in self.config.niveles:
            self.bloques[str(nivel)] = BloqueFractal(self.config)
        
        # Evaluador global (entre niveles)
        self.evaluador_global = CajaEvalua(
            embed_dim=self.config.embed_dim,
            umbral=self.config.umbral_confianza
        )
        
        # Stats globales
        self.stats = {
            'total_calls': 0,
            'exits_por_nivel': {nivel: 0 for nivel in self.config.niveles}
        }
    
    def forward(
        self,
        x: torch.Tensor,
        forzar_nivel: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward por todos los niveles con early exit.
        
        Args:
            x: (batch, seq_len, embed_dim)
            forzar_nivel: si se especifica, solo usa ese nivel
            
        Returns:
            output: (batch, seq_len, embed_dim)
            info: estadísticas completas
        """
        self.stats['total_calls'] += 1
        
        info = {
            'niveles_ejecutados': [],
            'info_por_nivel': {},
            'nivel_salida': None,
            'early_exit_global': False
        }
        
        current = x
        
        for nivel in sorted(self.config.niveles):
            if forzar_nivel is not None and nivel != forzar_nivel:
                continue
            
            info['niveles_ejecutados'].append(nivel)
            
            # Ejecutar bloque de este nivel
            bloque = self.bloques[str(nivel)]
            current, bloque_info = bloque(current)
            info['info_por_nivel'][nivel] = bloque_info
            
            # Evaluar si continuar al siguiente nivel
            if forzar_nivel is None and nivel < max(self.config.niveles):
                confianza, suficiente = self.evaluador_global(current)
                info[f'confianza_nivel_{nivel}'] = confianza
                
                if suficiente:
                    info['early_exit_global'] = True
                    info['nivel_salida'] = nivel
                    self.stats['exits_por_nivel'][nivel] += 1
                    return current, info
        
        info['nivel_salida'] = max(self.config.niveles)
        return current, info
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas globales."""
        stats = dict(self.stats)
        total = stats['total_calls']
        if total > 0:
            stats['nivel_promedio'] = sum(
                nivel * count for nivel, count in stats['exits_por_nivel'].items()
            ) / max(1, sum(stats['exits_por_nivel'].values()))
        return stats
    
    def print_stats(self):
        """Imprime estadísticas globales."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("BLOQUE FRACTAL MULTINIVEL - ESTADÍSTICAS")
        print("="*60)
        print(f"Total llamadas: {stats['total_calls']}")
        print(f"\nExits por nivel:")
        for nivel in sorted(self.config.niveles):
            count = stats['exits_por_nivel'][nivel]
            pct = count / stats['total_calls'] * 100 if stats['total_calls'] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  Nivel {nivel:3d}: {count:4d} ({pct:5.1f}%) {bar}")
        print("="*60)


# Test
if __name__ == "__main__":
    print("="*60)
    print("TEST BLOQUE FRACTAL")
    print("="*60)
    
    # Config pequeño para test
    config = BloqueFractalConfig(
        embed_dim=64,
        num_heads=4,
        ffn_expansion=2.0,
        num_vecinos=3,
        niveles=[2, 4, 8, 16],
        umbral_confianza=0.5  # Bajo para ver early exits
    )
    
    # Test bloque simple
    print("\n" + "-"*40)
    print("TEST BLOQUE SIMPLE:")
    print("-"*40)
    
    bloque = BloqueFractal(config)
    bloque.print_arquitectura()
    
    # Forward
    x = torch.randn(2, 8, config.embed_dim)
    output, info = bloque(x)
    
    print(f"\nInput: {tuple(x.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Cajas ejecutadas: {info['cajas_ejecutadas']}")
    print(f"Early exit: {info['early_exit']}")
    print(f"Caja salida: {info['caja_salida']}")
    print(f"Contribuciones: {info['contribuciones']}")
    
    # Simular muchas llamadas
    print("\n" + "-"*40)
    print("SIMULACIÓN (100 llamadas):")
    print("-"*40)
    
    for _ in range(100):
        x = torch.randn(1, torch.randint(4, 16, (1,)).item(), config.embed_dim)
        bloque(x)
    
    bloque.print_stats()
    
    # Test multinivel
    print("\n" + "-"*40)
    print("TEST MULTINIVEL:")
    print("-"*40)
    
    multinivel = BloqueFractalMultinivel(config)
    
    x = torch.randn(1, 8, config.embed_dim)
    output, info = multinivel(x)
    
    print(f"\nInput: {tuple(x.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Niveles ejecutados: {info['niveles_ejecutados']}")
    print(f"Nivel salida: {info['nivel_salida']}")
    print(f"Early exit global: {info['early_exit_global']}")
    
    # Simular
    for _ in range(100):
        x = torch.randn(1, torch.randint(4, 32, (1,)).item(), config.embed_dim)
        multinivel(x)
    
    multinivel.print_stats()
    
    print("\n" + "="*60)
    print("✓ BLOQUE FRACTAL IMPLEMENTADO")
    print("="*60)
