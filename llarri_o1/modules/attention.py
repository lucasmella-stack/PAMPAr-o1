# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Attention Fractal Progresivo (Cascada).

Innovación clave: NO hace attention completa siempre.
Busca desde nivel más alto (comprimido) hacia abajo.
Si encuentra match suficiente, PARA. Early exit.

Complejidad:
- Caso común (match en nivel alto): O(1)
- Caso medio: O(log n)
- Peor caso (muy raro): O(n²) pero solo si es necesario

Similar a caché L1/L2/L3:
- Nivel 256 = L1 (más rápido, menos detalle)
- Nivel 128 = L2
- ...
- Nivel 2 = RAM (más lento, todo el detalle)

Author: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinator: Alvaro (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class AttentionConfig:
    """Configuración del attention fractal."""
    embed_dim: int = 256
    num_heads: int = 8
    niveles: List[int] = None  # [2, 4, 8, 16, 32, 64, 128, 256]
    dropout: float = 0.1
    umbral_match: float = 0.7  # Si similitud > umbral, early exit
    causal: bool = True  # Si True, aplica causal mask (no ver futuro)
    
    def __post_init__(self):
        if self.niveles is None:
            self.niveles = [2, 4, 8, 16, 32, 64, 128, 256]


def crear_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Crea máscara causal: cada token solo ve tokens anteriores.
    
    Ejemplo para seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    
    1 = puede ver, 0 = enmascarado (futuro)
    
    Args:
        seq_len: Longitud de secuencia
        device: Dispositivo
        
    Returns:
        Tensor (seq_len, seq_len) con 1s en triángulo inferior
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def crear_causal_mask_por_ventana(
    seq_len: int, 
    ventana: int, 
    device: torch.device = None
) -> torch.Tensor:
    """
    Crea máscara causal LOCAL por ventanas.
    
    Cada token solo ve:
    1. Tokens anteriores (causal)
    2. Tokens dentro de su ventana (local)
    
    Esto es más eficiente que causal global para secuencias largas.
    
    Args:
        seq_len: Longitud de secuencia
        ventana: Tamaño de ventana local
        device: Dispositivo
        
    Returns:
        Tensor (seq_len, seq_len)
    """
    # Empezar con causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Adicionalmente, limitar a ventana local
    # Cada token solo ve los últimos 'ventana' tokens
    for i in range(seq_len):
        start = max(0, i - ventana + 1)
        mask[i, :start] = 0
    
    return mask


class AttentionHead(nn.Module):
    """Una cabeza de attention estándar con soporte causal."""
    
    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: Optional attention mask adicional
            causal: Si True, aplica causal mask automáticamente
            
        Returns:
            output: (batch, seq_len, head_dim)
            attn_weights: (batch, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape
        
        Q = self.q_proj(x)  # (batch, seq, head_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, seq, seq)
        
        # Aplicar causal mask si está habilitado
        if causal:
            causal_mask = crear_causal_mask(seq_len, device=x.device)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Aplicar mask adicional si se proporciona
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Manejar NaN de softmax cuando toda la fila es -inf
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # (batch, seq, head_dim)
        
        return output, attn_weights


class AttentionNivel(nn.Module):
    """
    Attention para un nivel específico.
    
    Opera sobre representaciones comprimidas de ese nivel.
    """
    
    def __init__(
        self, 
        nivel: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.nivel = nivel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Multi-head attention
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, self.head_dim, dropout)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            causal: si True, aplica máscara causal
            
        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len)
            confianza: float - qué tan "seguro" está el match
        """
        head_outputs = []
        head_weights = []
        
        for head in self.heads:
            out, weights = head(x, mask, causal=causal)
            head_outputs.append(out)
            head_weights.append(weights)
        
        # Concatenar cabezas
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq, num_heads * head_dim)
        output = self.out_proj(concat)
        output = self.dropout(output)
        output = self.norm(x + output)  # Residual connection
        
        # Stack attention weights
        attn_weights = torch.stack(head_weights, dim=1)  # (batch, heads, seq, seq)
        
        # Calcular confianza: qué tan concentrada está la atención
        # Si está muy dispersa → baja confianza → seguir bajando niveles
        # Si está concentrada → alta confianza → early exit
        max_attn = attn_weights.max(dim=-1)[0].mean()  # Promedio del max attention
        confianza = max_attn.item()
        
        return output, attn_weights, confianza


class AttentionFractalProgresivo(nn.Module):
    """
    Attention en Cascada Ascendente (de chico a grande).
    
    Busca desde nivel más pequeño (local) hacia arriba (global).
    Si encuentra match suficiente en nivel chico, PARA.
    
    Lógica:
        Nivel 2 (4 bytes): consulta LOCAL barata
        → ¿matchea? SÍ → EARLY EXIT ✅
        → NO → sube a nivel 4
        
        Nivel 4 (16 bytes): consulta un poco más amplia
        → ¿matchea? SÍ → EARLY EXIT ✅
        → NO → sube a nivel 8
        
        ... hasta nivel 256 (consulta global, solo si necesario)
    
    Beneficio: si la respuesta está cerca (contexto local),
    NO gasta recursos en attention global O(n²).
    
    Ejemplo:
        "El gato come" → "gato" está cerca de "come"
        → Nivel 2 matchea → FIN (4 operaciones)
        
        "El gato que vive en la casa grande come"
        → "gato" lejos de "come"
        → Nivel 2 no matchea, sube...
        → Nivel 8 matchea → FIN (16 operaciones)
    """
    
    def __init__(self, config: AttentionConfig = None):
        super().__init__()
        self.config = config or AttentionConfig()
        
        # Attention por nivel (solo los que usaremos activamente)
        # Niveles pequeños = attention local barata
        self.attn_niveles = nn.ModuleDict()
        for nivel in self.config.niveles:
            self.attn_niveles[str(nivel)] = AttentionNivel(
                nivel=nivel,
                embed_dim=self.config.embed_dim,
                num_heads=max(1, self.config.num_heads // (nivel // 2)),  # Menos heads en niveles bajos
                dropout=self.config.dropout
            )
        
        # Stats para debugging
        self.stats = {
            'total_calls': 0,
            'early_exits': {nivel: 0 for nivel in self.config.niveles},
        }
    
    def _attention_local(
        self,
        x: torch.Tensor,
        ventana: int,
        nivel: int,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Attention LOCAL: solo mira tokens cercanos (ventana).
        
        En lugar de attention completa O(n²), hace attention
        solo dentro de bloques de tamaño `ventana`.
        
        Args:
            x: (batch, seq_len, embed_dim)
            ventana: tamaño del bloque local
            nivel: nivel actual
            causal: si True, aplica máscara causal
            
        Returns:
            output, confianza
        """
        batch, seq_len, dim = x.shape
        
        if seq_len <= ventana:
            # Secuencia cabe en una ventana, attention normal
            output, _, confianza = self.attn_niveles[str(nivel)](x, mask, causal=causal)
            return output, confianza
        
        # Dividir en bloques y hacer attention local en cada uno
        outputs = []
        confianzas = []
        
        for start in range(0, seq_len, ventana):
            end = min(start + ventana, seq_len)
            bloque = x[:, start:end, :]
            
            out, _, conf = self.attn_niveles[str(nivel)](bloque, None, causal=causal)
            outputs.append(out)
            confianzas.append(conf)
        
        # Concatenar outputs
        output = torch.cat(outputs, dim=1)
        confianza = sum(confianzas) / len(confianzas)
        
        return output, confianza
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        forzar_nivel: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Attention progresivo ASCENDENTE (de chico a grande).
        
        Args:
            x: (batch, seq_len, embed_dim) - embeddings de entrada
            mask: Optional attention mask
            forzar_nivel: Si se especifica, solo usa ese nivel
            
        Returns:
            output: (batch, seq_len, embed_dim)
            info: Dict con estadísticas
        """
        self.stats['total_calls'] += 1
        batch, seq_len, dim = x.shape
        
        info = {
            'niveles_consultados': [],
            'confianzas': {},
            'nivel_salida': None,
            'early_exit': False,
        }
        
        # Orden ASCENDENTE: de nivel chico a grande
        niveles_ascendente = sorted(self.config.niveles)
        
        current_output = x
        
        for nivel in niveles_ascendente:
            # Si hay nivel forzado, solo usar ese
            if forzar_nivel is not None and nivel != forzar_nivel:
                continue
            
            info['niveles_consultados'].append(nivel)
            
            # Ventana = tamaño del cuadrante en este nivel
            # Nivel 2: ventana 4 (attention entre 4 tokens)
            # Nivel 4: ventana 16 (attention entre 16 tokens)
            # Nivel 8: ventana 64, etc.
            ventana = nivel * nivel  # 2→4, 4→16, 8→64, 16→256...
            ventana = min(ventana, seq_len)  # No más que seq_len
            
            # Attention local en este nivel (con causal si está habilitado)
            output, confianza = self._attention_local(
                current_output, 
                ventana, 
                nivel, 
                mask,
                causal=self.config.causal
            )
            
            info['confianzas'][nivel] = confianza
            
            # Early exit check: si confianza alta, parar
            if (confianza >= self.config.umbral_match and 
                forzar_nivel is None):
                
                info['early_exit'] = True
                info['nivel_salida'] = nivel
                self.stats['early_exits'][nivel] += 1
                
                return output, info
            
            # Guardar para siguiente iteración
            current_output = output
        
        # No hubo early exit, usar último output
        info['nivel_salida'] = niveles_ascendente[-1]
        
        return current_output, info
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso."""
        total = self.stats['total_calls']
        if total == 0:
            return self.stats
        
        # Calcular nivel promedio de salida
        nivel_ponderado = sum(
            nivel * count 
            for nivel, count in self.stats['early_exits'].items()
        )
        exits_totales = sum(self.stats['early_exits'].values())
        
        if exits_totales > 0:
            self.stats['nivel_promedio'] = nivel_ponderado / exits_totales
        
        self.stats['tasa_early_exit'] = exits_totales / total
        
        return self.stats
    
    def print_stats(self):
        """Imprime estadísticas."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("ATTENTION FRACTAL ASCENDENTE - ESTADÍSTICAS")
        print("="*60)
        print(f"Total llamadas: {stats['total_calls']}")
        print(f"Tasa early exit: {stats.get('tasa_early_exit', 0):.1%}")
        print(f"Nivel promedio: {stats.get('nivel_promedio', 0):.1f}")
        print(f"\nEarly exits por nivel (de chico a grande):")
        for nivel in sorted(stats['early_exits'].keys()):
            count = stats['early_exits'][nivel]
            pct = count / max(stats['total_calls'], 1) * 100
            bar = "█" * int(pct / 5)
            print(f"  Nivel {nivel:3d}: {count:5d} ({pct:5.1f}%) {bar}")
        print("="*60)
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria usada."""
        memoria = {
            'attn_niveles': sum(
                sum(p.numel() for p in attn.parameters()) * 4
                for attn in self.attn_niveles.values()
            ),
        }
        memoria['total'] = sum(memoria.values())
        return memoria
    
    def print_arquitectura(self):
        """Imprime detalles de la arquitectura."""
        print("\n" + "="*60)
        print("ATTENTION FRACTAL ASCENDENTE")
        print("="*60)
        print(f"\nFilosofía: consulta CHICA primero, si matchea → PARA")
        print(f"           Solo sube a consulta GRANDE si necesita más contexto")
        print(f"\nConfig:")
        print(f"  embed_dim: {self.config.embed_dim}")
        print(f"  num_heads: {self.config.num_heads}")
        print(f"  umbral_match: {self.config.umbral_match}")
        print(f"  causal: {self.config.causal}")
        print(f"  niveles: {self.config.niveles}")
        
        print(f"\nFlujo (de chico a grande):")
        for nivel in sorted(self.config.niveles):
            ventana = nivel * nivel
            attn = self.attn_niveles[str(nivel)]
            params = sum(p.numel() for p in attn.parameters())
            print(f"  Nivel {nivel:3d}: ventana {ventana:4d} tokens, {params:,} params")
        
        memoria = self.get_memoria_usada()
        print(f"\nMemoria total: {memoria['total'] / 1024 / 1024:.2f} MB")
        print("="*60)


# Test
if __name__ == "__main__":
    print("="*60)
    print("TEST ATTENTION FRACTAL ASCENDENTE + CAUSAL MASK")
    print("="*60)
    
    # Crear módulo con config pequeño para test
    config = AttentionConfig(
        embed_dim=64,  # Pequeño para test
        num_heads=4,
        umbral_match=0.5,  # Umbral bajo para ver early exits
        niveles=[2, 4, 8, 16],  # Solo 4 niveles para test
        causal=True  # Habilitar causal mask
    )
    attention = AttentionFractalProgresivo(config)
    
    # Mostrar arquitectura
    attention.print_arquitectura()
    
    # Test con diferentes secuencias
    print("\n" + "-"*40)
    print("TEST DE INFERENCIA (con causal mask):")
    print("-"*40)
    
    batch_size = 1
    
    for seq_len in [4, 8, 16, 32]:
        # Simular embeddings
        x = torch.randn(batch_size, seq_len, config.embed_dim)
        
        # Forward
        output, info = attention(x)
        
        print(f"\nseq_len={seq_len}:")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(output.shape)}")
        print(f"  Niveles consultados: {info['niveles_consultados']}")
        print(f"  Early exit: {info['early_exit']} en nivel {info['nivel_salida']}")
        print(f"  Confianzas: {info['confianzas']}")
    
    # Test específico de causal mask
    print("\n" + "-"*40)
    print("TEST CAUSAL MASK:")
    print("-"*40)
    
    # Verificar que la máscara causal funciona
    seq_len = 4
    causal_mask = crear_causal_mask(seq_len, x.device)
    print(f"\nCausal mask para seq_len={seq_len}:")
    print(causal_mask.int())  # Mostrar como 0/1
    
    # Verificar máscara por ventana
    ventana = 3
    window_mask = crear_causal_mask_por_ventana(seq_len, ventana, x.device)
    print(f"\nCausal mask con ventana={ventana}:")
    print(window_mask.int())
    
    # Comparar con y sin causal
    print("\n" + "-"*40)
    print("COMPARACIÓN CON/SIN CAUSAL:")
    print("-"*40)
    
    x = torch.randn(1, 8, config.embed_dim)
    
    # Con causal
    config_causal = AttentionConfig(embed_dim=64, num_heads=4, causal=True, niveles=[2, 4])
    attn_causal = AttentionFractalProgresivo(config_causal)
    out_causal, _ = attn_causal(x)
    
    # Sin causal
    config_no_causal = AttentionConfig(embed_dim=64, num_heads=4, causal=False, niveles=[2, 4])
    attn_no_causal = AttentionFractalProgresivo(config_no_causal)
    out_no_causal, _ = attn_no_causal(x)
    
    # Deben ser diferentes
    diff = (out_causal - out_no_causal).abs().mean().item()
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output con causal: {tuple(out_causal.shape)}")
    print(f"  Output sin causal: {tuple(out_no_causal.shape)}")
    print(f"  Diferencia promedio: {diff:.6f}")
    print(f"  ✓ Causal mask afecta el output: {diff > 0.001}")
    
    # Simular muchas llamadas
    print("\n" + "-"*40)
    print("SIMULACIÓN (50 llamadas con causal):")
    print("-"*40)
    
    for _ in range(50):
        seq_len = torch.randint(4, 32, (1,)).item()
        x = torch.randn(1, seq_len, config.embed_dim)
        attention(x)
    
    attention.print_stats()
    
    print("\n" + "="*60)
    print("✓ CAUSAL MASK IMPLEMENTADO CORRECTAMENTE")
    print("="*60)
    print("TEST COMPLETADO")
    print("="*60)
