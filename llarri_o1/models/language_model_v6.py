# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Language Model con 27 Cajas

Arquitectura de REFLEXIÓN FRACTAL con EARLY EXIT:
- Cajas 1-3: PUERTA (gates rápidos O(n))
- Cajas 4-12: NEURAL 1 (aprende, todas las escalas 2→256)
- Cajas 13-15: REFLEXIÓN 1 (compara, corrige, ¿early exit?)
- Cajas 16-24: NEURAL 2 (profundiza)
- Cajas 25-26: REFLEXIÓN 2 (corrección final)
- Caja 27: OUTPUT (proyección a vocabulario)

Escalas: 2, 4, 8, 16, 32, 64, 128, 256 (8 niveles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from llarri_o1.modules.puerta import ModuloPuerta
from llarri_o1.modules.reflexion import ModuloReflexion, ModuloReflexionFinal
from llarri_o1.modules.bloque_neural_v6 import BloqueNeuralV6


@dataclass
class LLARRIConfigV6:
    """Configuración para LLARRI v6"""
    vocab_size: int = 256  # Byte-level
    embed_dim: int = 128
    n_heads: int = 4
    max_length: int = 512
    dropout: float = 0.1
    
    # Escalas fractales
    escalas: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
    
    # Early exit thresholds
    threshold_early_exit: float = 0.85
    threshold_correccion: float = 0.35
    
    @property
    def n_escalas(self) -> int:
        return len(self.escalas)


class EmbeddingMultiEscala(nn.Module):
    """
    Embedding que procesa a múltiples escalas.
    Cada escala agrupa tokens de diferente manera.
    """
    
    def __init__(self, config: LLARRIConfigV6):
        super().__init__()
        self.config = config
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Embedding posicional
        self.pos_embedding = nn.Embedding(config.max_length, config.embed_dim)
        
        # Proyección para combinar escalas
        self.scale_proj = nn.Linear(config.embed_dim * config.n_escalas, config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token ids
        Returns:
            embeddings: [batch, seq_len, embed_dim]
        """
        batch, seq_len = x.shape
        
        # Embedding básico
        tok_emb = self.token_embedding(x)  # [batch, seq_len, embed_dim]
        
        # Posiciones
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, embed_dim]
        
        # Combinar
        emb = tok_emb + pos_emb
        
        return self.dropout(emb)


class CajaOutput(nn.Module):
    """
    CAJA 27: Output final
    
    Proyecta embeddings a logits sobre vocabulario.
    """
    
    def __init__(self, config: LLARRIConfigV6):
        super().__init__()
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.proj(self.norm(x))


class LLARRILanguageModelV6(nn.Module):
    """
    LLARRI v6 - 27 Cajas con Reflexión y Early Exit
    
    Arquitectura:
    ┌─────────────┐
    │ Embedding   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ PUERTA 1-3  │ ← Gates rápidos
    └──────┬──────┘
           │ + residuo ─────────────────────────────────┐
    ┌──────▼──────┐                                     │
    │ NEURAL 4-12 │ ← 9 cajas, todas las escalas       │
    └──────┬──────┘                                     │
           │                                            │
    ┌──────▼─────────────────────────────────────────┐  │
    │ REFLEXIÓN 13-15                                │◄─┤
    │ ¿Early exit? ──────────────────────────────────│──│──► OUTPUT
    └──────┬──────┘                                  │  │
           │ (si no early exit)                      │  │
           │ + residuo ────────────────────────────┐ │  │
    ┌──────▼──────┐                                │ │  │
    │ NEURAL 16-24│ ← 9 cajas más                  │ │  │
    └──────┬──────┘                                │ │  │
           │                                       │ │  │
    ┌──────▼──────────────────────────────────────┐│ │  │
    │ REFLEXIÓN 25-26                             │◄┘ │  │
    │ Corrección final                            │   │  │
    └──────┬──────────────────────────────────────┘   │  │
           │                                          │  │
    ┌──────▼──────┐                                   │  │
    │ OUTPUT 27   │◄──────────────────────────────────┴──┘
    └──────┬──────┘
           │
         LOGITS
    """
    
    def __init__(self, config: LLARRIConfigV6):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("INICIALIZANDO LLARRI v6 - 27 CAJAS CON REFLEXIÓN")
        print("=" * 60)
        
        # Embedding
        self.embedding = EmbeddingMultiEscala(config)
        print(f"✓ Embedding: vocab={config.vocab_size}, dim={config.embed_dim}")
        
        # Cajas 1-3: PUERTA
        self.puerta = ModuloPuerta(
            embed_dim=config.embed_dim,
            escalas=list(config.escalas)
        )
        
        # Cajas 4-12: NEURAL BLOQUE 1
        self.neural1 = BloqueNeuralV6(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_escalas=config.n_escalas,
            dropout=config.dropout,
            nombre="Neural1 (4-12)"
        )
        
        # Cajas 13-15: REFLEXIÓN 1
        self.reflexion1 = ModuloReflexion(
            embed_dim=config.embed_dim,
            n_escalas=config.n_escalas,
            threshold_alto=config.threshold_early_exit,
            threshold_bajo=config.threshold_correccion,
            nombre="Reflexion1 (13-15)"
        )
        
        # Cajas 16-24: NEURAL BLOQUE 2
        self.neural2 = BloqueNeuralV6(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_escalas=config.n_escalas,
            dropout=config.dropout,
            nombre="Neural2 (16-24)"
        )
        
        # Cajas 25-26: REFLEXIÓN FINAL
        self.reflexion2 = ModuloReflexionFinal(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            n_escalas=config.n_escalas
        )
        
        # Caja 27: OUTPUT
        self.output_caja = CajaOutput(config)
        print(f"✓ Caja 27: Output → vocab_size={config.vocab_size}")
        
        # Máscara causal
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_length, config.max_length))
        )
        
        # Contadores para estadísticas de early exit
        self.early_exit_count = 0
        self.total_forward_count = 0
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 Total parámetros: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"📊 Entrenables: {trainable_params:,}")
        print("=" * 60)
        
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        historial: Optional[torch.Tensor] = None,
        return_early_exit_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len] token ids
            targets: [batch, seq_len] target ids para loss
            historial: [batch, n] tokens previos generados
            return_early_exit_stats: si devolver stats de early exit
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: escalar si hay targets
        """
        batch, seq_len = x.shape
        self.total_forward_count += batch
        
        # Embedding
        h = self.embedding(x)  # [batch, seq_len, embed_dim]
        input_original = h.clone()  # Guardar para reflexión
        
        # Máscara causal
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # ═══════════════════════════════════════════════════════════
        # CAJAS 1-3: PUERTA
        # ═══════════════════════════════════════════════════════════
        gates, metricas_puerta = self.puerta(h)
        
        # ═══════════════════════════════════════════════════════════
        # CAJAS 4-12: NEURAL 1
        # ═══════════════════════════════════════════════════════════
        h_neural1, gates_refined1 = self.neural1(h, gates, mask)
        
        # ═══════════════════════════════════════════════════════════
        # CAJAS 13-15: REFLEXIÓN 1 (¿Early Exit?)
        # ═══════════════════════════════════════════════════════════
        resultado_ref1 = self.reflexion1(h_neural1, input_original, historial)
        
        # Decidir si hacer early exit (solo en inferencia, no en training)
        if not self.training and resultado_ref1.early_exit.all():
            # Todos los elementos del batch pueden hacer early exit
            self.early_exit_count += batch
            h_final = resultado_ref1.output
        else:
            # Continuar con el procesamiento
            h_reflexionado = resultado_ref1.output
            
            # ═══════════════════════════════════════════════════════
            # CAJAS 16-24: NEURAL 2
            # ═══════════════════════════════════════════════════════
            # Usar gates refinados de la reflexión o del bloque anterior
            h_neural2, gates_refined2 = self.neural2(
                h_reflexionado, 
                gates_refined1, 
                mask
            )
            
            # ═══════════════════════════════════════════════════════
            # CAJAS 25-26: REFLEXIÓN FINAL
            # ═══════════════════════════════════════════════════════
            resultado_ref2 = self.reflexion2(
                h_neural2, 
                input_original,  # Comparar con input ORIGINAL
                historial
            )
            
            h_final = resultado_ref2.output
        
        # ═══════════════════════════════════════════════════════════
        # CAJA 27: OUTPUT
        # ═══════════════════════════════════════════════════════════
        logits = self.output_caja(h_final)
        
        # Calcular loss si hay targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        if return_early_exit_stats:
            return logits, loss, {
                'early_exit_rate': self.early_exit_count / max(1, self.total_forward_count),
                'score_ref1': resultado_ref1.score_confianza.mean().item(),
            }
        
        return logits, loss
    
    def reset_early_exit_stats(self):
        """Resetear contadores de early exit"""
        self.early_exit_count = 0
        self.total_forward_count = 0
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Genera tokens autoregressivamente.
        
        Args:
            prompt: [1, seq_len] tokens iniciales
            max_new_tokens: cuántos tokens generar
            temperature: temperatura para sampling
            top_k: filtrar a top k tokens
            top_p: nucleus sampling
        Returns:
            tokens: [1, seq_len + max_new_tokens]
        """
        self.eval()
        self.reset_early_exit_stats()
        
        tokens = prompt.clone()
        historial = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Truncar si es muy largo
            x = tokens[:, -self.config.max_length:]
            
            # Forward
            logits, _ = self.forward(x, historial=historial)
            
            # Tomar logits del último token
            logits_last = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits_last < torch.topk(logits_last, top_k)[0][..., -1, None]
                logits_last[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_last[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
            historial = torch.cat([historial, next_token], dim=1)
        
        return tokens
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8
    ) -> str:
        """
        Genera texto desde un prompt string.
        """
        # Encode prompt
        prompt_tokens = torch.tensor(
            [list(prompt.encode('utf-8'))],
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        
        # Generate
        output_tokens = self.generate(
            prompt_tokens,
            max_new_tokens=max_length,
            temperature=temperature
        )
        
        # Decode
        output_bytes = output_tokens[0].tolist()
        try:
            text = bytes(output_bytes).decode('utf-8', errors='replace')
        except:
            text = ''.join(chr(b) if 32 <= b < 127 else '?' for b in output_bytes)
        
        return text


def create_model(config: Optional[LLARRIConfigV6] = None) -> LLARRILanguageModelV6:
    """Factory function para crear modelo v6"""
    if config is None:
        config = LLARRIConfigV6()
    return LLARRILanguageModelV6(config)


# Test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Test LLARRI v6 - 27 Cajas")
    print("=" * 60)
    
    # Crear modelo
    config = LLARRIConfigV6(
        embed_dim=128,
        n_heads=4,
        max_length=256
    )
    model = create_model(config)
    
    # Test forward
    batch, seq_len = 4, 64
    x = torch.randint(0, 256, (batch, seq_len))
    targets = torch.randint(0, 256, (batch, seq_len))
    
    print("\n📝 Test Forward:")
    logits, loss = model(x, targets)
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test generación
    print("\n📝 Test Generación:")
    prompt = "Once upon a time"
    generated = model.generate_text(prompt, max_length=50, temperature=0.8)
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated[:100]}...")
    
    # Estadísticas de early exit
    print(f"\n📊 Early Exit Rate: {model.early_exit_count}/{model.total_forward_count}")
    
    print("\n✅ LLARRI v6 funcionando!")
