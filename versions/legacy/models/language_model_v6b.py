# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6b - Language Model con REFLEXIÓN V2 (Early Exit Inteligente)

CAMBIO CLAVE vs v6:
- La reflexión ahora analiza LOGITS, no solo embeddings
- Early Exit ocurre SOLO si el token predicho es CORRECTO
- Training: early_exit = (token_predicho == target)
- Inference: early_exit = pasa validaciones de calidad

Arquitectura de 27 Cajas:
- Cajas 1-3: PUERTA (gates rápidos O(n))
- Cajas 4-12: NEURAL 1 (aprende, todas las escalas 2→256)
- Cajas 13-15: REFLEXIÓN 1 (analiza logits, ¿early exit si correcto?)
- Cajas 16-24: NEURAL 2 (profundiza si no early exit)
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
from llarri_o1.modules.reflexion_v2 import ModuloReflexionV2, ModuloReflexionFinalV2
from llarri_o1.modules.bloque_neural_v6 import BloqueNeuralV6


@dataclass
class LLARRIConfigV6b:
    """Configuración para LLARRI v6b"""
    vocab_size: int = 256  # Byte-level
    embed_dim: int = 128
    n_heads: int = 4
    max_length: int = 512
    dropout: float = 0.1
    
    # Escalas fractales
    escalas: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
    
    # Early exit thresholds (ahora basados en coincidencia con target)
    threshold_early_exit: float = 1.0  # Solo si coincide exacto
    threshold_correccion: float = 0.35
    
    @property
    def n_escalas(self) -> int:
        return len(self.escalas)


class EmbeddingMultiEscala(nn.Module):
    """
    Embedding que procesa a múltiples escalas.
    """
    
    def __init__(self, config: LLARRIConfigV6b):
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


class CajaLogitsPreliminares(nn.Module):
    """
    Caja auxiliar para calcular logits ANTES de la reflexión.
    Necesarios para que la reflexión pueda evaluar la predicción.
    """
    
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.proj(self.norm(x))


class CajaOutput(nn.Module):
    """
    CAJA 27: Output final
    
    Proyecta embeddings a logits sobre vocabulario.
    """
    
    def __init__(self, config: LLARRIConfigV6b):
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


class LLARRILanguageModelV6b(nn.Module):
    """
    LLARRI v6b - 27 Cajas con Reflexión V2 (Early Exit Inteligente)
    
    DIFERENCIA CLAVE: Early exit solo cuando la predicción es CORRECTA.
    
    Flujo:
    1. Embedding → PUERTA
    2. NEURAL 1 (9 cajas)
    3. Calcular logits preliminares
    4. REFLEXIÓN 1: ¿token predicho == target? → early exit
    5. Si no: NEURAL 2 (9 cajas)
    6. REFLEXIÓN 2: corrección final
    7. OUTPUT
    """
    
    def __init__(self, config: LLARRIConfigV6b):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("INICIALIZANDO LLARRI v6b - REFLEXIÓN V2 (EARLY EXIT INTELIGENTE)")
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
        
        # Logits preliminares ANTES de reflexión 1
        self.logits_preliminares1 = CajaLogitsPreliminares(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size
        )
        print(f"✓ Logits preliminares 1: para Reflexión 1")
        
        # Cajas 13-15: REFLEXIÓN 1 (v2 - analiza logits)
        self.reflexion1 = ModuloReflexionV2(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
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
        
        # Logits preliminares ANTES de reflexión 2
        self.logits_preliminares2 = CajaLogitsPreliminares(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size
        )
        print(f"✓ Logits preliminares 2: para Reflexión 2")
        
        # Cajas 25-26: REFLEXIÓN FINAL (v2)
        self.reflexion2 = ModuloReflexionFinalV2(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size
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
        self.neural2_used_count = 0  # Nuevo: contar cuántas veces se usa Neural 2
        
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
        input_original = h.clone()  # Guardar para comparación
        
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
        # LOGITS PRELIMINARES (para que Reflexión pueda evaluar)
        # ═══════════════════════════════════════════════════════════
        logits_prelim1 = self.logits_preliminares1(h_neural1)
        
        # ═══════════════════════════════════════════════════════════
        # CAJAS 13-15: REFLEXIÓN 1 (¿Early Exit si correcto?)
        # ═══════════════════════════════════════════════════════════
        resultado_ref1 = self.reflexion1(
            embeddings=h_neural1,
            logits=logits_prelim1,
            historial=historial,
            targets=targets  # Pasar targets para comparar
        )
        
        # Decidir si hacer early exit
        # En TRAINING: early_exit = True si token_predicho == target
        # En INFERENCE: early_exit = True si pasa validaciones
        can_early_exit = resultado_ref1.early_exit.all()
        
        if can_early_exit:
            # Early exit: usar logits de reflexión 1
            self.early_exit_count += batch
            logits_final = resultado_ref1.logits
        else:
            # Continuar con Neural 2
            self.neural2_used_count += batch
            h_reflexionado = resultado_ref1.output
            
            # ═══════════════════════════════════════════════════════
            # CAJAS 16-24: NEURAL 2
            # ═══════════════════════════════════════════════════════
            h_neural2, gates_refined2 = self.neural2(
                h_reflexionado, 
                gates_refined1, 
                mask
            )
            
            # Logits preliminares para reflexión 2
            logits_prelim2 = self.logits_preliminares2(h_neural2)
            
            # ═══════════════════════════════════════════════════════
            # CAJAS 25-26: REFLEXIÓN FINAL
            # ═══════════════════════════════════════════════════════
            resultado_ref2 = self.reflexion2(
                embeddings=h_neural2,
                logits=logits_prelim2,
                historial=historial,
                targets=targets
            )
            
            logits_final = resultado_ref2.logits
        
        # ═══════════════════════════════════════════════════════════
        # CAJA 27: OUTPUT (refinamiento final)
        # ═══════════════════════════════════════════════════════════
        # Opción: usar output_caja para refinar, o usar logits de reflexión directamente
        # Aquí usamos los logits de reflexión que ya están corregidos
        logits = logits_final
        
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
                'neural2_usage': self.neural2_used_count / max(1, self.total_forward_count),
                'score_ref1': resultado_ref1.score_confianza.mean().item(),
                'coincide_target': resultado_ref1.metricas.get('coincide_target', torch.tensor([0])).float().mean().item()
            }
        
        return logits, loss
    
    def reset_early_exit_stats(self):
        """Resetear contadores de early exit"""
        self.early_exit_count = 0
        self.total_forward_count = 0
        self.neural2_used_count = 0
    
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
            
            # Forward (sin targets en inferencia)
            logits, _ = self.forward(x, targets=None, historial=historial)
            
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
            output_text = bytes(output_bytes).decode('utf-8', errors='replace')
        except:
            output_text = ''.join(chr(b) if 32 <= b < 127 else '?' for b in output_bytes)
        
        return output_text


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("TEST LLARRI v6b - Reflexión V2")
    print("=" * 60)
    
    config = LLARRIConfigV6b()
    model = LLARRILanguageModelV6b(config)
    
    # Test forward
    batch, seq_len = 2, 64
    x = torch.randint(0, 256, (batch, seq_len))
    targets = torch.randint(0, 256, (batch, seq_len))
    
    print("\n--- Forward con targets (training) ---")
    model.train()
    logits, loss, stats = model(x, targets, return_early_exit_stats=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Early exit rate: {stats['early_exit_rate']*100:.1f}%")
    print(f"Neural2 usage: {stats['neural2_usage']*100:.1f}%")
    print(f"Coincide target: {stats['coincide_target']*100:.1f}%")
    
    print("\n--- Forward sin targets (inference) ---")
    model.eval()
    model.reset_early_exit_stats()
    with torch.no_grad():
        logits, _, stats = model(x, targets=None, return_early_exit_stats=True)
    print(f"Early exit rate: {stats['early_exit_rate']*100:.1f}%")
    print(f"Neural2 usage: {stats['neural2_usage']*100:.1f}%")
    
    print("\n✅ LLARRI v6b funcionando correctamente!")
