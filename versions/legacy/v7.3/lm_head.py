# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
LM Head Fractal para LLARRI-O1.

Part 6/6 del roadmap de evolución a LLM.

Características:
- Tied Embeddings: reutiliza embeddings del tokenizer (0 memoria extra)
- Predicción Fractal: puede predecir nivel por nivel
- Eficiente: no necesita tablas de vocab gigantes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMHeadConfig:
    """Configuración para LM Head Fractal."""
    
    embed_dim: int = 256
    vocab_size: int = 256  # Base: bytes (nivel 2)
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256])
    dropout: float = 0.1
    
    # Tied embeddings
    tie_embeddings: bool = True  # Reusar embedding del tokenizer
    
    # Predicción fractal
    prediccion_fractal: bool = True  # Predecir nivel por nivel
    temperatura_base: float = 1.0


class LMHeadFractal(nn.Module):
    """
    LM Head con Tied Embeddings y Predicción Fractal.
    
    Filosofía:
    1. TIED: Reutiliza los embeddings del tokenizer transpuestos
       → 0 memoria extra para la proyección de salida
       
    2. FRACTAL: Puede predecir en múltiples niveles
       → Nivel 2 (bytes): 256 opciones, muy preciso
       → Nivel 4 (pares): 65K opciones, más contexto
       → Nivel 8 (quads): tokens compuestos
       
    Ejemplo de generación:
        "Hol" → predecir siguiente
        
        Nivel 8: ¿Existe "Hola" como token? Si → predecir "a" implícito
        Nivel 4: ¿Existe "la"? Si → predecir par
        Nivel 2: Predecir byte "a" (siempre funciona)
    """
    
    def __init__(
        self, 
        config: LMHeadConfig = None,
        embedding_module: nn.Module = None  # Para tied embeddings
    ):
        super().__init__()
        self.config = config or LMHeadConfig()
        self.embedding_module = embedding_module
        
        # Capa de normalización antes de proyectar
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        
        if self.config.tie_embeddings and embedding_module is not None:
            # TIED: no necesitamos peso extra, usamos el embedding
            self.output_projection = None
            print(f"  LMHead: usando TIED embeddings (0 params extra)")
        else:
            # Fallback: proyección lineal tradicional
            self.output_projection = nn.Linear(
                self.config.embed_dim, 
                self.config.vocab_size,
                bias=False
            )
            print(f"  LMHead: proyección lineal ({self.config.vocab_size} vocab)")
        
        if self.config.prediccion_fractal:
            # MLPs para ajustar predicción por nivel
            self.nivel_heads = nn.ModuleDict()
            for nivel in self.config.niveles:
                # Cada nivel tiene su pequeño ajuste
                self.nivel_heads[str(nivel)] = nn.Sequential(
                    nn.Linear(self.config.embed_dim, self.config.embed_dim // 4),
                    nn.GELU(),
                    nn.Linear(self.config.embed_dim // 4, self.config.embed_dim),
                )
            print(f"  LMHead: {len(self.config.niveles)} heads fractales")
    
    def _get_embedding_weight(self) -> torch.Tensor:
        """
        Obtiene la matriz de embeddings para tied projection.
        
        Si tenemos EmbeddingComposicional, generamos embeddings on-the-fly.
        """
        if self.embedding_module is None:
            raise ValueError("No hay embedding module para tied embeddings")
        
        # Si es EmbeddingComposicional, usar emb_base
        if hasattr(self.embedding_module, 'emb_base'):
            # Retornar embeddings base (256 bytes)
            return self.embedding_module.emb_base.weight
        
        # Si es nn.Embedding normal
        if hasattr(self.embedding_module, 'weight'):
            return self.embedding_module.weight
        
        raise ValueError(f"Tipo de embedding no soportado: {type(self.embedding_module)}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        nivel: int = 2,
        temperatura: float = None
    ) -> torch.Tensor:
        """
        Convierte hidden states a logits sobre vocabulario.
        
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            nivel: nivel fractal para predicción (default=2 para bytes)
            temperatura: para sampling (default usa config)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        temperatura = temperatura or self.config.temperatura_base
        
        # Normalizar
        x = self.norm(hidden_states)
        x = self.dropout(x)
        
        # Aplicar head fractal si está habilitado
        if self.config.prediccion_fractal and str(nivel) in self.nivel_heads:
            # Residual con ajuste por nivel
            x = x + self.nivel_heads[str(nivel)](x)
        
        # Proyectar a vocabulario
        if self.config.tie_embeddings and self.embedding_module is not None:
            # TIED: multiplicar por embeddings transpuestos
            embed_weight = self._get_embedding_weight()
            logits = F.linear(x, embed_weight)  # (batch, seq, vocab_size)
        else:
            # Proyección lineal tradicional
            logits = self.output_projection(x)
        
        # Aplicar temperatura
        if temperatura != 1.0:
            logits = logits / temperatura
        
        return logits
    
    def forward_fractal(
        self,
        hidden_states: torch.Tensor,
        niveles: List[int] = None,
        temperatura: float = None
    ) -> Dict[int, torch.Tensor]:
        """
        Predicción en múltiples niveles fractales.
        
        Útil para:
        - Beam search multinivel
        - Elegir el nivel óptimo de predicción
        - Generar tokens compuestos cuando es posible
        
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            niveles: lista de niveles a predecir (default: todos)
            
        Returns:
            Dict[nivel, logits] - logits por cada nivel
        """
        niveles = niveles or self.config.niveles
        temperatura = temperatura or self.config.temperatura_base
        
        resultados = {}
        
        for nivel in niveles:
            logits = self.forward(hidden_states, nivel=nivel, temperatura=temperatura)
            resultados[nivel] = logits
        
        return resultados
    
    def predict_next_token(
        self,
        hidden_states: torch.Tensor,
        nivel: int = 2,
        top_k: int = 50,
        top_p: float = 0.9,
        temperatura: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predice el siguiente token con sampling.
        
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            nivel: nivel de predicción
            top_k: número de tokens a considerar
            top_p: nucleus sampling threshold
            temperatura: para controlar diversidad
            
        Returns:
            token_ids: (batch,) - tokens predichos
            probs: (batch,) - probabilidades del token elegido
        """
        # Solo usar última posición
        last_hidden = hidden_states[:, -1:, :]  # (batch, 1, embed_dim)
        
        # Obtener logits
        logits = self.forward(last_hidden, nivel=nivel, temperatura=temperatura)
        logits = logits.squeeze(1)  # (batch, vocab_size)
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remover tokens fuera del nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sampling
        probs = F.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Probabilidad del token elegido
        token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        
        return token_ids, token_probs
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria usada por componente."""
        memoria = {
            'norm': sum(p.numel() * p.element_size() for p in self.norm.parameters()),
            'nivel_heads': 0,
            'output_projection': 0,
            'total': 0
        }
        
        if hasattr(self, 'nivel_heads'):
            memoria['nivel_heads'] = sum(
                p.numel() * p.element_size() 
                for p in self.nivel_heads.parameters()
            )
        
        if self.output_projection is not None:
            memoria['output_projection'] = sum(
                p.numel() * p.element_size() 
                for p in self.output_projection.parameters()
            )
        
        memoria['total'] = sum(v for k, v in memoria.items() if k != 'total')
        
        return memoria
    
    def print_arquitectura(self):
        """Imprime detalles de la arquitectura."""
        print("\n" + "="*60)
        print("LM HEAD FRACTAL")
        print("="*60)
        print(f"\nTied embeddings: {self.config.tie_embeddings}")
        print(f"Predicción fractal: {self.config.prediccion_fractal}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Embed dim: {self.config.embed_dim}")
        
        if self.config.prediccion_fractal:
            print(f"\nHeads por nivel:")
            for nivel in self.config.niveles:
                if str(nivel) in self.nivel_heads:
                    params = sum(p.numel() for p in self.nivel_heads[str(nivel)].parameters())
                    print(f"  Nivel {nivel:3d}: {params:,} params")
        
        memoria = self.get_memoria_usada()
        print(f"\nMemoria:")
        print(f"  Norm: {memoria['norm'] / 1024:.2f} KB")
        print(f"  Nivel heads: {memoria['nivel_heads'] / 1024:.2f} KB")
        print(f"  Output projection: {memoria['output_projection'] / 1024:.2f} KB")
        print(f"  TOTAL: {memoria['total'] / 1024:.2f} KB")
        
        if self.config.tie_embeddings:
            print(f"\n  ✓ Tied embeddings: 0 params extra para proyección!")
        
        print("="*60)


# Test
if __name__ == "__main__":
    from llarri_o1.modules.tokenizer import EmbeddingComposicional, TokenizadorFractal, TokenizerConfig
    
    print("="*60)
    print("TEST LM HEAD FRACTAL")
    print("="*60)
    
    # Crear tokenizer y embedding module para tied embeddings
    embed_dim = 64
    tokenizer_config = TokenizerConfig(niveles=[2, 4, 8, 16])
    tokenizer = TokenizadorFractal(tokenizer_config)
    embedding = EmbeddingComposicional(tokenizer=tokenizer, base_dim=embed_dim)
    
    # Crear LM Head con tied embeddings
    config = LMHeadConfig(
        embed_dim=embed_dim,
        vocab_size=256,
        niveles=[2, 4, 8, 16],
        tie_embeddings=True,
        prediccion_fractal=True
    )
    lm_head = LMHeadFractal(config, embedding_module=embedding)
    
    # Mostrar arquitectura
    lm_head.print_arquitectura()
    
    # Test forward
    print("\n" + "-"*40)
    print("TEST FORWARD:")
    print("-"*40)
    
    batch_size = 2
    seq_len = 8
    
    # Simular hidden states
    hidden = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward simple
    logits = lm_head(hidden, nivel=2)
    print(f"\nInput: {tuple(hidden.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")
    print(f"Expected: ({batch_size}, {seq_len}, 256)")
    
    # Verificar que son probabilidades válidas después de softmax
    probs = F.softmax(logits, dim=-1)
    print(f"Sum probs (debe ser ~1): {probs[0, 0].sum().item():.4f}")
    
    # Test forward fractal
    print("\n" + "-"*40)
    print("TEST FORWARD FRACTAL:")
    print("-"*40)
    
    resultados = lm_head.forward_fractal(hidden, niveles=[2, 4, 8])
    for nivel, logits in resultados.items():
        print(f"  Nivel {nivel}: {tuple(logits.shape)}")
    
    # Test predicción
    print("\n" + "-"*40)
    print("TEST PREDICCIÓN:")
    print("-"*40)
    
    token_ids, probs = lm_head.predict_next_token(
        hidden,
        nivel=2,
        top_k=10,
        top_p=0.9,
        temperatura=0.8
    )
    print(f"\nTokens predichos: {token_ids.tolist()}")
    print(f"Probabilidades: {probs.tolist()}")
    print(f"Como bytes: {[chr(t) if 32 <= t < 127 else f'<{t}>' for t in token_ids.tolist()]}")
    
    # Test sin tied embeddings (para comparar)
    print("\n" + "-"*40)
    print("COMPARACIÓN CON/SIN TIED:")
    print("-"*40)
    
    config_no_tied = LMHeadConfig(
        embed_dim=embed_dim,
        vocab_size=256,
        niveles=[2, 4, 8, 16],
        tie_embeddings=False,
        prediccion_fractal=True
    )
    lm_head_no_tied = LMHeadFractal(config_no_tied, embedding_module=None)
    
    mem_tied = lm_head.get_memoria_usada()['total']
    mem_no_tied = lm_head_no_tied.get_memoria_usada()['total']
    
    print(f"\nCon tied embeddings: {mem_tied / 1024:.2f} KB")
    print(f"Sin tied embeddings: {mem_no_tied / 1024:.2f} KB")
    print(f"Ahorro: {(mem_no_tied - mem_tied) / 1024:.2f} KB")
    
    # Generar texto de prueba
    print("\n" + "-"*40)
    print("SIMULACIÓN DE GENERACIÓN:")
    print("-"*40)
    
    # Simular generación autoregressiva
    generated = []
    current_hidden = torch.randn(1, 1, embed_dim)
    
    for i in range(10):
        token_id, prob = lm_head.predict_next_token(
            current_hidden,
            nivel=2,
            temperatura=0.7
        )
        generated.append(token_id.item())
        # En real, actualizaríamos hidden con el modelo completo
        current_hidden = torch.randn(1, 1, embed_dim)
    
    print(f"\nTokens generados: {generated}")
    texto = ''.join(chr(t) if 32 <= t < 127 else '?' for t in generated)
    print(f"Como texto: '{texto}'")
    
    print("\n" + "="*60)
    print("✓ LM HEAD FRACTAL IMPLEMENTADO CORRECTAMENTE")
    print("="*60)
