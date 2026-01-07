# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
LLARRI Language Model - Modelo de Lenguaje Fractal Unificado.

Integra todos los componentes desarrollados en el roadmap:
1. TokenizadorFractal - tokenización jerárquica
2. EmbeddingComposicional - embeddings eficientes (24x menos memoria)
3. EmbeddingPosicionalFractal - posición sinusoidal + jerárquica
4. BloqueFractal - 6 cajas (mezcla → procesa → evalúa → output)
5. Causal Mask - máscara autoregresiva
6. LMHeadFractal - predicción con tied embeddings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llarri_o1.modules.tokenizer import (
    TokenizadorFractal,
    TokenizerConfig,
    EmbeddingComposicional,
    EmbeddingPosicionalFractal,
)
from llarri_o1.modules.bloque_fractal import (
    BloqueFractal,
    BloqueFractalConfig,
    BloqueFractalMultinivel,
)
from llarri_o1.modules.lm_head import (
    LMHeadFractal,
    LMHeadConfig,
)


@dataclass
class LLARRIConfig:
    """Configuración unificada para LLARRI Language Model."""
    
    # Dimensiones
    embed_dim: int = 256
    base_dim: int = 64  # Para embedding composicional
    max_dim: int = 512
    
    # Niveles fractales
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256])
    
    # Bloque Fractal (6 cajas)
    num_heads: int = 4
    ffn_expansion: float = 2.0
    num_vecinos: int = 3  # Cajas que procesan (2, 3, 4)
    umbral_confianza: float = 0.7
    
    # LM Head
    tie_embeddings: bool = True
    prediccion_fractal: bool = True
    
    # General
    dropout: float = 0.1
    vocab_size: int = 256  # Bytes base
    
    # Generación
    max_length: int = 1024
    temperatura_default: float = 1.0


class LLARRILanguageModel(nn.Module):
    """
    LLARRI Language Model - Modelo de Lenguaje Fractal.
    
    Arquitectura con 6 Cajas:
    ```
    Texto → Tokenizer → Embedding → +Posición → BloqueFractal → LMHead → Logits
           fractal    composicional  fractal     6 cajas        tied
    ```
    
    BloqueFractal (6 cajas):
    - Caja 1: MEZCLA (Attention)
    - Cajas 2,3,4: PROCESAN (FFN distribuido, de cerca a lejos)
    - Caja 5: EVALÚA (early exit)
    - Caja 6: OUTPUT
    
    Características únicas:
    - Tokenización multinivel (bytes → pares → quads → ...)
    - Embeddings composicionales (24x menos memoria)
    - Procesamiento distribuido (mezcla → procesa cercano → lejano)
    - Early exit (para cuando es suficiente)
    """
    
    def __init__(self, config: LLARRIConfig = None):
        super().__init__()
        self.config = config or LLARRIConfig()
        
        print("="*60)
        print("INICIALIZANDO LLARRI LANGUAGE MODEL v2")
        print("="*60)
        
        # 1. Tokenizer Fractal
        tokenizer_config = TokenizerConfig(niveles=self.config.niveles)
        self.tokenizer = TokenizadorFractal(tokenizer_config)
        print(f"✓ TokenizadorFractal: {len(self.config.niveles)} niveles")
        
        # 2. Embedding Composicional
        self.embedding = EmbeddingComposicional(
            tokenizer=self.tokenizer,
            base_dim=self.config.base_dim,
            max_dim=self.config.max_dim,
            dropout=self.config.dropout
        )
        print(f"✓ EmbeddingComposicional: {self.config.base_dim}→{self.config.max_dim} dims")
        
        # 3. Embedding Posicional Fractal
        self.pos_embedding = EmbeddingPosicionalFractal(
            embed_dim=self.config.base_dim,
            max_seq_len=self.config.max_length,
            niveles=self.config.niveles,
            dropout=self.config.dropout
        )
        print(f"✓ EmbeddingPosicionalFractal: max_len={self.config.max_length}")
        
        # 4. Bloque Fractal (6 Cajas) - REEMPLAZA AttentionFractalProgresivo
        bloque_config = BloqueFractalConfig(
            embed_dim=self.config.base_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            ffn_expansion=self.config.ffn_expansion,
            num_vecinos=self.config.num_vecinos,
            niveles=self.config.niveles[:4],
            umbral_confianza=self.config.umbral_confianza
        )
        self.bloque_fractal = BloqueFractal(bloque_config)
        print(f"✓ BloqueFractal: 6 cajas, {self.config.num_vecinos} vecinos, umbral={self.config.umbral_confianza}")
        
        # 5. LM Head Fractal (con tied embeddings)
        lm_head_config = LMHeadConfig(
            embed_dim=self.config.base_dim,
            vocab_size=self.config.vocab_size,
            niveles=self.config.niveles[:4],
            tie_embeddings=self.config.tie_embeddings,
            prediccion_fractal=self.config.prediccion_fractal,
            dropout=self.config.dropout
        )
        self.lm_head = LMHeadFractal(
            config=lm_head_config,
            embedding_module=self.embedding
        )
        print(f"✓ LMHeadFractal: tied={self.config.tie_embeddings}")
        
        # Proyección para ajustar dimensiones si es necesario
        self.embed_proj = nn.Linear(self.config.base_dim, self.config.base_dim)
        
        print("="*60)
    
    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        nivel: int = 2
    ) -> torch.Tensor:
        """
        Obtiene embeddings de tokens.
        
        Args:
            input_ids: (batch, seq_len) tensor de tokens
            nivel: nivel de los tokens
            
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # EmbeddingComposicional espera tokens por nivel, no tensores directos
        # Para nivel 2 (bytes), usamos el embedding base directamente
        if nivel == 2:
            # Clamp a rango válido
            tokens = input_ids.clamp(0, 255)
            embeddings = self.embedding.emb_base(tokens)  # (batch, seq, base_dim)
        else:
            # Para niveles superiores, procesar batch por batch
            embeddings_list = []
            for b in range(batch_size):
                tokens_list = input_ids[b].tolist()
                tokens_dict = {nivel: tokens_list}
                # También necesitamos los bytes para el combinador
                # Pero si ya tenemos tokens de nivel superior, asumimos
                # que el embedding base se usará directamente
                emb = self.embedding.embed_tokens(tokens_dict, nivel_objetivo=nivel)
                embeddings_list.append(emb)
            embeddings = torch.stack(embeddings_list, dim=0)
        
        return embeddings

    def encode(
        self, 
        text: str, 
        nivel: int = 2
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Codifica texto a embeddings.
        
        Args:
            text: texto a codificar
            nivel: nivel de tokenización (default=2 para bytes)
            
        Returns:
            embeddings: (1, seq_len, embed_dim)
            token_ids: lista de token ids
        """
        # Tokenizar (encode devuelve dict por nivel)
        tokens_por_nivel = self.tokenizer.encode(text, max_nivel=nivel)
        token_ids = tokens_por_nivel.get(nivel, tokens_por_nivel[2])
        
        # Convertir a tensor
        tokens = torch.tensor([token_ids], dtype=torch.long)
        
        # Obtener embeddings
        embeddings = self._get_embeddings(tokens, nivel=nivel)
        
        # Agregar posición (pos_embedding espera seq_len, no nivel)
        batch_size, seq_len, _ = embeddings.shape
        pos_emb = self.pos_embedding(seq_len, device=embeddings.device)  # (seq_len, embed_dim)
        embeddings = embeddings + pos_emb.unsqueeze(0)  # broadcast batch
        
        return embeddings, token_ids
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        nivel: int = 2,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo.
        
        Args:
            input_ids: (batch, seq_len) token ids
            text: texto alternativo (se tokeniza internamente)
            nivel: nivel de tokenización
            labels: (batch, seq_len) para calcular loss
            
        Returns:
            Dict con:
                - logits: (batch, seq_len, vocab_size)
                - loss: si labels está presente
                - attention_info: estadísticas de attention
        """
        # Obtener embeddings
        if text is not None:
            embeddings, _ = self.encode(text, nivel=nivel)
        elif input_ids is not None:
            embeddings = self._get_embeddings(input_ids, nivel=nivel)
            # Agregar posición
            batch_size, seq_len, _ = embeddings.shape
            pos_emb = self.pos_embedding(seq_len, device=embeddings.device)
            embeddings = embeddings + pos_emb.unsqueeze(0)
        else:
            raise ValueError("Debe proporcionar input_ids o text")
        
        # Proyectar a dimensión fija
        x = self.embed_proj(embeddings)
        
        # Bloque Fractal (6 cajas: mezcla → procesa → evalúa → output)
        x, bloque_info = self.bloque_fractal(x)
        
        # LM Head → logits
        logits = self.lm_head(x, nivel=nivel)
        
        output = {
            'logits': logits,
            'bloque_info': bloque_info,
            'hidden_states': x
        }
        
        # Calcular loss si hay labels
        if labels is not None:
            # Shift para next-token prediction
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
        nivel: int = 2,
        temperatura: float = None,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt: texto inicial
            max_new_tokens: máximo de tokens a generar
            nivel: nivel de tokenización
            temperatura: para sampling
            top_k: top-k filtering
            top_p: nucleus sampling
            do_sample: si False, usa greedy decoding
            
        Returns:
            texto generado (incluyendo prompt)
        """
        temperatura = temperatura or self.config.temperatura_default
        self.eval()
        
        # Tokenizar prompt (encode devuelve dict por nivel)
        tokens_por_nivel = self.tokenizer.encode(prompt, max_nivel=nivel)
        token_ids = tokens_por_nivel.get(nivel, tokens_por_nivel[2])
        generated = list(token_ids)
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = generated[-self.config.max_length:]
            
            # Forward
            input_ids = torch.tensor([context], dtype=torch.long)
            output = self.forward(input_ids=input_ids, nivel=nivel)
            
            # Obtener logits del último token
            logits = output['logits'][:, -1, :]  # (1, vocab_size)
            
            if do_sample:
                # Sampling con temperatura
                logits = logits / temperatura
                
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
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy
                next_token = logits.argmax(dim=-1).item()
            
            generated.append(next_token)
            
            # Check for EOS (newline or special)
            if next_token == 0 or next_token == 10:  # null or newline
                break
        
        # Decodificar (decode espera dict, creamos uno con nivel 2)
        tokens_dict = {nivel: generated}
        return self.tokenizer.decode(tokens_dict, usar_nivel=nivel)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Cuenta parámetros del modelo."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.embedding.parameters())
        return n_params
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria por componente."""
        memoria = {}
        
        memoria['embedding'] = sum(
            p.numel() * p.element_size() for p in self.embedding.parameters()
        )
        memoria['pos_embedding'] = sum(
            p.numel() * p.element_size() for p in self.pos_embedding.parameters()
        )
        memoria['bloque_fractal'] = sum(
            p.numel() * p.element_size() for p in self.bloque_fractal.parameters()
        )
        memoria['lm_head'] = sum(
            p.numel() * p.element_size() for p in self.lm_head.parameters()
        )
        memoria['otros'] = sum(
            p.numel() * p.element_size() for p in self.embed_proj.parameters()
        )
        
        memoria['total'] = sum(memoria.values())
        
        return memoria
    
    def print_arquitectura(self):
        """Imprime resumen de la arquitectura."""
        print("\n" + "="*70)
        print("LLARRI LANGUAGE MODEL v2 - ARQUITECTURA")
        print("="*70)
        
        print(f"\n📊 CONFIGURACIÓN:")
        print(f"  Niveles fractales: {self.config.niveles}")
        print(f"  Embed dim: {self.config.base_dim} → {self.config.max_dim}")
        print(f"  Vocab size: {self.config.vocab_size} (bytes)")
        print(f"  Max length: {self.config.max_length}")
        
        print(f"\n🔧 COMPONENTES:")
        print(f"  1. TokenizadorFractal: {len(self.config.niveles)} niveles jerárquicos")
        print(f"  2. EmbeddingComposicional: 256 base + MLPs (24x menos memoria)")
        print(f"  3. EmbeddingPosicionalFractal: sinusoidal + jerárquico")
        print(f"  4. BloqueFractal: 6 cajas (mezcla→procesa→evalúa→output)")
        print(f"  5. LMHeadFractal: tied embeddings")
        
        print(f"\n📈 PARÁMETROS:")
        total = self.get_num_params()
        non_emb = self.get_num_params(non_embedding=True)
        print(f"  Total: {total:,}")
        print(f"  Sin embeddings: {non_emb:,}")
        
        memoria = self.get_memoria_usada()
        print(f"\n💾 MEMORIA:")
        for k, v in memoria.items():
            if k != 'total':
                print(f"  {k}: {v / 1024:.2f} KB")
        print(f"  TOTAL: {memoria['total'] / 1024 / 1024:.2f} MB")
        
        print("\n" + "="*70)
        print("FLUJO DE DATOS (6 CAJAS):")
        print("="*70)
        print("""
  Texto ─────────────────────────────────────────────────────────────┐
         │                                                           │
         ▼                                                           │
  ┌─────────────────┐                                                │
  │ TokenizadorFractal │  "Hola" → [72,111,108,97] (bytes)          │
  └────────┬────────┘                                                │
           │                                                         │
           ▼                                                         │
  ┌─────────────────────┐                                            │
  │ EmbeddingComposicional │  tokens → vectors (64 dims)            │
  └────────┬────────────┘                                            │
           │                                                         │
           ▼                                                         │
  ┌──────────────────────────┐                                       │
  │ EmbeddingPosicionalFractal │  + posición                        │
  └────────┬─────────────────┘                                       │
           │                                                         │
           ▼                                                         │
  ┌────────────────────────────────────────────────────────────────┐ │
  │                    BLOQUE FRACTAL (6 CAJAS)                    │ │
  │ ┌────────────────┐                                             │ │
  │ │ Caja 1: MEZCLA │ Attention - ¿qué es relevante?              │ │
  │ └───────┬────────┘                                             │ │
  │         ▼                                                      │ │
  │ ┌────────────────┐                                             │ │
  │ │ Caja 2: PROCESA│ FFN cercano (0.5x)                          │ │
  │ └───────┬────────┘                                             │ │
  │         ├──► Caja 5: EVALÚA ──► ¿Suficiente? ──► SALIR        │ │
  │         ▼                                                      │ │
  │ ┌────────────────┐                                             │ │
  │ │ Caja 3: PROCESA│ FFN medio (0.75x)                           │ │
  │ └───────┬────────┘                                             │ │
  │         ├──► Caja 5: EVALÚA ──► ¿Suficiente? ──► SALIR        │ │
  │         ▼                                                      │ │
  │ ┌────────────────┐                                             │ │
  │ │ Caja 4: PROCESA│ FFN lejano (1.0x)                           │ │
  │ └───────┬────────┘                                             │ │
  │         ▼                                                      │ │
  │ ┌────────────────┐                                             │ │
  │ │ Caja 6: OUTPUT │ Preparar resultado                          │ │
  │ └───────┬────────┘                                             │ │
  └─────────┼──────────────────────────────────────────────────────┘ │
            │                                                         │
            ▼                                                         │
  ┌─────────────────┐                                                │
  │  LMHeadFractal  │  hidden → logits (256 vocab)                  │
  └────────┬────────┘                                                │
           │                                                         │
           ▼                                                         │
      Siguiente Token ◄──────────────────────────────────────────────┘
""")
        print("="*70)


# Test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST LLARRI LANGUAGE MODEL v2 (con BloqueFractal)")
    print("="*70 + "\n")
    
    # Crear modelo con config pequeño para test
    config = LLARRIConfig(
        embed_dim=64,
        base_dim=64,
        max_dim=128,
        niveles=[2, 4, 8, 16],
        num_heads=4,
        ffn_expansion=2.0,
        num_vecinos=3,
        umbral_confianza=0.5,  # Bajo para ver early exits
        max_length=256
    )
    
    model = LLARRILanguageModel(config)
    model.print_arquitectura()
    
    # Test encoding
    print("\n" + "-"*40)
    print("TEST ENCODING:")
    print("-"*40)
    
    texto = "Hola mundo"
    embeddings, tokens = model.encode(texto, nivel=2)
    print(f"\nTexto: '{texto}'")
    print(f"Tokens: {tokens}")
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    
    # Test forward
    print("\n" + "-"*40)
    print("TEST FORWARD:")
    print("-"*40)
    
    output = model.forward(text="Hola mundo", nivel=2)
    print(f"\nLogits shape: {tuple(output['logits'].shape)}")
    print(f"Cajas ejecutadas: {output['bloque_info']['cajas_ejecutadas']}")
    print(f"Early exit: {output['bloque_info']['early_exit']}")
    print(f"Caja salida: {output['bloque_info']['caja_salida']}")
    print(f"Contribuciones: {output['bloque_info']['contribuciones']}")
    
    # Test forward con loss
    print("\n" + "-"*40)
    print("TEST FORWARD CON LOSS:")
    print("-"*40)
    
    tokens_dict = model.tokenizer.encode("Hola mundo", max_nivel=2)
    tokens = tokens_dict[2]
    input_ids = torch.tensor([tokens])
    labels = input_ids.clone()
    
    output = model.forward(input_ids=input_ids, nivel=2, labels=labels)
    print(f"\nLoss: {output['loss'].item():.4f}")
    
    # Test generación
    print("\n" + "-"*40)
    print("TEST GENERACIÓN:")
    print("-"*40)
    
    prompt = "Hola"
    print(f"\nPrompt: '{prompt}'")
    
    # Generación con sampling
    generated = model.generate(
        prompt=prompt,
        max_new_tokens=20,
        nivel=2,
        temperatura=0.8,
        top_k=10,
        do_sample=True
    )
    print(f"Generado (sampling): '{generated}'")
    
    # Generación greedy
    generated_greedy = model.generate(
        prompt=prompt,
        max_new_tokens=20,
        nivel=2,
        do_sample=False
    )
    print(f"Generado (greedy): '{generated_greedy}'")
    
    # Estadísticas del bloque fractal
    print("\n" + "-"*40)
    print("ESTADÍSTICAS BLOQUE FRACTAL:")
    print("-"*40)
    
    # Simular varias llamadas para ver early exit
    for _ in range(50):
        model.forward(text="Prueba de texto", nivel=2)
    
    model.bloque_fractal.print_stats()
    
    print("\n" + "="*70)
    print("✓ LLARRI LANGUAGE MODEL FUNCIONANDO")
    print("="*70)
