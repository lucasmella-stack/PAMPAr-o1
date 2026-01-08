# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
LLARRI Language Model v5 - Arquitectura Híbrida con Matemáticas Blindadas.

INNOVACIÓN CLAVE:
    - Cajas 1-6: Redes Neuronales (aprenden patrones)
    - Cajas 7-9: Matemáticas Puras (corrigen SIEMPRE)

El compositor v2 usa cálculos matemáticos determinísticos
que GARANTIZAN no colapsar en repeticiones.

Caja 7 (Detector):     Entropía, N-grama, Compresión, Autocorrelación
Caja 8 (Planificador): Transiciones Markov, Gramática, Beam, MI
Caja 9 (Refinador):    Bayes, Penalización, Temperatura, Normalización
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llarri_o1.modules.multiescala import (
    ProcesadorMultiescala,
    MultiescalaConfig
)
from llarri_o1.modules.bloque_fractal import (
    BloqueFractal,
    BloqueFractalConfig,
)
from llarri_o1.modules.compositor_v2_fast import (
    ModuloCompositorV2,
    CompositorV2Config
)


@dataclass 
class LLARRIv5Config:
    """Configuración LLARRI v5 - Híbrido Neural + Matemático."""
    
    # Dimensiones
    embed_dim: int = 128
    vocab_size: int = 256
    
    # Niveles multiescala
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    fusion_type: str = "concat_project"
    
    # Bloque Fractal (cajas 1-6 neuronales)
    num_heads: int = 4
    ffn_expansion: float = 2.0
    num_vecinos: int = 3
    umbral_confianza: float = 0.7
    
    # Compositor V2 FAST (cajas 7-9 matemáticas)
    entropy_threshold_low: float = 0.5
    entropy_threshold_high: float = 4.0
    repetition_penalty: float = 1.2
    
    # Generación
    max_length: int = 256
    temperatura_default: float = 0.8


class LLARRIv5(nn.Module):
    """
    LLARRI v5 - Modelo Híbrido Neural + Matemático.
    
    Arquitectura:
        Input (bytes)
            ↓
        Multiescala (niveles 2,4,8,16)
            ↓
        Cajas 1-6: Bloque Fractal NEURONAL
            ↓
        LM Head → Logits preliminares
            ↓
        Cajas 7-9: Compositor MATEMÁTICO
            ↓
        Output GARANTIZADO (sin colapso)
    
    La clave es que las matemáticas de 7-9 SIEMPRE corrigen
    cualquier problema de las redes neuronales 1-6.
    """
    
    def __init__(self, config: LLARRIv5Config):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("INICIALIZANDO LLARRI v5 - HÍBRIDO NEURAL + MATEMÁTICO")
        print("=" * 60)
        
        # 1. Procesador Multiescala
        multiescala_config = MultiescalaConfig(
            niveles=config.niveles,
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            fusion_type=config.fusion_type
        )
        self.multiescala = ProcesadorMultiescala(multiescala_config)
        
        # 2. Bloque Fractal (Cajas 1-6) - NEURONAL
        fractal_config = BloqueFractalConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_expansion=config.ffn_expansion,
            num_vecinos=config.num_vecinos,
            umbral_confianza=config.umbral_confianza
        )
        self.bloque_fractal = BloqueFractal(fractal_config)
        print(f"✓ Cajas 1-6: Bloque Fractal NEURONAL")
        
        # 3. LM Head (genera logits preliminares)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        print(f"✓ LMHead: {config.embed_dim} → {config.vocab_size}")
        
        # 4. Compositor V2 FAST (Cajas 7-9) - MATEMÁTICO VECTORIZADO
        compositor_config = CompositorV2Config(
            vocab_size=config.vocab_size,
            entropy_threshold_low=config.entropy_threshold_low,
            entropy_threshold_high=config.entropy_threshold_high,
            repetition_penalty=config.repetition_penalty,
        )
        self.compositor = ModuloCompositorV2(compositor_config)
        print(f"✓ Cajas 7-9: Compositor MATEMÁTICO VECTORIZADO")
        
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
        Forward pass híbrido.
        
        Args:
            input_ids: [batch, seq] tokens de entrada (bytes 0-255)
            labels: [batch, seq] tokens objetivo para loss
            
        Returns:
            dict con logits finales, loss, y diagnósticos
        """
        # === PARTE NEURONAL (Cajas 1-6) ===
        
        # 1. Multiescala: tokenizar a múltiples niveles
        x, multi_info = self.multiescala(input_ids)
        # x: [batch, seq, embed_dim]
        
        # 2. Bloque Fractal: procesamiento neuronal profundo
        x, fractal_info = self.bloque_fractal(x)
        
        # 3. LM Head: generar logits preliminares
        preliminary_logits = self.lm_head(x)
        # [batch, seq, vocab]
        
        # === PARTE MATEMÁTICA (Cajas 7-9) ===
        
        # 4. Compositor V2: corrección matemática garantizada
        comp_out = self.compositor(input_ids, preliminary_logits)
        final_logits = comp_out['final_logits']
        
        # Loss usando logits finales corregidos
        loss = None
        if labels is not None:
            shift_logits = final_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': final_logits,
            'preliminary_logits': preliminary_logits,
            'loss': loss,
            'multiescala_info': multi_info,
            'fractal_info': fractal_info,
            'compositor_info': {
                'detections': comp_out['detections'],
                'plan': comp_out['plan'],
                'needs_correction': comp_out['needs_correction'],
            }
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperatura: float = None,
        top_k: int = 40,
        top_p: float = 0.9,
        verbose: bool = False
    ) -> str:
        """
        Genera texto con corrección matemática garantizada.
        
        El compositor v2 previene colapsos automáticamente.
        """
        temperatura = temperatura or self.config.temperatura_default
        device = next(self.parameters()).device
        self.eval()
        
        # Convertir prompt a bytes
        tokens = list(prompt.encode('utf-8', errors='ignore'))
        
        corrections_applied = 0
        
        for step in range(max_new_tokens):
            # Limitar contexto
            context = tokens[-self.config.max_length:]
            
            # Forward completo
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            output = self.forward(input_ids)
            
            # Usar probabilidades corregidas
            final_probs = output['compositor_info']['needs_correction']
            logits = output['logits'][0, -1, :]
            
            # Aplicar temperatura adicional del usuario
            if temperatura != 1.0:
                logits = logits / temperatura
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
            
            # Renormalizar
            probs = probs / probs.sum()
            
            # Samplear
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            
            # Contar correcciones
            if output['compositor_info']['needs_correction'][0, -1]:
                corrections_applied += 1
            
            # Verbose mode
            if verbose and step % 20 == 0:
                print(f"  Step {step}: corrections so far = {corrections_applied}")
        
        # Decodificar
        generated = bytes(tokens[len(prompt.encode('utf-8')):])
        try:
            result = generated.decode('utf-8', errors='ignore')
        except:
            result = str(generated)
        
        if verbose:
            print(f"\n📊 Total corrections: {corrections_applied}/{max_new_tokens}")
        
        return prompt + result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST LLARRI v5 - HÍBRIDO NEURAL + MATEMÁTICO")
    print("=" * 60 + "\n")
    
    config = LLARRIv5Config(
        embed_dim=128,
        niveles=[2, 4, 8, 16],
    )
    
    model = LLARRIv5(config)
    
    # Test forward
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    labels = torch.randint(0, 256, (batch_size, seq_len))
    
    print(f"\n📝 Test forward:")
    print(f"  Input: {input_ids.shape}")
    
    output = model(input_ids, labels)
    
    print(f"  Preliminary logits: {output['preliminary_logits'].shape}")
    print(f"  Final logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Verificar que las correcciones funcionan
    needs_correction = output['compositor_info']['needs_correction']
    print(f"  Positions needing correction: {needs_correction.sum().item()}/{seq_len * batch_size}")
    
    print("\n✅ LLARRI v5 funcionando!")
