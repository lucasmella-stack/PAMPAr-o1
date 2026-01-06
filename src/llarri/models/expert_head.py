"""
expert_head.py - Módulos especializados (Expert Heads) para fine-tuning

Este módulo implementa diferentes estrategias para crear expertos especializados:
1. Adapter Layers: Capas pequeñas insertadas en el modelo base
2. LoRA (Low-Rank Adaptation): Matrices de bajo rango para fine-tuning eficiente
3. Expert Decoder: Decoder completo especializado que se conecta al encoder base

Uso:
    # Adapter approach
    expert = AdapterExpertHead(hidden_size=768, adapter_size=64)
    
    # LoRA approach
    expert = LoRAExpertHead(hidden_size=768, rank=8)
    
    # Full decoder approach
    expert = ExpertDecoder(encoder_hidden_size=768, vocab_size=50000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import TrOCRForCausalLM, TrOCRConfig


class AdapterLayer(nn.Module):
    """
    Adapter layer según Houlsby et al. (2019).
    
    Arquitectura: down_project → activation → up_project + residual
    """
    def __init__(self, hidden_size: int, adapter_size: int, activation: str = "gelu"):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Inicialización: empezar cerca de identidad
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual connection."""
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer según Hu et al. (2021).
    
    Descompone W en W + BA donde B y A son matrices de bajo rango.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 8, 
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices de bajo rango
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout opcional
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Inicialización
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        # x @ A @ B con scaling
        result = self.dropout(x @ self.lora_A)
        result = result @ self.lora_B
        return result * self.scaling


class AdapterExpertHead(nn.Module):
    """
    Expert head basado en Adapter layers.
    
    Inserta adapters después de cada capa del decoder.
    """
    def __init__(
        self, 
        base_decoder: TrOCRForCausalLM,
        adapter_size: int = 64,
        num_adapter_layers: int = 6,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.adapter_size = adapter_size
        
        # Congelar modelo base si se requiere
        if freeze_base:
            for param in self.base_decoder.parameters():
                param.requires_grad = False
        
        # Crear adapters
        hidden_size = base_decoder.config.hidden_size
        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_size, adapter_size)
            for _ in range(num_adapter_layers)
        ])
        
        print(f"✅ AdapterExpertHead creado con {num_adapter_layers} adapters (size={adapter_size})")
        print(f"   Parámetros entrenables: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, encoder_hidden_states, labels=None, **kwargs):
        """Forward pass con adapters."""
        # TODO: Integrar adapters en las capas del decoder
        # Por ahora, usamos el decoder base directamente
        return self.base_decoder(
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            **kwargs
        )
    
    def generate(self, encoder_hidden_states, **kwargs):
        """Generation con adapters."""
        return self.base_decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )


class LoRAExpertHead(nn.Module):
    """
    Expert head basado en LoRA.
    
    Añade matrices de bajo rango a las capas attention del decoder.
    """
    def __init__(
        self,
        base_decoder: TrOCRForCausalLM,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[list] = None,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.rank = rank
        self.alpha = alpha
        
        # Congelar modelo base
        if freeze_base:
            for param in self.base_decoder.parameters():
                param.requires_grad = False
        
        # Módulos objetivo (por defecto: query y value de attention)
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        # Aplicar LoRA a módulos objetivo
        self.lora_layers = nn.ModuleDict()
        total_params = 0
        
        for name, module in self.base_decoder.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha
                    )
                    self.lora_layers[name.replace('.', '_')] = lora
                    total_params += sum(p.numel() for p in lora.parameters())
        
        print(f"✅ LoRAExpertHead creado con rank={rank}, alpha={alpha}")
        print(f"   LoRA aplicado a {len(self.lora_layers)} módulos")
        print(f"   Parámetros LoRA: {total_params:,}")
        print(f"   Parámetros entrenables totales: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, encoder_hidden_states, labels=None, **kwargs):
        """Forward pass con LoRA."""
        # TODO: Integrar LoRA en el forward pass
        # Por ahora, usamos el decoder base directamente
        return self.base_decoder(
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            **kwargs
        )
    
    def generate(self, encoder_hidden_states, **kwargs):
        """Generation con LoRA."""
        return self.base_decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )


class ExpertDecoder(nn.Module):
    """
    Decoder completo especializado.
    
    Usa un decoder TrOCR independiente, inicializado desde el modelo base.
    Permite fine-tuning completo o parcial.
    """
    def __init__(
        self,
        base_decoder: TrOCRForCausalLM,
        freeze_embeddings: bool = True,
        freeze_n_layers: int = 0,
    ):
        super().__init__()
        
        # Clonar configuración del decoder base
        self.config = base_decoder.config
        
        # Crear nuevo decoder con los pesos del base
        self.decoder = TrOCRForCausalLM(self.config)
        self.decoder.load_state_dict(base_decoder.state_dict())
        
        # Estrategia de congelamiento
        if freeze_embeddings:
            # Congelar embeddings
            for param in self.decoder.model.decoder.embed_tokens.parameters():
                param.requires_grad = False
            if hasattr(self.decoder.model.decoder, 'embed_positions'):
                for param in self.decoder.model.decoder.embed_positions.parameters():
                    param.requires_grad = False
        
        if freeze_n_layers > 0:
            # Congelar primeras N capas del decoder
            for i in range(min(freeze_n_layers, len(self.decoder.model.decoder.layers))):
                for param in self.decoder.model.decoder.layers[i].parameters():
                    param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"✅ ExpertDecoder creado")
        print(f"   Embeddings congelados: {freeze_embeddings}")
        print(f"   Capas congeladas: {freeze_n_layers}")
        print(f"   Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def forward(self, encoder_hidden_states, labels=None, **kwargs):
        """Forward pass del decoder especializado."""
        return self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            **kwargs
        )
    
    def generate(self, encoder_hidden_states, **kwargs):
        """Generation del decoder especializado."""
        return self.decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )


class ExpertHead(nn.Module):
    """
    Wrapper unificado para diferentes tipos de expert heads.
    
    Factory class que crea el tipo apropiado de expert head.
    """
    def __init__(
        self,
        base_decoder: TrOCRForCausalLM,
        expert_type: str = "adapter",
        **kwargs
    ):
        super().__init__()
        
        self.expert_type = expert_type
        
        if expert_type == "adapter":
            self.expert = AdapterExpertHead(base_decoder, **kwargs)
        elif expert_type == "lora":
            self.expert = LoRAExpertHead(base_decoder, **kwargs)
        elif expert_type == "full":
            self.expert = ExpertDecoder(base_decoder, **kwargs)
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")
    
    def forward(self, encoder_hidden_states, labels=None, **kwargs):
        return self.expert.forward(encoder_hidden_states, labels, **kwargs)
    
    def generate(self, encoder_hidden_states, **kwargs):
        return self.expert.generate(encoder_hidden_states, **kwargs)
    
    @property
    def config(self):
        """Acceso a la configuración del decoder."""
        if hasattr(self.expert, 'config'):
            return self.expert.config
        elif hasattr(self.expert, 'decoder'):
            return self.expert.decoder.config
        elif hasattr(self.expert, 'base_decoder'):
            return self.expert.base_decoder.config
        return None

