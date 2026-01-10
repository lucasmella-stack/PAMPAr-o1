# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Configuración del modelo PampaR.

Este archivo contiene:
- Config: Configuración legacy para MNIST (v4)
- ConfigPampaR: Configuración para modelo de lenguaje v8
- Presets escalables: LOCAL_4GB, SERVER_8GB, SERVER_24GB, SERVER_80GB
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    """
    Configuración del modelo PampaR v4.0 HyperComprimido (Legacy - MNIST).
    """
    input_dim: int = 784
    hidden_dim: int = 1024
    output_dim: int = 10
    num_cajas_datos: int = 3
    num_cajas_calculos: int = 3
    niveles_fractales: List[int] = None
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validaciones y defaults de configuración."""
        assert self.hidden_dim % 4 == 0, "hidden_dim debe ser divisible por 4"
        
        quad_dim = self.hidden_dim // 4
        
        if self.niveles_fractales is None:
            self.niveles_fractales = []
            nivel = 2
            while nivel <= quad_dim:
                self.niveles_fractales.append(nivel)
                nivel *= 2
        
        max_nivel = max(self.niveles_fractales)
        assert quad_dim >= max_nivel, (
            f"quad_dim ({quad_dim}) debe ser >= nivel máximo ({max_nivel}). "
            f"Aumenta hidden_dim a {max_nivel * 4} o reduce niveles_fractales."
        )


@dataclass
class ConfigPampaR:
    """
    Configuración del modelo PampaR - Modelo de Lenguaje.
    
    Arquitectura modular con:
    - 6 módulos especializados (Lenguaje, Lógica, Matemáticas, Patrones, Contexto, Creatividad)
    - Sistema de LLAVES (reglas explícitas) en el Tálamo
    - SINAPSIS (conexiones inter-módulo)
    - AXIOMAS (razonamiento deductivo)
    - Memoria práctica (aprendizaje de éxitos/fracasos)
    """
    # Modelo
    vocab_size: int = 8000
    dim: int = 128
    n_heads: int = 4
    n_capas: int = 3
    dropout: float = 0.1
    max_seq_len: int = 256
    
    # Tálamo - Sistema de LLAVES
    peso_llaves: float = 0.7  # 70% reglas, 30% aprendido
    
    # Axiomas - Razonamiento deductivo
    usar_axiomas: bool = True
    
    # Memoria práctica
    usar_memoria: bool = True
    capacidad_memoria: int = 200
    
    # Eficiencia (para escalar)
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False
    
    # Entrenamiento
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    
    def estimate_params(self) -> Dict[str, int]:
        """Estima el número de parámetros del modelo."""
        # Embeddings
        emb = self.vocab_size * self.dim + self.max_seq_len * self.dim
        
        # 6 módulos × n_capas bloques
        # Cada módulo: attention + ffn ≈ 4 * dim^2 + 8 * dim^2 = 12 * dim^2
        modulos = 6 * self.n_capas * 12 * self.dim * self.dim
        
        # Axiomas
        axiomas = 3 * 3 * self.dim * self.dim if self.usar_axiomas else 0
        
        # Memoria
        memoria = self.capacidad_memoria * self.dim * 2 if self.usar_memoria else 0
        
        # LM Head (comparte pesos con embeddings)
        lm_head = 0
        
        return {
            "embeddings": emb,
            "modulos": modulos,
            "axiomas": axiomas,
            "memoria": memoria,
            "lm_head": lm_head,
            "total": emb + modulos + axiomas + memoria + lm_head
        }
    
    def estimate_vram_gb(self, training: bool = True) -> float:
        """
        Estima VRAM necesaria en GB.
        
        Durante entrenamiento incluye:
        - Modelo (params * 4 bytes para FP32, o * 2 para FP16)
        - Gradientes (igual que modelo)
        - Optimizer states AdamW (2x params)
        - Activaciones (batch * seq * dim * n_capas * factor)
        
        Durante inferencia solo modelo + activaciones mínimas.
        """
        params = self.estimate_params()["total"]
        
        # Bytes por parámetro
        if self.use_mixed_precision:
            param_bytes = 2  # FP16
        else:
            param_bytes = 4  # FP32
        
        # Modelo
        model_bytes = params * param_bytes
        
        if training:
            # Gradientes (mismo tamaño que modelo)
            grad_bytes = params * param_bytes
            
            # Optimizer states (AdamW: momentum + variance = 2x)
            # Siempre FP32 para estabilidad
            optimizer_bytes = params * 4 * 2
            
            # Activaciones: batch * seq * dim * capas * 6_modulos * factor
            # Factor ~10 para attention + ffn intermedios
            activation_bytes = (
                self.batch_size * self.max_seq_len * self.dim * 
                self.n_capas * 6 * 10 * param_bytes
            )
            
            total_bytes = model_bytes + grad_bytes + optimizer_bytes + activation_bytes
        else:
            # Inferencia: solo modelo + activaciones mínimas
            activation_bytes = self.max_seq_len * self.dim * self.n_capas * 6 * param_bytes
            total_bytes = model_bytes + activation_bytes
        
        return total_bytes / (1024**3)


# =============================================================================
# PRESETS ESCALABLES
# =============================================================================

# Para GTX 1650 (4GB VRAM) - Desarrollo local
# CONSERVADOR para no quedarse sin VRAM
LOCAL_4GB = ConfigPampaR(
    vocab_size=8000,
    dim=128,
    n_heads=4,
    n_capas=3,
    dropout=0.1,
    max_seq_len=256,
    peso_llaves=0.7,
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=200,
    use_gradient_checkpointing=False,  # No necesario con este tamaño
    use_mixed_precision=True,  # FP16 para ahorrar VRAM
    batch_size=8,  # Batch pequeño para 4GB
    learning_rate=3e-4,
    max_epochs=20,
)

# Para GPU 8GB (RTX 3060/3070) - Servidor pequeño
SERVER_8GB = ConfigPampaR(
    vocab_size=16000,
    dim=256,
    n_heads=8,
    n_capas=4,
    dropout=0.1,
    max_seq_len=512,
    peso_llaves=0.6,
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=500,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=32,
    learning_rate=2e-4,
    max_epochs=30,
)

# Para GPU 24GB (RTX 3090/4090) - Servidor mediano
SERVER_24GB = ConfigPampaR(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_capas=6,
    dropout=0.1,
    max_seq_len=1024,
    peso_llaves=0.5,
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=1000,
    use_gradient_checkpointing=True,
    use_mixed_precision=True,
    batch_size=64,
    learning_rate=1e-4,
    max_epochs=50,
)

# Para A100 80GB - Servidor grande (causa justa)
SERVER_80GB = ConfigPampaR(
    vocab_size=50000,
    dim=768,  # Reducido para caber en 80GB
    n_heads=12,
    n_capas=8,
    dropout=0.1,
    max_seq_len=1024,
    peso_llaves=0.4,  # Más aprendizaje con más datos
    usar_axiomas=True,
    usar_memoria=True,
    capacidad_memoria=2000,
    use_gradient_checkpointing=True,  # Necesario
    use_mixed_precision=True,
    batch_size=32,  # Conservador
    learning_rate=5e-5,
    max_epochs=100,
)


def get_config_for_vram(vram_gb: float) -> ConfigPampaR:
    """Devuelve la configuración óptima para la VRAM disponible."""
    if vram_gb <= 4:
        return LOCAL_4GB
    elif vram_gb <= 8:
        return SERVER_8GB
    elif vram_gb <= 24:
        return SERVER_24GB
    else:
        return SERVER_80GB


def print_config_comparison():
    """Imprime comparación de todas las configuraciones."""
    configs = [
        ("LOCAL_4GB", LOCAL_4GB),
        ("SERVER_8GB", SERVER_8GB),
        ("SERVER_24GB", SERVER_24GB),
        ("SERVER_80GB", SERVER_80GB),
    ]
    
    print("\n" + "=" * 80)
    print("CONFIGURACIONES ESCALABLES PampaR")
    print("=" * 80)
    
    for name, cfg in configs:
        params = cfg.estimate_params()
        vram_train = cfg.estimate_vram_gb(training=True)
        vram_infer = cfg.estimate_vram_gb(training=False)
        print(f"\n{name}:")
        print(f"  Parámetros: {params['total']:,}")
        print(f"  VRAM entrenamiento: {vram_train:.2f} GB")
        print(f"  VRAM inferencia: {vram_infer:.2f} GB")
        print(f"  Dimensión: {cfg.dim}, Capas: {cfg.n_capas}, Heads: {cfg.n_heads}")
        print(f"  Vocab: {cfg.vocab_size:,}, Seq: {cfg.max_seq_len}, Batch: {cfg.batch_size}")


if __name__ == "__main__":
    print_config_comparison()
