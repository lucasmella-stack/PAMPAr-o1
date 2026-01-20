# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas.mella@outlook.com)
"""
Módulo Compositor - Cajas 7, 8, 9 de LLARRI v4.

Las cajas compositoras añaden capacidad de "razonamiento" al modelo:
- Caja 7 (Detector): Analiza patrones y contexto
- Caja 8 (Planificador): Decide qué tipo de token debería seguir
- Caja 9 (Refinador): Ajusta probabilidades para coherencia

Cada caja usa 4 cuadrantes especializados, igual que las cajas fractales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompositorConfig:
    """Configuración del módulo compositor."""
    embed_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    vocab_size: int = 256
    
    # Ventana de contexto para detección de patrones
    pattern_window: int = 16
    
    # Umbral para detección de repetición
    repetition_threshold: float = 0.7


class CuadranteDetector(nn.Module):
    """
    Cuadrante especializado en detectar un tipo de patrón.
    
    Tipos:
    - palabra_estado: ¿completa o parcial?
    - tipo_token: ¿letra, espacio, puntuación, número?
    - repeticion: ¿hay loop?
    - contexto: ¿qué vino antes semánticamente?
    """
    
    def __init__(self, embed_dim: int, detection_type: str):
        super().__init__()
        self.detection_type = detection_type
        
        self.analyzer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Salida de detección (4 categorías por tipo)
        self.classifier = nn.Linear(embed_dim, 4)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim]
        Returns:
            features: [batch, seq, embed_dim] - features procesadas
            detection: [batch, seq, 4] - probabilidades de cada categoría
        """
        features = self.analyzer(x)
        detection = self.classifier(features)
        return features, F.softmax(detection, dim=-1)


class CajaDetectora(nn.Module):
    """
    Caja 7: Detecta patrones y estado del contexto.
    
    4 cuadrantes:
    - Q1: Estado de palabra (completa/parcial/inicio/fin)
    - Q2: Tipo de token actual (letra/espacio/punt/num)
    - Q3: Detector de repetición
    - Q4: Contexto semántico
    """
    
    def __init__(self, config: CompositorConfig):
        super().__init__()
        self.config = config
        
        # 4 cuadrantes detectores
        self.cuadrantes = nn.ModuleDict({
            'palabra_estado': CuadranteDetector(config.embed_dim, 'palabra'),
            'tipo_token': CuadranteDetector(config.embed_dim, 'tipo'),
            'repeticion': CuadranteDetector(config.embed_dim, 'repeticion'),
            'contexto': CuadranteDetector(config.embed_dim, 'contexto'),
        })
        
        # Fusión de detecciones
        self.fusion = nn.Linear(config.embed_dim * 4, config.embed_dim)
        
        # Atención para contexto
        self.self_attn = nn.MultiheadAttention(
            config.embed_dim, 
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim]
        Returns:
            dict con features procesadas y detecciones
        """
        # Procesar cada cuadrante
        features_list = []
        detections = {}
        
        for nombre, cuadrante in self.cuadrantes.items():
            feat, det = cuadrante(x)
            features_list.append(feat)
            detections[nombre] = det
        
        # Fusionar features
        combined = torch.cat(features_list, dim=-1)
        fused = self.fusion(combined)
        
        # Self-attention para contexto
        fused = self.norm1(fused)
        attn_out, _ = self.self_attn(fused, fused, fused)
        output = self.norm2(x + attn_out)
        
        return {
            'features': output,
            'detections': detections,
        }


class CuadrantePlanificador(nn.Module):
    """
    Cuadrante que planifica basado en un aspecto.
    """
    
    def __init__(self, embed_dim: int, plan_type: str):
        super().__init__()
        self.plan_type = plan_type
        
        self.planner = nn.Sequential(
            nn.Linear(embed_dim + 16, embed_dim),  # +16 para detections
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Plan de acción (4 tipos)
        self.action = nn.Linear(embed_dim, 4)
        
    def forward(
        self, 
        x: torch.Tensor, 
        detections: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim]
            detections: [batch, seq, 16] - todas las detecciones concatenadas
        Returns:
            features, plan
        """
        combined = torch.cat([x, detections], dim=-1)
        features = self.planner(combined)
        plan = self.action(features)
        return features, F.softmax(plan, dim=-1)


class CajaPlanificadora(nn.Module):
    """
    Caja 8: Planifica qué tipo de token debería seguir.
    
    4 cuadrantes:
    - Q1: Completar palabra actual
    - Q2: Terminar palabra (espacio/punt)
    - Q3: Nuevo concepto
    - Q4: Mantener estructura gramatical
    """
    
    def __init__(self, config: CompositorConfig):
        super().__init__()
        self.config = config
        
        self.cuadrantes = nn.ModuleDict({
            'completar': CuadrantePlanificador(config.embed_dim, 'completar'),
            'terminar': CuadrantePlanificador(config.embed_dim, 'terminar'),
            'nuevo': CuadrantePlanificador(config.embed_dim, 'nuevo'),
            'gramatica': CuadrantePlanificador(config.embed_dim, 'gramatica'),
        })
        
        # Fusión de planes
        self.fusion = nn.Linear(config.embed_dim * 4, config.embed_dim)
        
        # Selector de plan dominante
        self.plan_selector = nn.Linear(config.embed_dim, 4)
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        detections: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim]
            detections: dict de detecciones de Caja 7
        Returns:
            dict con features y planes
        """
        # Concatenar todas las detecciones
        det_concat = torch.cat(list(detections.values()), dim=-1)  # [batch, seq, 16]
        
        # Procesar cada cuadrante
        features_list = []
        plans = {}
        
        for nombre, cuadrante in self.cuadrantes.items():
            feat, plan = cuadrante(x, det_concat)
            features_list.append(feat)
            plans[nombre] = plan
        
        # Fusionar
        combined = torch.cat(features_list, dim=-1)
        fused = self.fusion(combined)
        
        # Plan dominante
        dominant_plan = F.softmax(self.plan_selector(fused), dim=-1)
        
        output = self.norm(x + fused)
        
        return {
            'features': output,
            'plans': plans,
            'dominant_plan': dominant_plan,
        }


class CuadranteRefinador(nn.Module):
    """
    Cuadrante que refina logits basado en un criterio.
    """
    
    def __init__(self, embed_dim: int, vocab_size: int, refine_type: str):
        super().__init__()
        self.refine_type = refine_type
        
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim + 20, embed_dim),  # +20 para detections + plans
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Ajuste de logits (bias por token)
        self.logit_bias = nn.Linear(embed_dim, vocab_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features, logit_adjustment
        """
        combined = torch.cat([x, context], dim=-1)
        features = self.refiner(combined)
        bias = self.logit_bias(features)
        return features, bias


class CajaRefinadora(nn.Module):
    """
    Caja 9: Refina las probabilidades finales.
    
    4 cuadrantes:
    - Q1: Boost a tokens que completan palabras
    - Q2: Penalizar repeticiones
    - Q3: Ajuste gramatical
    - Q4: Coherencia semántica
    """
    
    def __init__(self, config: CompositorConfig):
        super().__init__()
        self.config = config
        
        self.cuadrantes = nn.ModuleDict({
            'completar_boost': CuadranteRefinador(
                config.embed_dim, config.vocab_size, 'completar'
            ),
            'repeticion_penalty': CuadranteRefinador(
                config.embed_dim, config.vocab_size, 'repeticion'
            ),
            'gramatica_ajuste': CuadranteRefinador(
                config.embed_dim, config.vocab_size, 'gramatica'
            ),
            'semantica': CuadranteRefinador(
                config.embed_dim, config.vocab_size, 'semantica'
            ),
        })
        
        # Pesos de cada refinador
        self.refiner_weights = nn.Linear(config.embed_dim, 4)
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        detections: Dict[str, torch.Tensor],
        plans: Dict[str, torch.Tensor],
        dominant_plan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim]
            detections: de Caja 7
            plans: de Caja 8
            dominant_plan: [batch, seq, 4]
        Returns:
            dict con features refinadas y ajustes de logits
        """
        # Contexto completo
        det_concat = torch.cat(list(detections.values()), dim=-1)  # [batch, seq, 16]
        plan_concat = torch.cat([dominant_plan], dim=-1)  # [batch, seq, 4]
        context = torch.cat([det_concat, plan_concat], dim=-1)  # [batch, seq, 20]
        
        # Procesar cada cuadrante
        features_list = []
        biases = []
        
        for nombre, cuadrante in self.cuadrantes.items():
            feat, bias = cuadrante(x, context)
            features_list.append(feat)
            biases.append(bias)
        
        # Pesos para cada refinador
        weights = F.softmax(self.refiner_weights(x), dim=-1)  # [batch, seq, 4]
        
        # Combinar biases ponderados
        biases_stack = torch.stack(biases, dim=-1)  # [batch, seq, vocab, 4]
        weighted_bias = (biases_stack * weights.unsqueeze(-2)).sum(dim=-1)  # [batch, seq, vocab]
        
        # Features
        combined = sum(features_list) / len(features_list)
        output = self.norm(x + combined)
        
        return {
            'features': output,
            'logit_bias': weighted_bias,
            'refiner_weights': weights,
        }


class ModuloCompositor(nn.Module):
    """
    Módulo completo de composición: Cajas 7, 8, 9.
    
    Recibe las features del Bloque Fractal (cajas 1-6)
    y añade capacidad de razonamiento/planificación.
    """
    
    def __init__(self, config: CompositorConfig):
        super().__init__()
        self.config = config
        
        # Las 3 cajas compositoras
        self.caja_7_detector = CajaDetectora(config)
        self.caja_8_planificador = CajaPlanificadora(config)
        self.caja_9_refinador = CajaRefinadora(config)
        
        print(f"✓ ModuloCompositor: 3 cajas (7-8-9)")
        print(f"  Cada caja: 4 cuadrantes especializados")
        print(f"  Vocab size: {config.vocab_size}")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq, embed_dim] - salida del Bloque Fractal
        Returns:
            dict con features refinadas y logit_bias
        """
        # Caja 7: Detectar
        det_out = self.caja_7_detector(x)
        
        # Caja 8: Planificar
        plan_out = self.caja_8_planificador(
            det_out['features'], 
            det_out['detections']
        )
        
        # Caja 9: Refinar
        refine_out = self.caja_9_refinador(
            plan_out['features'],
            det_out['detections'],
            plan_out['plans'],
            plan_out['dominant_plan']
        )
        
        return {
            'features': refine_out['features'],
            'logit_bias': refine_out['logit_bias'],
            'detections': det_out['detections'],
            'plans': plan_out['plans'],
            'dominant_plan': plan_out['dominant_plan'],
            'refiner_weights': refine_out['refiner_weights'],
        }


# Test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST MÓDULO COMPOSITOR (Cajas 7-8-9)")
    print("=" * 60 + "\n")
    
    config = CompositorConfig(
        embed_dim=128,
        vocab_size=256
    )
    
    compositor = ModuloCompositor(config)
    
    # Contar parámetros
    params = sum(p.numel() for p in compositor.parameters())
    print(f"\nParámetros compositor: {params:,}")
    
    # Test forward
    x = torch.randn(2, 64, 128)  # batch=2, seq=64, embed=128
    
    output = compositor(x)
    
    print(f"\nTest forward:")
    print(f"  Input: {x.shape}")
    print(f"  Features: {output['features'].shape}")
    print(f"  Logit bias: {output['logit_bias'].shape}")
    print(f"  Dominant plan: {output['dominant_plan'].shape}")
    
    print("\n✅ Módulo Compositor funcionando!")
