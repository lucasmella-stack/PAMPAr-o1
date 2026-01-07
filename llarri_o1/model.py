# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Modelo principal LLARRI-O1 v4.0 HyperComprimido.
"""

import torch
import torch.nn as nn
from typing import Optional

from llarri_o1.config import Config
from llarri_o1.modules.cache import CacheBinario
from llarri_o1.modules.niveles import CuadranteProgresivo
from llarri_o1.modules.cajas import CajaDatos, CajaCalculos
from llarri_o1.modules.flujo import SistemaFlujoCompleto


class LlarriO1(nn.Module):
    """
    LLARRI-O1 v4.0 HyperComprimido
    
    Arquitectura fractal con:
        - 6 cajas: 3 de datos (A, B, C) + 3 de cálculos (D, E, F)
        - 8 niveles fractales: 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256
        - Cache binario para nivel 2 (lookup instantáneo)
        - Flujo completo: IDA + VUELTA + BIDIRECCIONAL
        - Retroalimentación: cálculos influyen en datos
    
    Example:
        >>> from llarri_o1 import LlarriO1, Config
        >>> config = Config(hidden_dim=1024)
        >>> model = LlarriO1(config)
        >>> x = torch.randn(32, 784)  # Batch de 32 imágenes MNIST
        >>> output = model(x)  # Shape: (32, 10)
    
    Author: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
    Coordinator: Alvaro (Segunda Cabeza)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: Configuración del modelo (usa default si None)
        """
        super().__init__()
        self.config = config or Config()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache binario compartido
        self.cache_binario = CacheBinario(device)
        
        # Cuadrante compartido (TODOS los cuadrantes usan los mismos pesos)
        quad_dim = self.config.hidden_dim // 4
        self.cuadrante_compartido = CuadranteProgresivo(
            dim_cuadrante=quad_dim,
            niveles=self.config.niveles_fractales,
            dropout=self.config.dropout,
            cache_binario=self.cache_binario
        )
        
        # 3 Cajas de DATOS (A, B, C)
        self.cajas_datos = nn.ModuleList([
            CajaDatos(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                cuadrante=self.cuadrante_compartido
            )
            for _ in range(self.config.num_cajas_datos)
        ])
        
        # 3 Cajas de CÁLCULOS (D, E, F)
        self.cajas_calculos = nn.ModuleList([
            CajaCalculos(
                hidden_dim=self.config.hidden_dim,
                cuadrante=self.cuadrante_compartido
            )
            for _ in range(self.config.num_cajas_calculos)
        ])
        
        # Sistema de flujo completo (IDA + VUELTA + BIDI)
        total_cajas = self.config.num_cajas_datos + self.config.num_cajas_calculos
        self.flujo_completo = SistemaFlujoCompleto(
            dim=self.config.hidden_dim,
            num_cajas=total_cajas
        )
        
        # Retroalimentación: cálculos -> datos
        self.retro = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
            for _ in range(self.config.num_cajas_datos)
        ])
        self.retro_norm = nn.LayerNorm(self.config.hidden_dim)
        
        # Capa de salida
        dim_fusion = self.config.hidden_dim * total_cajas
        self.output_layer = nn.Sequential(
            nn.Linear(dim_fusion, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )
        
        # Info
        self._print_info()
    
    def _print_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calcular relaciones teóricas
        n_datos = self.config.num_cajas_datos
        n_calc = self.config.num_cajas_calculos
        niveles = len(self.config.niveles_fractales)
        relaciones = (n_datos + n_calc) * 4 * niveles * (2 ** niveles)
        
        compression = relaciones / total_params if total_params > 0 else 0
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           LLARRI-O1 v4.0 - HYPERCOMPRIMIDO                   ║
║                  (FLUJO COMPLETO)                            ║
╠══════════════════════════════════════════════════════════════╣
║  Cajas Datos:    {n_datos}  (A, B, C)                              ║
║  Cajas Cálculos: {n_calc}  (D, E, F)                              ║
║  Niveles:        {niveles} ({self.config.niveles_fractales})          
║  Hidden dim:     {self.config.hidden_dim}                                    ║
║  Flujo:          IDA → VUELTA ← BIDI ↔                       ║
╠══════════════════════════════════════════════════════════════╣
║  Parámetros:     {total_params:,}                             
║  Entrenables:    {trainable:,}                              
║  Compresión:     {compression:.1f}%                                     
╚══════════════════════════════════════════════════════════════╝
""")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada (batch, input_dim)
            
        Returns:
            Logits de clasificación (batch, output_dim)
        """
        # Asegurar cache en el dispositivo correcto
        if self.cache_binario.device != x.device:
            self.cache_binario.to(x.device)
        
        # ===== CAJAS DE DATOS (A, B, C) =====
        datos = []
        h = x
        for caja in self.cajas_datos:
            h = caja(h)
            datos.append(h)
        
        # ===== CAJAS DE CÁLCULOS (D, E, F) =====
        # D = f(A, B)
        # E = f(B, C, D)
        # F = f(C, A, E)
        calc_D = self.cajas_calculos[0](datos[0], datos[1], None)
        calc_E = self.cajas_calculos[1](datos[1], datos[2], calc_D)
        calc_F = self.cajas_calculos[2](datos[2], datos[0], calc_E)
        
        calculos = [calc_D, calc_E, calc_F]
        
        # ===== FLUJO COMPLETO (IDA + VUELTA + BIDI) =====
        todas_las_cajas = datos + calculos  # [A, B, C, D, E, F]
        todas_las_cajas = self.flujo_completo(todas_las_cajas)
        
        # Separar después del flujo
        datos = todas_las_cajas[:3]
        calculos = todas_las_cajas[3:]
        
        # ===== RETROALIMENTACIÓN =====
        # Los cálculos influyen de vuelta en los datos
        for i in range(len(datos)):
            retro_signal = self.retro[i](calculos[i % len(calculos)])
            datos[i] = self.retro_norm(datos[i] + retro_signal * 0.3)
        
        # ===== FUSION FINAL =====
        fusion = torch.cat(datos + calculos, dim=-1)
        
        return self.output_layer(fusion)
    
    def to(self, device):
        """Override to también para mover el cache."""
        super().to(device)
        self.cache_binario.to(device)
        return self
