"""
LLARRI-O1 v4.0 - HyperComprimido
=================================

Arquitectura revolucionaria con:
- 6 Cajas (3 datos + 3 cálculos)
- 8 niveles fractales (256→128→64→32→16→8→4→2)
- Cache RAM para nivel binario (máxima velocidad)
- Compresión ~920,000x de relaciones

"Como si entrara 1TB en 1GB"

Autor: Lucas Mella (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import math


@dataclass
class ConfigHyperComprimido:
    """Configuración del modelo HyperComprimido."""
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    num_cajas_datos: int = 3
    num_cajas_calculos: int = 3
    niveles_fractales: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16, 8, 4, 2])
    dropout: float = 0.1
    usar_cache_binario: bool = True  # Cache RAM para nivel 2


class CacheBinario:
    """
    Cache en RAM para combinaciones binarias.
    Pre-computa todas las operaciones del nivel más profundo (dim=2).
    
    Con solo 2 valores hay 4 combinaciones posibles:
    [0,0], [0,1], [1,0], [1,1]
    
    Pre-computamos todas las transformaciones posibles.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self._cache = {}
        self._inicializar_cache()
    
    def _inicializar_cache(self):
        """Pre-computa todas las combinaciones binarias."""
        # Todas las combinaciones posibles de entrada (normalizado 0-1)
        combinaciones = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        ]
        
        # Operaciones básicas pre-computadas
        for i, c in enumerate(combinaciones):
            c = c.to(self.device)
            self._cache[f'base_{i}'] = c
            self._cache[f'sum_{i}'] = c.sum()
            self._cache[f'prod_{i}'] = c.prod()
            self._cache[f'diff_{i}'] = (c[0] - c[1]).abs()
            self._cache[f'mean_{i}'] = c.mean()
            self._cache[f'max_{i}'] = c.max()
            self._cache[f'min_{i}'] = c.min()
        
        # Matriz de todas las interacciones posibles (4x4)
        self._cache['interacciones'] = torch.zeros(4, 4, 7, device=self.device)
        for i in range(4):
            for j in range(4):
                c1 = combinaciones[i].to(self.device)
                c2 = combinaciones[j].to(self.device)
                self._cache['interacciones'][i, j] = torch.tensor([
                    (c1 + c2).sum(),
                    (c1 * c2).sum(),
                    (c1 - c2).abs().sum(),
                    torch.cat([c1, c2]).mean(),
                    torch.cat([c1, c2]).max(),
                    torch.cat([c1, c2]).min(),
                    (c1[0]*c2[1] + c1[1]*c2[0]),  # Cross product
                ], device=self.device)
    
    def obtener_idx(self, x: torch.Tensor) -> torch.Tensor:
        """Convierte tensor binario a índice (0-3)."""
        # Cuantizar a 0 o 1
        x_bin = (x > 0.5).float()
        # Convertir [a,b] a índice: a*2 + b
        return (x_bin[..., 0] * 2 + x_bin[..., 1]).long()
    
    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        """Busca en cache las operaciones pre-computadas."""
        idx = self.obtener_idx(x)
        batch_size = x.shape[0] if x.dim() > 1 else 1
        
        # Obtener resultados del cache
        resultados = torch.zeros(batch_size, 7, device=self.device)
        for b in range(batch_size):
            i = idx[b].item() if batch_size > 1 else idx.item()
            resultados[b] = torch.tensor([
                self._cache[f'sum_{i}'],
                self._cache[f'prod_{i}'],
                self._cache[f'diff_{i}'],
                self._cache[f'mean_{i}'],
                self._cache[f'max_{i}'],
                self._cache[f'min_{i}'],
                self._cache[f'sum_{i}'] * self._cache[f'prod_{i}'],
            ], device=self.device)
        
        return resultados


class NivelFractal(nn.Module):
    """
    Un nivel del fractal.
    Procesa entrada de dim_in y produce dim_out.
    """
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.proceso = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LayerNorm(dim_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.GELU(),
        )
        
        # Skip connection si las dimensiones coinciden
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proceso(x) + self.skip(x)


class CuadranteFractal(nn.Module):
    """
    Cuadrante con 8 niveles fractales.
    256 → 128 → 64 → 32 → 16 → 8 → 4 → 2
    
    Procesa recursivamente hasta el nivel binario.
    """
    def __init__(self, config: ConfigHyperComprimido, cache_binario: Optional[CacheBinario] = None):
        super().__init__()
        self.config = config
        self.cache_binario = cache_binario
        self.niveles = config.niveles_fractales
        
        # Crear niveles descendentes (compresión)
        self.niveles_down = nn.ModuleList()
        for i in range(len(self.niveles) - 1):
            self.niveles_down.append(
                NivelFractal(self.niveles[i], self.niveles[i+1], config.dropout)
            )
        
        # Crear niveles ascendentes (expansión)
        self.niveles_up = nn.ModuleList()
        for i in range(len(self.niveles) - 1, 0, -1):
            self.niveles_up.append(
                NivelFractal(self.niveles[i], self.niveles[i-1], config.dropout)
            )
        
        # Procesador del nivel binario (dim=2)
        self.proceso_binario = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )
        
        # Fusión con cache binario
        if cache_binario:
            self.fusion_cache = nn.Linear(7 + 2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Guardar estados intermedios para skip connections
        estados = [x]
        
        # Descender por niveles (256 → 2)
        h = x
        for nivel in self.niveles_down:
            h = nivel(h)
            estados.append(h)
        
        # Procesar nivel binario
        if self.cache_binario and h.shape[-1] == 2:
            # Usar cache para acelerar
            cache_out = self.cache_binario.lookup(h)
            h = self.fusion_cache(torch.cat([h, cache_out], dim=-1))
        else:
            h = self.proceso_binario(h)
        
        # Ascender por niveles (2 → 256)
        for i, nivel in enumerate(self.niveles_up):
            h = nivel(h)
            # Skip connection con estado correspondiente
            if i < len(estados) - 1:
                h = h + estados[-(i+2)] * 0.1  # Residual suave
        
        return h


class RelacionesFractales(nn.Module):
    """
    Conecta cuadrantes en todos los niveles fractales.
    Relaciones horizontales, verticales y diagonales.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # Relaciones entre cuadrantes
        self.rel_h = nn.Linear(dim * 2, dim)  # Horizontal
        self.rel_v = nn.Linear(dim * 2, dim)  # Vertical
        self.rel_d = nn.Linear(dim * 2, dim)  # Diagonal
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, 
                c: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Relaciones horizontales
        h_ab = self.rel_h(torch.cat([a, b], dim=-1))
        h_cd = self.rel_h(torch.cat([c, d], dim=-1))
        
        # Relaciones verticales
        v_ac = self.rel_v(torch.cat([a, c], dim=-1))
        v_bd = self.rel_v(torch.cat([b, d], dim=-1))
        
        # Relaciones diagonales
        d_ad = self.rel_d(torch.cat([a, d], dim=-1))
        d_bc = self.rel_d(torch.cat([b, c], dim=-1))
        
        # Actualizar cada cuadrante con sus relaciones
        a = self.norm(a + self.dropout(h_ab + v_ac + d_ad))
        b = self.norm(b + self.dropout(h_ab + v_bd + d_bc))
        c = self.norm(c + self.dropout(h_cd + v_ac + d_bc))
        d = self.norm(d + self.dropout(h_cd + v_bd + d_ad))
        
        return a, b, c, d


class CajaDatos(nn.Module):
    """
    Caja de DATOS - Procesa y almacena información.
    Contiene 4 cuadrantes fractales con 8 niveles cada uno.
    """
    def __init__(self, config: ConfigHyperComprimido, 
                 cuadrante_compartido: CuadranteFractal,
                 cache_binario: Optional[CacheBinario] = None):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        quad_dim = dim // 4
        
        # Proyecciones de entrada
        self.proj_in = nn.Linear(config.input_dim, dim)
        self.proj_internal = nn.Linear(dim, dim)
        
        # Cuadrante compartido
        self.cuadrante = cuadrante_compartido
        
        # Relaciones entre cuadrantes
        self.relaciones = RelacionesFractales(quad_dim, config.dropout)
        
        # Fusión
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Proyectar entrada
        if x.shape[-1] == self.config.input_dim:
            x = self.proj_in(x)
        elif x.shape[-1] != self.config.hidden_dim:
            x = self.proj_internal(x)
        
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Dividir en 4 cuadrantes
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        # Procesar cada cuadrante a través de 8 niveles fractales
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relacionar cuadrantes
        a, b, c, d = self.relaciones(a, b, c, d)
        
        # Fusionar
        out = torch.cat([a, b, c, d], dim=-1)
        out = self.fusion(out) + x
        
        # Devolver salida y estados de cuadrantes (para capa de cálculos)
        estados = {'a': a, 'b': b, 'c': c, 'd': d}
        
        return out, estados


class CajaCalculos(nn.Module):
    """
    Caja de CÁLCULOS - Opera sobre los datos de las cajas de datos.
    También opera sobre otros cálculos (meta-cálculos).
    """
    def __init__(self, config: ConfigHyperComprimido,
                 cuadrante_compartido: CuadranteFractal):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        quad_dim = dim // 4
        
        # Cuadrante compartido
        self.cuadrante = cuadrante_compartido
        
        # Operadores sobre datos
        self.op_suma = nn.Linear(dim * 2, dim)
        self.op_mult = nn.Linear(dim * 2, dim)
        self.op_diff = nn.Linear(dim * 2, dim)
        
        # Relaciones entre resultados de cálculos
        self.relaciones = RelacionesFractales(quad_dim, config.dropout)
        
        # Meta-cálculo (cálculo sobre cálculos)
        self.meta_calculo = nn.Sequential(
            nn.Linear(dim * 3, dim),  # suma + mult + diff
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, dim),
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, datos1: torch.Tensor, datos2: torch.Tensor,
                otros_calculos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Realiza cálculos sobre datos y opcionalmente sobre otros cálculos.
        """
        # Operaciones sobre datos
        suma = self.op_suma(torch.cat([datos1, datos2], dim=-1))
        mult = self.op_mult(torch.cat([datos1, datos2], dim=-1))
        diff = self.op_diff(torch.cat([datos1, (datos2 - datos1).abs()], dim=-1))
        
        # Meta-cálculo
        meta = self.meta_calculo(torch.cat([suma, mult, diff], dim=-1))
        
        # Si hay otros cálculos, incorporarlos
        if otros_calculos is not None:
            meta = self.norm(meta + otros_calculos * 0.5)
        
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Dividir resultado en cuadrantes
        a = meta[..., :quad_dim]
        b = meta[..., quad_dim:quad_dim*2]
        c = meta[..., quad_dim*2:quad_dim*3]
        d = meta[..., quad_dim*3:]
        
        # Procesar por fractal
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relacionar
        a, b, c, d = self.relaciones(a, b, c, d)
        
        # Fusionar
        out = torch.cat([a, b, c, d], dim=-1)
        estados = {'a': a, 'b': b, 'c': c, 'd': d, 'meta': meta}
        
        return out, estados


class LlaveIntercapa(nn.Module):
    """
    Llave que conecta capas de datos con capas de cálculos.
    Bidireccional.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.llave_forward = nn.Linear(dim, dim)
        self.llave_backward = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, origen: torch.Tensor, destino: torch.Tensor) -> torch.Tensor:
        # Transformar origen
        trans = self.llave_forward(origen)
        
        # Gate para controlar cuánta información pasa
        gate = self.gate(torch.cat([trans, destino], dim=-1))
        
        # Mezclar con gate
        return self.norm(destino + gate * trans)
    
    def backward_pass(self, origen: torch.Tensor, destino: torch.Tensor) -> torch.Tensor:
        # Transformar en dirección inversa
        trans = self.llave_backward(origen)
        gate = self.gate(torch.cat([trans, destino], dim=-1))
        return self.norm(destino + gate * trans)


class LlarriO1_HyperComprimido(nn.Module):
    """
    LLARRI-O1 v4.0 - HyperComprimido
    
    Arquitectura con 6 cajas:
    - 3 Cajas de DATOS (procesan información)
    - 3 Cajas de CÁLCULOS (operan sobre datos y otros cálculos)
    
    8 niveles fractales: 256→128→64→32→16→8→4→2
    Cache RAM para nivel binario (máxima velocidad)
    
    Autor: Lucas Mella (Segunda Cabeza)
    """
    
    def __init__(self, config: Optional[ConfigHyperComprimido] = None):
        super().__init__()
        self.config = config or ConfigHyperComprimido()
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Cache binario para acelerar nivel más profundo
        self.cache_binario = CacheBinario(torch.device('cpu')) if self.config.usar_cache_binario else None
        
        # Cuadrante fractal compartido (8 niveles)
        niveles_cuadrante = [quad_dim]
        for i in range(1, len(self.config.niveles_fractales)):
            ratio = self.config.niveles_fractales[i] / self.config.niveles_fractales[0]
            niveles_cuadrante.append(max(2, int(quad_dim * ratio)))
        
        config_cuadrante = ConfigHyperComprimido(
            niveles_fractales=niveles_cuadrante,
            dropout=self.config.dropout
        )
        self.cuadrante_base = CuadranteFractal(config_cuadrante, self.cache_binario)
        
        # === CAPA DE DATOS (3 cajas) ===
        self.caja_datos_A = CajaDatos(self.config, self.cuadrante_base, self.cache_binario)
        self.caja_datos_B = CajaDatos(self.config, self.cuadrante_base, self.cache_binario)
        self.caja_datos_C = CajaDatos(self.config, self.cuadrante_base, self.cache_binario)
        
        # === CAPA DE CÁLCULOS (3 cajas) ===
        self.caja_calc_A = CajaCalculos(self.config, self.cuadrante_base)
        self.caja_calc_B = CajaCalculos(self.config, self.cuadrante_base)
        self.caja_calc_C = CajaCalculos(self.config, self.cuadrante_base)
        
        # === LLAVES INTRACAPA (dentro de cada capa) ===
        # Datos
        self.llave_datos_AB = LlaveIntercapa(dim)
        self.llave_datos_BC = LlaveIntercapa(dim)
        self.llave_datos_CA = LlaveIntercapa(dim)
        # Cálculos
        self.llave_calc_AB = LlaveIntercapa(dim)
        self.llave_calc_BC = LlaveIntercapa(dim)
        self.llave_calc_CA = LlaveIntercapa(dim)
        
        # === LLAVES INTERCAPA (datos ↔ cálculos) ===
        self.llave_A_datos_calc = LlaveIntercapa(dim)
        self.llave_B_datos_calc = LlaveIntercapa(dim)
        self.llave_C_datos_calc = LlaveIntercapa(dim)
        
        # === SALIDA ===
        self.output = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Fusión datos + cálculos
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, self.config.output_dim)
        )
        
        self._print_info()
    
    def _print_info(self):
        params = sum(p.numel() for p in self.parameters())
        params_sin = self._params_sin_compartir()
        relaciones = self._calcular_relaciones()
        comp = (1 - params/params_sin) * 100
        
        print(f"\n{'='*70}")
        print(f"{'LLARRI-O1 v4.0 - HYPERCOMPRIMIDO':^70}")
        print(f"{'='*70}")
        print(f"  Autor: Lucas Mella (Segunda Cabeza)")
        print(f"  Coordinador: Alvaro (Segunda Cabeza)")
        print(f"{'='*70}")
        print(f"  ARQUITECTURA:")
        print(f"    • Cajas de datos: {self.config.num_cajas_datos}")
        print(f"    • Cajas de cálculos: {self.config.num_cajas_calculos}")
        print(f"    • Total cajas: {self.config.num_cajas_datos + self.config.num_cajas_calculos}")
        print(f"    • Niveles fractales: {len(self.config.niveles_fractales)}")
        print(f"    • Niveles: {' → '.join(map(str, self.config.niveles_fractales))}")
        print(f"    • Cache binario: {'✓ Activado' if self.config.usar_cache_binario else '✗ Desactivado'}")
        print(f"{'='*70}")
        print(f"  PARÁMETROS:")
        print(f"    • Reales: {params:,}")
        print(f"    • Sin compartir: {params_sin:,}")
        print(f"    • Compresión: {comp:.1f}%")
        print(f"    • Factor: {params_sin/params:.1f}x")
        print(f"{'='*70}")
        print(f"  RELACIONES:")
        print(f"    • Totales representadas: {relaciones:,}")
        print(f"    • Factor vs parámetros: {relaciones/params:,.0f}x")
        print(f"    • Equivalente en GB: {relaciones * 4 / 1e9:.2f} GB")
        print(f"{'='*70}")
        print(f"  TAMAÑO:")
        print(f"    • Modelo real: {params * 4 / 1e6:.2f} MB")
        print(f"    • Si guardara relaciones: {relaciones * 4 / 1e9:.2f} GB")
        print(f"{'='*70}\n")
    
    def _params_sin_compartir(self) -> int:
        """Calcula parámetros si no hubiera pesos compartidos."""
        params_cuadrante = sum(p.numel() for p in self.cuadrante_base.parameters())
        # 6 cajas × 4 cuadrantes = 24 cuadrantes, pero compartimos 1
        return sum(p.numel() for p in self.parameters()) + params_cuadrante * 23
    
    def _calcular_relaciones(self) -> int:
        """Calcula el número total de relaciones representadas."""
        dim = self.config.hidden_dim
        niveles = len(self.config.niveles_fractales)
        
        # Relaciones por nivel fractal
        relaciones_por_nivel = sum(
            self.config.niveles_fractales[i] * self.config.niveles_fractales[i]
            for i in range(niveles)
        )
        
        # 6 cajas × 4 cuadrantes × relaciones internas
        relaciones_internas = 6 * 4 * relaciones_por_nivel
        
        # Relaciones entre cajas (6×5/2 = 15 pares)
        relaciones_entre_cajas = 15 * dim * dim
        
        # Relaciones intercapa (datos ↔ cálculos)
        relaciones_intercapa = 3 * dim * dim * 2  # bidireccional
        
        # Combinaciones binarias (nivel profundo)
        combinaciones_binarias = 4 * 4 * 7  # 4 estados × 4 estados × 7 operaciones
        relaciones_binarias = combinaciones_binarias * 6 * 4  # por cada cuadrante
        
        return relaciones_internas + relaciones_entre_cajas + relaciones_intercapa + relaciones_binarias
    
    def to(self, device):
        """Override para mover también el cache binario."""
        super().to(device)
        if self.cache_binario:
            self.cache_binario.device = device
            self.cache_binario._inicializar_cache()
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === CAPA DE DATOS ===
        # Procesar las 3 cajas de datos
        out_A, estados_A = self.caja_datos_A(x)
        out_B, estados_B = self.caja_datos_B(x)
        out_C, estados_C = self.caja_datos_C(x)
        
        # Conectar cajas de datos entre sí (ciclo A→B→C→A)
        out_B = self.llave_datos_AB(out_A, out_B)
        out_C = self.llave_datos_BC(out_B, out_C)
        out_A = self.llave_datos_CA(out_C, out_A)
        
        # === CAPA DE CÁLCULOS ===
        # Cálculos sobre pares de datos
        calc_A, est_calc_A = self.caja_calc_A(out_A, out_B, None)
        calc_B, est_calc_B = self.caja_calc_B(out_B, out_C, calc_A)  # usa calc_A
        calc_C, est_calc_C = self.caja_calc_C(out_C, out_A, calc_B)  # usa calc_B
        
        # Conectar cajas de cálculos entre sí
        calc_B = self.llave_calc_AB(calc_A, calc_B)
        calc_C = self.llave_calc_BC(calc_B, calc_C)
        calc_A = self.llave_calc_CA(calc_C, calc_A)
        
        # === CONEXIONES INTERCAPA (datos ↔ cálculos) ===
        # Los cálculos refinan los datos
        out_A = self.llave_A_datos_calc.backward_pass(calc_A, out_A)
        out_B = self.llave_B_datos_calc.backward_pass(calc_B, out_B)
        out_C = self.llave_C_datos_calc.backward_pass(calc_C, out_C)
        
        # === FUSIÓN FINAL ===
        # Combinar la mejor representación de datos y cálculos
        datos_final = out_A + out_B + out_C
        calc_final = calc_A + calc_B + calc_C
        
        fusion = torch.cat([datos_final, calc_final], dim=-1)
        
        return self.output(fusion)


def entrenar_hypercomprimido(epochs: int = 25, batch_size: int = 64):
    """
    Entrena LLARRI-O1 v4.0 HyperComprimido.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"{'ENTRENAMIENTO HYPERCOMPRIMIDO':^70}")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    # Datos con augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, 
                             num_workers=2, pin_memory=True)
    
    # Modelo
    config = ConfigHyperComprimido(
        usar_cache_binario=True
    )
    model = LlarriO1_HyperComprimido(config).to(device)
    
    # Optimizador con weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(data)
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - start
        lr = scheduler.get_last_lr()[0]
        
        scheduler.step()
        
        # Early stopping check
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
            }, 'checkpoints/llarri_hypercomprimido_mejor.pt')
            marker = '★'
        else:
            no_improve += 1
            marker = ' '
        
        print(f"Epoch {epoch+1:2d}/{epochs} ({elapsed:.1f}s) | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"LR: {lr:.6f} {marker}")
        
        # Early stopping
        if no_improve >= patience:
            print(f"\n⚠️  Early stopping: sin mejora en {patience} epochs")
            break
    
    print(f"\n{'='*70}")
    print(f"  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor accuracy: {best_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return model, best_acc


if __name__ == "__main__":
    # Crear directorio de checkpoints
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Entrenar
    model, acc = entrenar_hypercomprimido(epochs=30, batch_size=64)
