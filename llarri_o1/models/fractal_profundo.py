# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
====================================================

ARQUITECTURA DEFINITIVA diseñada por Lucas Mella (Segunda Cabeza)

CARACTERÍSTICAS:
- Recursión fractal hasta el límite matemático
- Pesos compartidos en todos los niveles (99%+ compresión)
- Modo HÍBRIDO CPU/GPU adaptativo:
  * GPU para cálculos pesados (multiplicación de matrices)
  * CPU/RAM para operaciones ligeras (no usar topadora para botellas)

Autor: Lucas Mella (Segunda Cabeza)
Licencia: Propietaria con atribución
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import math


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class LlarriFractalConfig:
    """
    Configuración del modelo LLARRI-O1 v3.0
    
    El nivel de profundidad se calcula automáticamente basado en hidden_dim
    """
    
    # Dimensiones
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    
    # Arquitectura fractal
    num_cajas: int = 3
    cuadrantes_por_nivel: int = 4
    
    # Profundidad fractal (-1 = automático)
    profundidad_fractal: int = -1
    dim_minima_cuadrante: int = 4
    
    # Llaves
    usar_llaves_bidireccionales: bool = True
    usar_retroalimentacion: bool = True
    
    # Pesos compartidos
    compartir_pesos_todos_niveles: bool = True
    compartir_pesos_entre_cajas: bool = True
    
    # Regularización
    dropout: float = 0.1
    
    # Dispositivo y modo híbrido
    device: str = "auto"
    modo_hibrido: str = "auto"  # "auto", "gpu", "hibrido", "cpu"
    umbral_vram_mb: int = 500  # Si el modelo usa más, considerar híbrido
    
    # Operaciones en CPU (ligeras - "no usar topadora")
    ops_cpu: List[str] = field(default_factory=lambda: [
        'dropout', 'layernorm', 'activation', 'residual'
    ])
    
    def calcular_profundidad_maxima(self) -> int:
        profundidad = int(math.log(self.hidden_dim / self.dim_minima_cuadrante) / math.log(4))
        return max(1, profundidad)
    
    def get_profundidad_efectiva(self) -> int:
        if self.profundidad_fractal == -1:
            return self.calcular_profundidad_maxima()
        return min(self.profundidad_fractal, self.calcular_profundidad_maxima())
    
    def get_dims_por_nivel(self) -> List[int]:
        profundidad = self.get_profundidad_efectiva()
        dims = []
        dim_actual = self.hidden_dim
        for i in range(profundidad + 1):
            dims.append(dim_actual)
            dim_actual = dim_actual // 4
        return dims


class PosicionCuadrante(Enum):
    SUPERIOR_IZQUIERDA = 0
    SUPERIOR_DERECHA = 1
    INFERIOR_IZQUIERDA = 2
    INFERIOR_DERECHA = 3


# ==============================================================================
# DETECTOR DE RECURSOS Y MODO HÍBRIDO
# ==============================================================================

class ResourceManager:
    """
    Gestiona la distribución inteligente entre CPU y GPU.
    
    Principio: "No usar topadora para levantar botellas"
    - GPU: Multiplicaciones de matrices grandes (el trabajo pesado)
    - CPU: Operaciones ligeras (dropout, activaciones, normalización)
    """
    
    def __init__(self, config: LlarriFractalConfig):
        self.config = config
        self.gpu_disponible = torch.cuda.is_available()
        
        # Detectar recursos
        self.vram_total = self._get_vram_total()
        self.vram_libre = self._get_vram_libre()
        self.ram_libre = self._get_ram_libre()
        
        # Decidir modo
        self.modo = self._decidir_modo()
        
        # Dispositivos
        self.device_pesado = torch.device("cuda" if self.modo != "cpu" and self.gpu_disponible else "cpu")
        self.device_ligero = torch.device("cpu") if self.modo == "hibrido" else self.device_pesado
    
    def _get_vram_total(self) -> int:
        if self.gpu_disponible:
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    def _get_vram_libre(self) -> int:
        if self.gpu_disponible:
            return self.vram_total - torch.cuda.memory_allocated()
        return 0
    
    def _get_ram_libre(self) -> int:
        try:
            import psutil
            return psutil.virtual_memory().available
        except:
            return 16 * 1024**3
    
    def _decidir_modo(self) -> str:
        if self.config.modo_hibrido != "auto":
            return self.config.modo_hibrido
        
        if not self.gpu_disponible:
            return "cpu"
        
        # Estimar memoria del modelo
        params_estimados = self._estimar_parametros()
        memoria_modelo = params_estimados * 4 * 4  # float32 * 4 (pesos+grads+optim)
        
        umbral = self.config.umbral_vram_mb * 1024 * 1024
        
        if memoria_modelo < self.vram_libre * 0.7:
            return "gpu"
        elif memoria_modelo < self.ram_libre * 0.5:
            return "hibrido"
        else:
            return "cpu"
    
    def _estimar_parametros(self) -> int:
        dim = self.config.hidden_dim
        profundidad = self.config.get_profundidad_efectiva()
        
        # Estimación aproximada
        params = dim * dim * 10 * (profundidad + 1)
        return params
    
    def get_device_for_op(self, op_type: str) -> torch.device:
        """
        Retorna el dispositivo óptimo para un tipo de operación.
        
        - 'matmul', 'linear', 'conv': GPU (trabajo pesado)
        - 'dropout', 'layernorm', 'activation': CPU si híbrido (trabajo ligero)
        """
        if self.modo == "gpu":
            return self.device_pesado
        elif self.modo == "cpu":
            return self.device_ligero
        else:  # hibrido
            if op_type in ['matmul', 'linear', 'conv', 'attention']:
                return self.device_pesado
            else:
                return self.device_ligero
    
    def print_info(self):
        print(f"\n  {'─'*50}")
        print(f"  GESTIÓN DE RECURSOS")
        print(f"  {'─'*50}")
        print(f"  Modo: {self.modo.upper()}")
        print(f"  GPU disponible: {self.gpu_disponible}")
        if self.gpu_disponible:
            print(f"  VRAM total: {self.vram_total/1e9:.1f} GB")
            print(f"  VRAM libre: {self.vram_libre/1e9:.1f} GB")
        print(f"  RAM libre: {self.ram_libre/1e9:.1f} GB")
        print(f"  Device pesado: {self.device_pesado}")
        print(f"  Device ligero: {self.device_ligero}")
        print(f"  {'─'*50}\n")


# ==============================================================================
# OPERACIONES HÍBRIDAS
# ==============================================================================

class LinearHibrido(nn.Module):
    """
    Capa Linear que puede ejecutar en GPU y mover resultado a CPU.
    El trabajo pesado (matmul) va en GPU, el resto puede ir en CPU.
    """
    
    def __init__(self, in_features: int, out_features: int, resource_manager: ResourceManager):
        super().__init__()
        self.rm = resource_manager
        
        # Los pesos SIEMPRE en el dispositivo pesado (GPU si disponible)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.to(self.rm.device_pesado)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mover input a GPU para el cálculo
        x_gpu = x.to(self.rm.device_pesado)
        
        # Cálculo pesado en GPU
        out = self.linear(x_gpu)
        
        # Si modo híbrido, mover resultado a CPU para operaciones ligeras
        if self.rm.modo == "hibrido":
            return out.to(self.rm.device_ligero)
        return out


class LayerNormHibrido(nn.Module):
    """LayerNorm que puede ejecutarse en CPU (operación ligera)"""
    
    def __init__(self, dim: int, resource_manager: ResourceManager):
        super().__init__()
        self.rm = resource_manager
        self.norm = nn.LayerNorm(dim)
        self.norm.to(self.rm.device_ligero)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.to(self.rm.device_ligero)
        return self.norm(x_cpu)


class DropoutHibrido(nn.Module):
    """Dropout en CPU (operación muy ligera)"""
    
    def __init__(self, p: float, resource_manager: ResourceManager):
        super().__init__()
        self.rm = resource_manager
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dropout es tan ligero que siempre puede ir en CPU
        if self.rm.modo == "hibrido":
            x_cpu = x.to(self.rm.device_ligero)
            return self.dropout(x_cpu)
        return self.dropout(x)


# ==============================================================================
# CUADRANTE FRACTAL RECURSIVO (CON HÍBRIDO)
# ==============================================================================

class CuadranteFractal(nn.Module):
    """
    Unidad fundamental recursiva con soporte híbrido CPU/GPU.
    
    - Cálculos pesados (Linear): GPU
    - Operaciones ligeras (Norm, Dropout, GELU): CPU si híbrido
    """
    
    def __init__(
        self, 
        dim: int, 
        nivel: int,
        nivel_maximo: int,
        posicion: PosicionCuadrante,
        resource_manager: ResourceManager,
        pesos_compartidos: Optional[Dict[int, nn.Module]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.nivel = nivel
        self.nivel_maximo = nivel_maximo
        self.posicion = posicion
        self.rm = resource_manager
        self.es_nivel_final = (nivel >= nivel_maximo) or (dim // 4 < 1)
        self.sub_dim = dim // 4 if not self.es_nivel_final else dim
        
        # Cálculo interno (pesado en GPU, ligero en CPU)
        self.linear1 = LinearHibrido(dim, dim, resource_manager)
        self.norm1 = LayerNormHibrido(dim, resource_manager)
        self.dropout1 = DropoutHibrido(dropout, resource_manager)
        self.linear2 = LinearHibrido(dim, dim, resource_manager)
        
        if not self.es_nivel_final:
            # Sub-cuadrantes recursivos
            if pesos_compartidos is not None and nivel + 1 in pesos_compartidos:
                self.subcuadrante_base = pesos_compartidos[nivel + 1]
            else:
                self.subcuadrante_base = CuadranteFractal(
                    dim=self.sub_dim,
                    nivel=nivel + 1,
                    nivel_maximo=nivel_maximo,
                    posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
                    resource_manager=resource_manager,
                    pesos_compartidos=pesos_compartidos,
                    dropout=dropout
                )
                if pesos_compartidos is not None:
                    pesos_compartidos[nivel + 1] = self.subcuadrante_base
            
            # Relaciones posicionales (pesadas - GPU)
            self.rel_horizontal = LinearHibrido(self.sub_dim * 2, self.sub_dim, resource_manager)
            self.rel_vertical = LinearHibrido(self.sub_dim * 2, self.sub_dim, resource_manager)
            self.rel_diagonal = LinearHibrido(self.sub_dim * 2, self.sub_dim, resource_manager)
            
            # Fusión
            self.fusion = LinearHibrido(self.sub_dim * 4, dim, resource_manager)
            self.norm_fusion = LayerNormHibrido(dim, resource_manager)
            self.dropout_fusion = DropoutHibrido(dropout, resource_manager)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Cálculo interno
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.gelu(x)  # GELU es ligero
        x = self.dropout1(x)
        x = self.linear2(x)
        
        # Residual (asegurar mismo dispositivo)
        if residual.device != x.device:
            residual = residual.to(x.device)
        x = x + residual
        
        if self.es_nivel_final:
            return x, {'nivel': self.nivel, 'dim': self.dim, 'final': True}
        
        # Dividir en sub-cuadrantes
        s1 = x[..., :self.sub_dim]
        s2 = x[..., self.sub_dim:self.sub_dim*2]
        s3 = x[..., self.sub_dim*2:self.sub_dim*3]
        s4 = x[..., self.sub_dim*3:self.sub_dim*4]
        
        # Procesar recursivamente
        s1, _ = self.subcuadrante_base(s1)
        s2, _ = self.subcuadrante_base(s2)
        s3, _ = self.subcuadrante_base(s3)
        s4, _ = self.subcuadrante_base(s4)
        
        # Asegurar mismo dispositivo para concatenación
        device = s1.device
        s2 = s2.to(device)
        s3 = s3.to(device)
        s4 = s4.to(device)
        
        # Relaciones posicionales
        rel_h = self.rel_horizontal(torch.cat([s1, s2], dim=-1))
        rel_v = self.rel_vertical(torch.cat([s1, s3], dim=-1))
        rel_d = self.rel_diagonal(torch.cat([s1, s4], dim=-1))
        
        # Actualizar
        s1 = s1 + rel_h.to(s1.device) + rel_v.to(s1.device) + rel_d.to(s1.device)
        s2 = s2 + rel_h.to(s2.device)
        s3 = s3 + rel_v.to(s3.device)
        s4 = s4 + rel_d.to(s4.device)
        
        # Fusionar
        fusion_input = torch.cat([s1, s2, s3, s4], dim=-1)
        output = self.fusion(fusion_input)
        output = self.norm_fusion(output)
        output = F.gelu(output)
        output = self.dropout_fusion(output)
        
        # Residual
        if x.device != output.device:
            x = x.to(output.device)
        output = output + x
        
        return output, {'nivel': self.nivel, 'sub_dim': self.sub_dim}


# ==============================================================================
# CAJA TRINITY
# ==============================================================================

class CajaTrinityFractal(nn.Module):
    """Caja Trinity con 4 Cuadrantes Fractales y soporte híbrido"""
    
    def __init__(
        self,
        config: LlarriFractalConfig,
        resource_manager: ResourceManager,
        cuadrante_compartido: Optional[CuadranteFractal] = None,
        pesos_compartidos: Optional[Dict[int, nn.Module]] = None
    ):
        super().__init__()
        self.config = config
        self.rm = resource_manager
        dim = config.hidden_dim
        profundidad = config.get_profundidad_efectiva()
        
        # Proyección de entrada (pesado)
        self.input_proj = LinearHibrido(config.input_dim, dim, resource_manager)
        self.internal_proj = LinearHibrido(dim, dim, resource_manager)
        
        # Cuadrante base
        if cuadrante_compartido is not None and config.compartir_pesos_entre_cajas:
            self.cuadrante_base = cuadrante_compartido
        else:
            self.cuadrante_base = CuadranteFractal(
                dim=dim // 4,
                nivel=0,
                nivel_maximo=profundidad,
                posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
                resource_manager=resource_manager,
                pesos_compartidos=pesos_compartidos,
                dropout=config.dropout
            )
        
        # Relaciones entre cuadrantes
        quad_dim = dim // 4
        self.rel_horizontal = LinearHibrido(quad_dim * 2, quad_dim, resource_manager)
        self.rel_vertical = LinearHibrido(quad_dim * 2, quad_dim, resource_manager)
        self.rel_diagonal = LinearHibrido(quad_dim * 2, quad_dim, resource_manager)
        
        # Fusión
        self.fusion_caja = LinearHibrido(dim, dim, resource_manager)
        self.norm_caja = LayerNormHibrido(dim, resource_manager)
        self.dropout_caja = DropoutHibrido(config.dropout, resource_manager)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Proyectar entrada
        if x.shape[-1] == self.config.input_dim:
            x = self.input_proj(x)
        elif x.shape[-1] != dim:
            x = self.internal_proj(x)
        
        # Dividir en cuadrantes
        xA = x[..., :quad_dim]
        xB = x[..., quad_dim:quad_dim*2]
        xC = x[..., quad_dim*2:quad_dim*3]
        xD = x[..., quad_dim*3:]
        
        # Procesar cuadrantes (mismo peso)
        A, _ = self.cuadrante_base(xA)
        B, _ = self.cuadrante_base(xB)
        C, _ = self.cuadrante_base(xC)
        D, _ = self.cuadrante_base(xD)
        
        # Asegurar dispositivo
        device = A.device
        B, C, D = B.to(device), C.to(device), D.to(device)
        
        # Relaciones
        rel_h = self.rel_horizontal(torch.cat([A, B], dim=-1))
        rel_v = self.rel_vertical(torch.cat([A, C], dim=-1))
        rel_d = self.rel_diagonal(torch.cat([A, D], dim=-1))
        
        A = A + rel_h.to(A.device) + rel_v.to(A.device) + rel_d.to(A.device)
        B = B + rel_h.to(B.device)
        C = C + rel_v.to(C.device)
        D = D + rel_d.to(D.device)
        
        # Fusionar
        fusion = torch.cat([A, B, C, D], dim=-1)
        output = self.fusion_caja(fusion)
        output = self.norm_caja(output)
        output = F.gelu(output)
        output = self.dropout_caja(output)
        
        return output, {}


# ==============================================================================
# LLAVES TRINITY
# ==============================================================================

class LlaveTrinity(nn.Module):
    """Llave para conexión entre cajas (operación pesada)"""
    
    def __init__(self, dim: int, resource_manager: ResourceManager, bidireccional: bool = True):
        super().__init__()
        self.bidireccional = bidireccional
        self.rm = resource_manager
        
        self.llave_ida = LinearHibrido(dim, dim, resource_manager)
        if bidireccional:
            self.llave_vuelta = LinearHibrido(dim, dim, resource_manager)
    
    def forward(self, origen: torch.Tensor, destino: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Asegurar dispositivo
        if origen.device != destino.device:
            destino = destino.to(origen.device)
        
        destino_mod = destino + self.llave_ida(origen).to(destino.device)
        
        if self.bidireccional:
            origen_mod = origen + self.llave_vuelta(destino).to(origen.device)
        else:
            origen_mod = origen
            
        return origen_mod, destino_mod


# ==============================================================================
# MODELO PRINCIPAL
# ==============================================================================

class LlarriO1_FractalProfundo(nn.Module):
    """
    LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
    =====================================================
    
    Con modo HÍBRIDO CPU/GPU:
    - GPU: Multiplicaciones de matrices (trabajo pesado)
    - CPU: Dropout, LayerNorm, activaciones (trabajo ligero)
    
    "No usar topadora para levantar botellas"
    
    Autor: Lucas Mella (Segunda Cabeza)
    """
    
    def __init__(self, config: Optional[LlarriFractalConfig] = None):
        super().__init__()
        self.config = config or LlarriFractalConfig()
        
        # Gestor de recursos
        self.rm = ResourceManager(self.config)
        
        dim = self.config.hidden_dim
        profundidad = self.config.get_profundidad_efectiva()
        
        # Pesos compartidos
        self.pesos_compartidos: Dict[int, nn.Module] = {}
        
        # Cuadrante base
        self.cuadrante_base = CuadranteFractal(
            dim=dim // 4,
            nivel=0,
            nivel_maximo=profundidad,
            posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
            resource_manager=self.rm,
            pesos_compartidos=self.pesos_compartidos if self.config.compartir_pesos_todos_niveles else None,
            dropout=self.config.dropout
        )
        
        # 3 Cajas Trinity
        self.caja1 = CajaTrinityFractal(
            self.config, self.rm,
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        self.caja2 = CajaTrinityFractal(
            self.config, self.rm,
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        self.caja3 = CajaTrinityFractal(
            self.config, self.rm,
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        
        # Llaves
        self.llave_1_2 = LlaveTrinity(dim, self.rm, bidireccional=self.config.usar_llaves_bidireccionales)
        self.llave_1_3 = LlaveTrinity(dim, self.rm, bidireccional=False)
        self.llave_2_3 = LlaveTrinity(dim, self.rm, bidireccional=False)
        
        if self.config.usar_retroalimentacion:
            self.llave_3_1 = LlaveTrinity(dim, self.rm, bidireccional=False)
        
        # Capa de salida
        self.output_linear = LinearHibrido(dim, dim, self.rm)
        self.output_norm = LayerNormHibrido(dim, self.rm)
        self.output_dropout = DropoutHibrido(self.config.dropout, self.rm)
        self.output_final = LinearHibrido(dim, self.config.output_dim, self.rm)
        
        # Info
        self._print_info()
    
    def _print_info(self):
        params = sum(p.numel() for p in self.parameters())
        params_sin = self._calcular_params_sin_compartir()
        compresion = (1 - params / params_sin) * 100 if params_sin > 0 else 0
        profundidad = self.config.get_profundidad_efectiva()
        dims = self.config.get_dims_por_nivel()
        
        print(f"\n{'='*70}")
        print(f"  LLARRI-O1 v3.0 - TRINITY FRACTAL RECURSIVO PROFUNDO")
        print(f"{'='*70}")
        print(f"  Autor: Lucas Mella (Segunda Cabeza)")
        print(f"{'='*70}")
        print(f"\n  ARQUITECTURA:")
        print(f"  • Cajas Trinity: 3")
        print(f"  • Cuadrantes por caja: 4")
        print(f"  • Profundidad fractal: {profundidad} niveles")
        print(f"  • Dimensiones por nivel: {dims}")
        print(f"\n  COMPRESIÓN:")
        print(f"  • Parámetros reales:    {params:,}")
        print(f"  • Sin compartir serían: {params_sin:,}")
        print(f"  • Compresión:           {compresion:.1f}%")
        print(f"  • Factor de reducción:  {params_sin/max(params,1):.1f}x")
        
        # Info de recursos
        self.rm.print_info()
    
    def _calcular_params_sin_compartir(self) -> int:
        dim = self.config.hidden_dim
        profundidad = self.config.get_profundidad_efectiva()
        total_cuadrantes = (4 ** profundidad) * 3 * 4
        params_cuadrante = dim * dim * 4
        return total_cuadrantes * params_cuadrante
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input al dispositivo pesado
        x = x.to(self.rm.device_pesado)
        
        # Procesar cajas
        out1, _ = self.caja1(x)
        out2, _ = self.caja2(x)
        
        # Llave 1↔2
        out1, out2 = self.llave_1_2(out1, out2)
        
        # Llaves hacia caja 3
        _, out3_from_1 = self.llave_1_3(out1, torch.zeros_like(out1))
        _, out3_from_2 = self.llave_2_3(out2, torch.zeros_like(out2))
        
        input_caja3 = out3_from_1.to(out3_from_2.device) + out3_from_2
        
        # Caja 3
        out3, _ = self.caja3(input_caja3)
        
        # Retroalimentación
        if self.config.usar_retroalimentacion:
            _, out1 = self.llave_3_1(out3, out1)
        
        # Salida
        output = self.output_linear(out3)
        output = self.output_norm(output)
        output = F.gelu(output)
        output = self.output_dropout(output)
        output = self.output_final(output)
        
        return output
    
    def get_compression_stats(self) -> Dict:
        params = sum(p.numel() for p in self.parameters())
        params_sin = self._calcular_params_sin_compartir()
        
        return {
            'parametros_reales': params,
            'parametros_sin_compartir': params_sin,
            'compresion_porcentaje': (1 - params / params_sin) * 100,
            'factor_reduccion': params_sin / max(params, 1),
            'profundidad_fractal': self.config.get_profundidad_efectiva(),
            'dimensiones_por_nivel': self.config.get_dims_por_nivel(),
            'modo': self.rm.modo
        }
    
    def get_estructura_fractal(self) -> str:
        dims = self.config.get_dims_por_nivel()
        profundidad = self.config.get_profundidad_efectiva()
        
        lineas = [
            f"ESTRUCTURA FRACTAL LLARRI-O1 v3.0 (Modo: {self.rm.modo.upper()})",
            "=" * 55,
            "",
            "CAJA (×3)"
        ]
        
        for nivel in range(profundidad + 1):
            indent = "  " * (nivel + 1)
            dim = dims[nivel] if nivel < len(dims) else "?"
            nombres = ["A,B,C,D", "α", "β", "γ", "δ", "ε"][min(nivel, 5)]
            lineas.append(f"{indent}├── Cuadrante [{nombres}] (dim={dim})")
        
        lineas.extend([
            "",
            f"Total niveles: {profundidad + 1}",
            f"Dimensiones: {dims}"
        ])
        
        return "\n".join(lineas)


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def crear_modelo_fractal(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    profundidad: int = -1,
    modo_hibrido: str = "auto",
    **kwargs
) -> LlarriO1_FractalProfundo:
    """Crea una instancia del modelo fractal profundo"""
    config = LlarriFractalConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        profundidad_fractal=profundidad,
        modo_hibrido=modo_hibrido,
        **kwargs
    )
    return LlarriO1_FractalProfundo(config)


def explorar_profundidades():
    """Explora diferentes configuraciones"""
    print("\n" + "="*70)
    print("EXPLORACIÓN DE PROFUNDIDADES FRACTALES")
    print("="*70)
    
    for dim in [64, 128, 256, 512, 1024, 2048, 4096]:
        config = LlarriFractalConfig(hidden_dim=dim)
        prof = config.get_profundidad_efectiva()
        dims = config.get_dims_por_nivel()
        print(f"\nhidden_dim={dim:4d} → profundidad={prof} → dims={dims}")


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMO: LLARRI-O1 v3.0 CON MODO HÍBRIDO")
    print("="*70)
    
    # Crear modelo
    modelo = crear_modelo_fractal(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        modo_hibrido="auto"
    )
    
    print(modelo.get_estructura_fractal())
    
    # Test
    import torch
    x = torch.randn(32, 784)
    output = modelo(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    stats = modelo.get_compression_stats()
    print(f"\nModo: {stats['modo']}")
    print(f"Compresión: {stats['compresion_porcentaje']:.1f}%")
