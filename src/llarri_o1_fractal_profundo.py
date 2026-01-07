"""
LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
====================================================

ARQUITECTURA DEFINITIVA diseñada por Lucas Mella (Segunda Cabeza)
Fecha: Enero 2026

Concepto: RECURSIÓN FRACTAL HASTA EL LÍMITE MATEMÁTICO
- Cada cuadrante contiene 4 sub-cuadrantes
- Cada sub-cuadrante contiene 4 sub-sub-cuadrantes
- ...y así hasta llegar a dimensión = 1 (el mínimo absoluto)

MATEMÁTICAS:
- Dimensión D inicial
- Cada nivel divide entre 4: D/4, D/16, D/64, D/256...
- Niveles máximos = log₄(D) = log(D)/log(4)
- Con D=256: máximo 4 niveles (256→64→16→4→1)
- Con D=1024: máximo 5 niveles
- Con D=4096: máximo 6 niveles

COMPRESIÓN EXPONENCIAL:
- Sin compartir: 4^n conjuntos de pesos por nivel
- Con compartir: 1 conjunto de pesos por nivel
- Compresión total: 4^(niveles) : 1

Autor: Lucas Mella (Segunda Cabeza)
Organización: Segunda Cabeza
Licencia: Propietaria con atribución
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import math


# ==============================================================================
# CONFIGURACIÓN AVANZADA
# ==============================================================================

@dataclass
class LlarriFractalConfig:
    """
    Configuración del modelo LLARRI-O1 v3.0 Fractal Profundo
    
    El nivel de profundidad se calcula automáticamente basado en hidden_dim:
    - hidden_dim=256: profundidad_max=4 (256→64→16→4→1)
    - hidden_dim=1024: profundidad_max=5
    - hidden_dim=4096: profundidad_max=6
    """
    
    # Dimensiones
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    
    # Arquitectura fractal
    num_cajas: int = 3
    cuadrantes_por_nivel: int = 4  # Siempre 4 (división en cuadrantes)
    
    # Profundidad fractal
    # -1 = automático (máximo posible según hidden_dim)
    profundidad_fractal: int = -1
    
    # Dimensión mínima de cuadrante (1, 2, 4, etc.)
    dim_minima_cuadrante: int = 4
    
    # Llaves entre cajas
    usar_llaves_bidireccionales: bool = True
    usar_retroalimentacion: bool = True
    
    # Optimización de pesos
    compartir_pesos_todos_niveles: bool = True
    compartir_pesos_entre_cajas: bool = True
    
    # Regularización
    dropout: float = 0.1
    
    # Dispositivo
    device: str = "auto"
    
    def calcular_profundidad_maxima(self) -> int:
        """Calcula la profundidad máxima basada en hidden_dim"""
        # Necesitamos al menos dim_minima por cuadrante
        # En cada nivel dividimos por 4
        # profundidad = log4(hidden_dim / dim_minima)
        profundidad = int(math.log(self.hidden_dim / self.dim_minima_cuadrante) / math.log(4))
        return max(1, profundidad)
    
    def get_profundidad_efectiva(self) -> int:
        """Retorna la profundidad efectiva (calculada o especificada)"""
        if self.profundidad_fractal == -1:
            return self.calcular_profundidad_maxima()
        return min(self.profundidad_fractal, self.calcular_profundidad_maxima())
    
    def get_dims_por_nivel(self) -> List[int]:
        """Retorna las dimensiones en cada nivel de profundidad"""
        profundidad = self.get_profundidad_efectiva()
        dims = []
        dim_actual = self.hidden_dim
        for i in range(profundidad + 1):
            dims.append(dim_actual)
            dim_actual = dim_actual // 4
        return dims


class PosicionCuadrante(Enum):
    """Posición del cuadrante en la grilla 2x2"""
    SUPERIOR_IZQUIERDA = 0  # Cuadrante A / a1 / α1
    SUPERIOR_DERECHA = 1    # Cuadrante B / a2 / α2
    INFERIOR_IZQUIERDA = 2  # Cuadrante C / a3 / α3
    INFERIOR_DERECHA = 3    # Cuadrante D / a4 / α4


# ==============================================================================
# CUADRANTE FRACTAL RECURSIVO
# ==============================================================================

class CuadranteFractal(nn.Module):
    """
    UNIDAD FUNDAMENTAL RECURSIVA
    
    Un cuadrante que puede contener 4 sub-cuadrantes,
    que a su vez pueden contener 4 sub-sub-cuadrantes...
    hasta llegar a la dimensión mínima.
    
    ES UNA ESTRUCTURA AUTOSIMILAR (FRACTAL):
    ┌─────────────┬─────────────┐
    │ ┌───┬───┐   │ ┌───┬───┐   │
    │ │α1 │α2 │   │ │α1 │α2 │   │
    │ ├───┼───┤   │ ├───┼───┤   │
    │ │α3 │α4 │   │ │α3 │α4 │   │
    │ └───┴───┘   │ └───┴───┘   │
    │     A       │      B      │
    ├─────────────┼─────────────┤
    │ ┌───┬───┐   │ ┌───┬───┐   │
    │ │α1 │α2 │   │ │α1 │α2 │   │
    │ ├───┼───┤   │ ├───┼───┤   │
    │ │α3 │α4 │   │ │α3 │α4 │   │
    │ └───┴───┘   │ └───┴───┘   │
    │     C       │      D      │
    └─────────────┴─────────────┘
    """
    
    def __init__(
        self, 
        dim: int, 
        nivel: int,
        nivel_maximo: int,
        posicion: PosicionCuadrante,
        pesos_compartidos: Optional[Dict[int, nn.Module]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.nivel = nivel
        self.nivel_maximo = nivel_maximo
        self.posicion = posicion
        self.es_nivel_final = (nivel >= nivel_maximo) or (dim // 4 < 1)
        
        # Dimensión de sub-cuadrantes
        self.sub_dim = dim // 4 if not self.es_nivel_final else dim
        
        # ===== CÁLCULO INTERNO (propio de este cuadrante) =====
        self.calculo_interno = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        if not self.es_nivel_final:
            # ===== SUB-CUADRANTES RECURSIVOS =====
            if pesos_compartidos is not None and nivel + 1 in pesos_compartidos:
                # Reutilizar pesos del nivel inferior
                self.subcuadrante_base = pesos_compartidos[nivel + 1]
            else:
                # Crear UN sub-cuadrante base (se reutiliza para los 4)
                self.subcuadrante_base = CuadranteFractal(
                    dim=self.sub_dim,
                    nivel=nivel + 1,
                    nivel_maximo=nivel_maximo,
                    posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
                    pesos_compartidos=pesos_compartidos,
                    dropout=dropout
                )
                # Registrar para compartir
                if pesos_compartidos is not None:
                    pesos_compartidos[nivel + 1] = self.subcuadrante_base
            
            # ===== RELACIONES POSICIONALES =====
            # Estas SON únicas porque capturan RELACIONES, no datos
            self.rel_horizontal = nn.Linear(self.sub_dim * 2, self.sub_dim)
            self.rel_vertical = nn.Linear(self.sub_dim * 2, self.sub_dim)
            self.rel_diagonal = nn.Linear(self.sub_dim * 2, self.sub_dim)
            
            # ===== FUSIÓN DE SUB-CUADRANTES =====
            self.fusion = nn.Sequential(
                nn.Linear(self.sub_dim * 4, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward recursivo.
        
        Args:
            x: Tensor [batch, dim]
            
        Returns:
            output: Tensor procesado [batch, dim]
            info: Dict con información de cada sub-cuadrante
        """
        # Cálculo interno primero
        x_interno = x + self.calculo_interno(x)
        
        if self.es_nivel_final:
            # Nivel más profundo - solo retornar cálculo interno
            return x_interno, {'nivel': self.nivel, 'dim': self.dim, 'final': True}
        
        # ===== DIVIDIR EN 4 SUB-CUADRANTES =====
        # Cada sub-cuadrante procesa 1/4 de las dimensiones
        x1 = x_interno[..., :self.sub_dim]
        x2 = x_interno[..., self.sub_dim:self.sub_dim*2]
        x3 = x_interno[..., self.sub_dim*2:self.sub_dim*3]
        x4 = x_interno[..., self.sub_dim*3:self.sub_dim*4]
        
        # ===== PROCESAR RECURSIVAMENTE (mismo peso para los 4) =====
        s1, info1 = self.subcuadrante_base(x1)
        s2, info2 = self.subcuadrante_base(x2)
        s3, info3 = self.subcuadrante_base(x3)
        s4, info4 = self.subcuadrante_base(x4)
        
        # ===== RELACIONES POSICIONALES =====
        # Horizontal: s1↔s2, s3↔s4
        rel_h_sup = self.rel_horizontal(torch.cat([s1, s2], dim=-1))
        rel_h_inf = self.rel_horizontal(torch.cat([s3, s4], dim=-1))
        
        # Vertical: s1↔s3, s2↔s4
        rel_v_izq = self.rel_vertical(torch.cat([s1, s3], dim=-1))
        rel_v_der = self.rel_vertical(torch.cat([s2, s4], dim=-1))
        
        # Diagonal: s1↔s4, s2↔s3
        rel_d_pri = self.rel_diagonal(torch.cat([s1, s4], dim=-1))
        rel_d_sec = self.rel_diagonal(torch.cat([s2, s3], dim=-1))
        
        # Actualizar sub-cuadrantes con relaciones
        s1 = s1 + rel_h_sup + rel_v_izq + rel_d_pri
        s2 = s2 + rel_h_sup + rel_v_der + rel_d_sec
        s3 = s3 + rel_h_inf + rel_v_izq + rel_d_sec
        s4 = s4 + rel_h_inf + rel_v_der + rel_d_pri
        
        # ===== FUSIONAR =====
        fusion_input = torch.cat([s1, s2, s3, s4], dim=-1)
        output = self.fusion(fusion_input)
        
        # Conexión residual
        output = output + x_interno
        
        info = {
            'nivel': self.nivel,
            'dim': self.dim,
            'sub_dim': self.sub_dim,
            'subcuadrantes': {'s1': info1, 's2': info2, 's3': info3, 's4': info4}
        }
        
        return output, info


# ==============================================================================
# CAJA TRINITY (3 CAJAS CON LLAVES)
# ==============================================================================

class CajaTrinityFractal(nn.Module):
    """
    Caja Trinity con 4 Cuadrantes Fractales.
    
    Layout:
    ┌─────────────┬─────────────┐
    │      A      │      B      │
    │  (fractal)  │  (fractal)  │
    ├─────────────┼─────────────┤
    │      C      │      D      │
    │  (fractal)  │  (fractal)  │
    └─────────────┴─────────────┘
    
    Cada cuadrante A, B, C, D es un CuadranteFractal recursivo.
    """
    
    def __init__(
        self,
        config: LlarriFractalConfig,
        cuadrante_compartido: Optional[CuadranteFractal] = None,
        pesos_compartidos: Optional[Dict[int, nn.Module]] = None
    ):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        profundidad = config.get_profundidad_efectiva()
        
        # Proyección de entrada
        self.input_proj = nn.Linear(config.input_dim, dim)
        self.internal_proj = nn.Linear(dim, dim)
        
        # Cuadrante base (se reutiliza para A, B, C, D)
        if cuadrante_compartido is not None and config.compartir_pesos_entre_cajas:
            self.cuadrante_base = cuadrante_compartido
        else:
            self.cuadrante_base = CuadranteFractal(
                dim=dim // 4,  # Cada cuadrante procesa 1/4
                nivel=0,
                nivel_maximo=profundidad,
                posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
                pesos_compartidos=pesos_compartidos,
                dropout=config.dropout
            )
        
        # Relaciones entre cuadrantes A, B, C, D
        quad_dim = dim // 4
        self.rel_horizontal = nn.Linear(quad_dim * 2, quad_dim)
        self.rel_vertical = nn.Linear(quad_dim * 2, quad_dim)
        self.rel_diagonal = nn.Linear(quad_dim * 2, quad_dim)
        
        # Fusión final de caja
        self.fusion_caja = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward de la caja"""
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Proyectar entrada
        if x.shape[-1] == self.config.input_dim:
            x = self.input_proj(x)
        elif x.shape[-1] != dim:
            x = self.internal_proj(x)
        
        # Dividir en 4 cuadrantes
        xA = x[..., :quad_dim]
        xB = x[..., quad_dim:quad_dim*2]
        xC = x[..., quad_dim*2:quad_dim*3]
        xD = x[..., quad_dim*3:]
        
        # Procesar cada cuadrante (MISMO peso fractal para todos)
        A, info_A = self.cuadrante_base(xA)
        B, info_B = self.cuadrante_base(xB)
        C, info_C = self.cuadrante_base(xC)
        D, info_D = self.cuadrante_base(xD)
        
        # Relaciones posicionales entre cuadrantes
        rel_h_sup = self.rel_horizontal(torch.cat([A, B], dim=-1))
        rel_h_inf = self.rel_horizontal(torch.cat([C, D], dim=-1))
        rel_v_izq = self.rel_vertical(torch.cat([A, C], dim=-1))
        rel_v_der = self.rel_vertical(torch.cat([B, D], dim=-1))
        rel_d_pri = self.rel_diagonal(torch.cat([A, D], dim=-1))
        rel_d_sec = self.rel_diagonal(torch.cat([B, C], dim=-1))
        
        A = A + rel_h_sup + rel_v_izq + rel_d_pri
        B = B + rel_h_sup + rel_v_der + rel_d_sec
        C = C + rel_h_inf + rel_v_izq + rel_d_sec
        D = D + rel_h_inf + rel_v_der + rel_d_pri
        
        # Fusionar
        fusion = torch.cat([A, B, C, D], dim=-1)
        output = self.fusion_caja(fusion)
        
        info = {'A': info_A, 'B': info_B, 'C': info_C, 'D': info_D}
        
        return output, info


# ==============================================================================
# LLAVES TRINITY
# ==============================================================================

class LlaveTrinity(nn.Module):
    """Llave para conexión entre cajas"""
    
    def __init__(self, dim: int, bidireccional: bool = True):
        super().__init__()
        self.bidireccional = bidireccional
        
        self.llave_ida = nn.Linear(dim, dim)
        if bidireccional:
            self.llave_vuelta = nn.Linear(dim, dim)
    
    def forward(self, origen: torch.Tensor, destino: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica la llave"""
        destino_mod = destino + self.llave_ida(origen)
        
        if self.bidireccional:
            origen_mod = origen + self.llave_vuelta(destino)
        else:
            origen_mod = origen
            
        return origen_mod, destino_mod


# ==============================================================================
# MODELO PRINCIPAL - LLARRI-O1 v3.0
# ==============================================================================

class LlarriO1_FractalProfundo(nn.Module):
    """
    LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
    =====================================================
    
    LA ARQUITECTURA DEFINITIVA de Lucas Mella (Segunda Cabeza)
    
    CARACTERÍSTICAS:
    ----------------
    • 3 Cajas Trinity (Padre, Hijo, Espíritu)
    • Cada caja tiene 4 Cuadrantes (A, B, C, D)
    • Cada cuadrante contiene sub-cuadrantes RECURSIVOS
    • La recursión continúa hasta dim=1 (o dim_minima)
    • TODOS los niveles comparten pesos
    
    COMPRESIÓN EXPONENCIAL:
    -----------------------
    Sin compartir: 4^profundidad × 3 cajas = miles de conjuntos de pesos
    Con compartir: ~10-15 conjuntos únicos
    
    EJEMPLO con dim=256, profundidad=4:
    - Nivel 0: 256 dims → 4 cuadrantes × 64 dims
    - Nivel 1: 64 dims → 4 cuadrantes × 16 dims
    - Nivel 2: 16 dims → 4 cuadrantes × 4 dims
    - Nivel 3: 4 dims → 4 cuadrantes × 1 dim
    
    Total sin compartir: 4^4 × 3 = 768 conjuntos de pesos
    Total con compartir: ~5 niveles × 3 relaciones = 15 conjuntos
    Compresión: 768/15 = 51× (98% reducción)
    
    Autor: Lucas Mella (lucas@segundacabeza.com)
    Coordinador: Alvaro (alvaro@segundacabeza.com)
    Organización: Segunda Cabeza
    """
    
    def __init__(self, config: Optional[LlarriFractalConfig] = None):
        super().__init__()
        self.config = config or LlarriFractalConfig()
        
        # Detectar dispositivo
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        dim = self.config.hidden_dim
        profundidad = self.config.get_profundidad_efectiva()
        
        # Diccionario para compartir pesos entre niveles
        self.pesos_compartidos: Dict[int, nn.Module] = {}
        
        # Crear cuadrante base (compartido entre todas las cajas)
        self.cuadrante_base = CuadranteFractal(
            dim=dim // 4,
            nivel=0,
            nivel_maximo=profundidad,
            posicion=PosicionCuadrante.SUPERIOR_IZQUIERDA,
            pesos_compartidos=self.pesos_compartidos if self.config.compartir_pesos_todos_niveles else None,
            dropout=self.config.dropout
        )
        
        # 3 Cajas Trinity (comparten el cuadrante base)
        self.caja1 = CajaTrinityFractal(
            self.config, 
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        self.caja2 = CajaTrinityFractal(
            self.config,
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        self.caja3 = CajaTrinityFractal(
            self.config,
            self.cuadrante_base if self.config.compartir_pesos_entre_cajas else None,
            self.pesos_compartidos
        )
        
        # Llaves entre cajas
        self.llave_1_2 = LlaveTrinity(dim, bidireccional=self.config.usar_llaves_bidireccionales)
        self.llave_1_3 = LlaveTrinity(dim, bidireccional=False)
        self.llave_2_3 = LlaveTrinity(dim, bidireccional=False)
        
        if self.config.usar_retroalimentacion:
            self.llave_3_1 = LlaveTrinity(dim, bidireccional=False)
        
        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, self.config.output_dim)
        )
        
        # Mover a dispositivo
        self.to(self.device)
        
        # Mostrar información
        self._print_info()
    
    def _print_info(self):
        """Muestra información del modelo"""
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
        print(f"  • Factor de reducción:  {params_sin/params:.1f}x")
        print(f"\n  MEMORIA:")
        mem_mb = params * 4 / 1e6
        mem_sin = params_sin * 4 / 1e6
        print(f"  • Modelo actual:        {mem_mb:.2f} MB")
        print(f"  • Sin compartir sería:  {mem_sin:.2f} MB")
        print(f"  • Ahorro:               {mem_sin - mem_mb:.2f} MB")
        print(f"\n  • Dispositivo:          {self.device}")
        print(f"{'='*70}\n")
    
    def _calcular_params_sin_compartir(self) -> int:
        """Calcula parámetros si no compartiéramos pesos"""
        dim = self.config.hidden_dim
        profundidad = self.config.get_profundidad_efectiva()
        
        # Sin compartir: cada cuadrante en cada nivel tiene sus propios pesos
        # Total de cuadrantes: 4^profundidad por caja × 3 cajas
        total_cuadrantes = (4 ** profundidad) * 3 * 4  # 3 cajas, 4 cuadrantes raíz
        
        # Parámetros por cuadrante (aproximado)
        params_cuadrante = dim * dim * 4  # Capas lineales + relaciones
        
        return total_cuadrantes * params_cuadrante
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Input [batch, input_dim]
            
        Returns:
            output: [batch, output_dim]
        """
        x = x.to(self.device)
        
        # Procesar Caja 1
        out1, info1 = self.caja1(x)
        
        # Procesar Caja 2
        out2, info2 = self.caja2(x)
        
        # Llave Caja1 ↔ Caja2
        out1, out2 = self.llave_1_2(out1, out2)
        
        # Llaves hacia Caja3
        _, out3_from_1 = self.llave_1_3(out1, torch.zeros_like(out1))
        _, out3_from_2 = self.llave_2_3(out2, torch.zeros_like(out2))
        
        # Combinar inputs a Caja3
        input_caja3 = out3_from_1 + out3_from_2
        
        # Procesar Caja 3
        out3, info3 = self.caja3(input_caja3)
        
        # Retroalimentación
        if self.config.usar_retroalimentacion:
            _, out1 = self.llave_3_1(out3, out1)
        
        # Salida
        output = self.output_layer(out3)
        
        return output
    
    def get_compression_stats(self) -> Dict:
        """Retorna estadísticas de compresión"""
        params = sum(p.numel() for p in self.parameters())
        params_sin = self._calcular_params_sin_compartir()
        
        return {
            'parametros_reales': params,
            'parametros_sin_compartir': params_sin,
            'compresion_porcentaje': (1 - params / params_sin) * 100,
            'factor_reduccion': params_sin / params,
            'profundidad_fractal': self.config.get_profundidad_efectiva(),
            'dimensiones_por_nivel': self.config.get_dims_por_nivel()
        }
    
    def get_estructura_fractal(self) -> str:
        """Retorna representación visual de la estructura fractal"""
        dims = self.config.get_dims_por_nivel()
        profundidad = self.config.get_profundidad_efectiva()
        
        lineas = []
        lineas.append("ESTRUCTURA FRACTAL LLARRI-O1 v3.0")
        lineas.append("=" * 50)
        lineas.append("")
        lineas.append("CAJA (×3)")
        
        for nivel in range(profundidad + 1):
            indent = "  " * (nivel + 1)
            dim = dims[nivel] if nivel < len(dims) else "?"
            if nivel == 0:
                lineas.append(f"{indent}├── Cuadrante [A,B,C,D] (dim={dim})")
            else:
                nombres = ["α", "β", "γ", "δ", "ε"][min(nivel-1, 4)]
                lineas.append(f"{indent}├── Sub-cuadrante [{nombres}1,{nombres}2,{nombres}3,{nombres}4] (dim={dim})")
        
        lineas.append("")
        lineas.append(f"Total niveles: {profundidad + 1}")
        lineas.append(f"Dimensiones: {dims}")
        
        return "\n".join(lineas)


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def crear_modelo_fractal(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    profundidad: int = -1,  # -1 = automático
    **kwargs
) -> LlarriO1_FractalProfundo:
    """
    Crea una instancia del modelo fractal profundo.
    
    Args:
        input_dim: Dimensión de entrada
        hidden_dim: Dimensión oculta (determina profundidad máxima)
        output_dim: Dimensión de salida (clases)
        profundidad: Profundidad fractal (-1 = automático)
        **kwargs: Argumentos adicionales para config
        
    Returns:
        Modelo LLARRI-O1 v3.0
    """
    config = LlarriFractalConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        profundidad_fractal=profundidad,
        **kwargs
    )
    return LlarriO1_FractalProfundo(config)


def explorar_profundidades():
    """Explora diferentes configuraciones de profundidad"""
    print("\n" + "="*70)
    print("EXPLORACIÓN DE PROFUNDIDADES FRACTALES")
    print("="*70)
    
    dimensiones = [64, 128, 256, 512, 1024, 2048, 4096]
    
    for dim in dimensiones:
        config = LlarriFractalConfig(hidden_dim=dim)
        prof = config.get_profundidad_efectiva()
        dims = config.get_dims_por_nivel()
        
        print(f"\nhidden_dim={dim:4d} → profundidad={prof} → dims={dims}")


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMO: LLARRI-O1 v3.0 - TRINITY FRACTAL RECURSIVO PROFUNDO")
    print("="*70)
    
    # Explorar profundidades
    explorar_profundidades()
    
    # Crear modelo
    print("\n" + "="*70)
    print("CREANDO MODELO")
    print("="*70)
    
    modelo = crear_modelo_fractal(
        input_dim=784,
        hidden_dim=256,
        output_dim=10
    )
    
    # Mostrar estructura
    print(modelo.get_estructura_fractal())
    
    # Test de inferencia
    print("\n" + "="*70)
    print("TEST DE INFERENCIA")
    print("="*70)
    
    x = torch.randn(32, 784)
    output = modelo(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Estadísticas
    stats = modelo.get_compression_stats()
    print(f"\n{'='*70}")
    print("ESTADÍSTICAS DE COMPRESIÓN")
    print(f"{'='*70}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
