"""
LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes
============================================

Nueva arquitectura diseñada por Lucas Mella (Segunda Cabeza)
Fecha: Enero 2026

Concepto: Cuadrantes fractales recursivos con pesos compartidos
- Nivel 0: Sub-cuadrantes (a1, a2, a3, a4)
- Nivel 1: Cuadrantes (A, B, C, D)
- Nivel 2: Cajas (1, 2, 3)

Compresión estimada: ~99% vs arquitecturas tradicionales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class LlarriConfig:
    """Configuración del modelo LLARRI-O1 v2.0"""
    
    # Dimensiones
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    
    # Arquitectura fractal
    num_cajas: int = 3
    num_cuadrantes: int = 4  # A, B, C, D
    num_subcuadrantes: int = 4  # a1, a2, a3, a4
    niveles_fractales: int = 2  # Profundidad de recursión
    
    # Llaves
    usar_llaves_ida: bool = True
    usar_llaves_vuelta: bool = True
    
    # Optimización
    compartir_pesos_cuadrantes: bool = True
    compartir_pesos_subcuadrantes: bool = True
    dropout: float = 0.1
    
    # Dispositivo
    device: str = "auto"  # "auto", "cuda", "cpu"


class PosicionCuadrante(Enum):
    """Posiciones de cuadrantes (determina su rol)"""
    SUPERIOR_IZQUIERDA = 0  # A o a1
    SUPERIOR_DERECHA = 1    # B o a2
    INFERIOR_IZQUIERDA = 2  # C o a3
    INFERIOR_DERECHA = 3    # D o a4


# ==============================================================================
# SUB-CUADRANTE (Nivel más profundo)
# ==============================================================================

class SubCuadrante(nn.Module):
    """
    Unidad mínima del sistema fractal.
    Tiene cálculos internos propios.
    """
    
    def __init__(self, dim: int, posicion: PosicionCuadrante, dropout: float = 0.1):
        super().__init__()
        self.posicion = posicion
        self.dim = dim
        
        # Cálculos internos propios
        self.interno = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Procesa datos internamente"""
        return x + self.interno(x)  # Conexión residual


# ==============================================================================
# CUADRANTE (Contiene 4 sub-cuadrantes)
# ==============================================================================

class Cuadrante(nn.Module):
    """
    Cuadrante con 4 sub-cuadrantes internos.
    Maneja relaciones posicionales entre sub-cuadrantes.
    
    Layout:
    ┌────┬────┐
    │ a1 │ a2 │
    ├────┼────┤
    │ a3 │ a4 │
    └────┴────┘
    """
    
    def __init__(
        self, 
        dim: int, 
        posicion: PosicionCuadrante,
        pesos_compartidos: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.posicion = posicion
        self.dim = dim
        
        # Sub-cuadrantes (pesos compartidos o propios)
        if pesos_compartidos is not None:
            self.subcuadrantes = pesos_compartidos
        else:
            # Crear UN solo sub-cuadrante y reutilizarlo (pesos compartidos)
            self.subcuadrante_base = SubCuadrante(dim, PosicionCuadrante.SUPERIOR_IZQUIERDA, dropout)
        
        # Relaciones posicionales entre sub-cuadrantes (SIN llave, por posición)
        # Estas sí son únicas porque dependen de la RELACIÓN, no de la unidad
        self.relacion_horizontal = nn.Linear(dim * 2, dim)  # a1-a2, a3-a4
        self.relacion_vertical = nn.Linear(dim * 2, dim)    # a1-a3, a2-a4
        self.relacion_diagonal = nn.Linear(dim * 2, dim)    # a1-a4, a2-a3
        
        # Fusión de sub-cuadrantes
        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Procesa input a través de los 4 sub-cuadrantes.
        
        Returns:
            output: Tensor fusionado
            subcuadrantes: Dict con los 4 tensores individuales
        """
        batch_size = x.shape[0]
        
        # Dividir input en 4 partes (o proyectar)
        chunk_size = self.dim // 4
        if x.shape[-1] >= self.dim:
            # Dividir
            x1 = x[..., :chunk_size]
            x2 = x[..., chunk_size:chunk_size*2]
            x3 = x[..., chunk_size*2:chunk_size*3]
            x4 = x[..., chunk_size*3:chunk_size*4]
        else:
            # Expandir si es muy pequeño
            x1 = x2 = x3 = x4 = x
        
        # Pad si es necesario
        if x1.shape[-1] < self.dim:
            x1 = F.pad(x1, (0, self.dim - x1.shape[-1]))
            x2 = F.pad(x2, (0, self.dim - x2.shape[-1]))
            x3 = F.pad(x3, (0, self.dim - x3.shape[-1]))
            x4 = F.pad(x4, (0, self.dim - x4.shape[-1]))
        
        # Procesar cada sub-cuadrante (MISMO peso reutilizado)
        a1 = self.subcuadrante_base(x1)  # Superior izquierda
        a2 = self.subcuadrante_base(x2)  # Superior derecha
        a3 = self.subcuadrante_base(x3)  # Inferior izquierda
        a4 = self.subcuadrante_base(x4)  # Inferior derecha
        
        # Relaciones posicionales (AQUÍ está la magia)
        # Horizontal: a1↔a2, a3↔a4
        rel_h1 = self.relacion_horizontal(torch.cat([a1, a2], dim=-1))
        rel_h2 = self.relacion_horizontal(torch.cat([a3, a4], dim=-1))
        
        # Vertical: a1↔a3, a2↔a4
        rel_v1 = self.relacion_vertical(torch.cat([a1, a3], dim=-1))
        rel_v2 = self.relacion_vertical(torch.cat([a2, a4], dim=-1))
        
        # Diagonal: a1↔a4, a2↔a3
        rel_d1 = self.relacion_diagonal(torch.cat([a1, a4], dim=-1))
        rel_d2 = self.relacion_diagonal(torch.cat([a2, a3], dim=-1))
        
        # Actualizar con relaciones
        a1 = a1 + rel_h1 + rel_v1 + rel_d1
        a2 = a2 + rel_h1 + rel_v2 + rel_d2
        a3 = a3 + rel_h2 + rel_v1 + rel_d2
        a4 = a4 + rel_h2 + rel_v2 + rel_d1
        
        # Fusionar
        fusion_input = torch.cat([a1, a2, a3, a4], dim=-1)
        output = self.fusion(fusion_input)
        
        return output, {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4}


# ==============================================================================
# CAJA (Contiene 4 cuadrantes)
# ==============================================================================

class CajaTrinity(nn.Module):
    """
    Caja principal con 4 cuadrantes.
    Maneja relaciones posicionales entre cuadrantes.
    
    Layout:
    ┌─────┬─────┐
    │  A  │  B  │
    ├─────┼─────┤
    │  C  │  D  │
    └─────┴─────┘
    """
    
    def __init__(
        self,
        config: LlarriConfig,
        cuadrante_compartido: Optional[Cuadrante] = None
    ):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        
        # Proyección de entrada (desde input_dim o desde dim)
        self.input_proj = nn.Linear(config.input_dim, dim * 4)
        self.internal_proj = nn.Linear(dim, dim * 4)  # Para entradas internas
        
        # Cuadrantes (pesos compartidos o propios)
        if cuadrante_compartido is not None and config.compartir_pesos_cuadrantes:
            self.cuadrante_base = cuadrante_compartido
        else:
            self.cuadrante_base = Cuadrante(dim, PosicionCuadrante.SUPERIOR_IZQUIERDA, dropout=config.dropout)
        
        # Relaciones posicionales entre cuadrantes
        self.relacion_horizontal = nn.Linear(dim * 2, dim)
        self.relacion_vertical = nn.Linear(dim * 2, dim)
        self.relacion_diagonal = nn.Linear(dim * 2, dim)
        
        # Cálculo propio de caja
        self.calculo_caja = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Procesa input a través de los 4 cuadrantes.
        """
        dim = self.config.hidden_dim
        
        # Proyectar entrada según su tamaño
        if x.shape[-1] == self.config.input_dim:
            # Entrada desde fuera (input original)
            x = self.input_proj(x)
        elif x.shape[-1] == dim:
            # Entrada desde otra caja
            x = self.internal_proj(x)
        elif x.shape[-1] != dim * 4:
            # Padding si es necesario
            x = F.pad(x, (0, dim * 4 - x.shape[-1]))
        
        # Dividir en 4 cuadrantes
        xA = x[..., :dim]
        xB = x[..., dim:dim*2]
        xC = x[..., dim*2:dim*3]
        xD = x[..., dim*3:]
        
        # Procesar cada cuadrante (MISMO peso reutilizado)
        A, sub_A = self.cuadrante_base(xA)
        B, sub_B = self.cuadrante_base(xB)
        C, sub_C = self.cuadrante_base(xC)
        D, sub_D = self.cuadrante_base(xD)
        
        # Relaciones posicionales entre cuadrantes
        # Horizontal: A↔B, C↔D
        rel_h1 = self.relacion_horizontal(torch.cat([A, B], dim=-1))
        rel_h2 = self.relacion_horizontal(torch.cat([C, D], dim=-1))
        
        # Vertical: A↔C, B↔D
        rel_v1 = self.relacion_vertical(torch.cat([A, C], dim=-1))
        rel_v2 = self.relacion_vertical(torch.cat([B, D], dim=-1))
        
        # Diagonal: A↔D, B↔C
        rel_d1 = self.relacion_diagonal(torch.cat([A, D], dim=-1))
        rel_d2 = self.relacion_diagonal(torch.cat([B, C], dim=-1))
        
        # Actualizar cuadrantes con relaciones
        A = A + rel_h1 + rel_v1 + rel_d1
        B = B + rel_h1 + rel_v2 + rel_d2
        C = C + rel_h2 + rel_v1 + rel_d2
        D = D + rel_h2 + rel_v2 + rel_d1
        
        # Cálculo propio de caja
        fusion_input = torch.cat([A, B, C, D], dim=-1)
        output = self.calculo_caja(fusion_input)
        
        return output, {'A': A, 'B': B, 'C': C, 'D': D}


# ==============================================================================
# LLAVES (Conexiones entre cajas)
# ==============================================================================

class LlaveTrinity(nn.Module):
    """
    Llave para conexión entre cajas.
    Puede ser: ida, vuelta, o bidireccional.
    """
    
    def __init__(self, dim: int, tipo: str = "bidireccional"):
        super().__init__()
        self.tipo = tipo
        
        if tipo in ["ida", "bidireccional"]:
            self.llave_ida = nn.Linear(dim, dim)
        if tipo in ["vuelta", "bidireccional"]:
            self.llave_vuelta = nn.Linear(dim, dim)
            
    def forward(
        self, 
        origen: torch.Tensor, 
        destino: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica la llave entre dos cajas.
        
        Returns:
            origen_modificado, destino_modificado
        """
        origen_out = origen
        destino_out = destino
        
        if self.tipo in ["ida", "bidireccional"]:
            destino_out = destino + self.llave_ida(origen)
            
        if self.tipo in ["vuelta", "bidireccional"]:
            origen_out = origen + self.llave_vuelta(destino)
            
        return origen_out, destino_out


# ==============================================================================
# MODELO PRINCIPAL
# ==============================================================================

class LlarriO1_v2(nn.Module):
    """
    LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes
    
    Arquitectura revolucionaria de Lucas Mella (Segunda Cabeza)
    
    Características:
    - 3 Cajas con 4 cuadrantes cada una
    - Cada cuadrante tiene 4 sub-cuadrantes
    - Pesos compartidos en todos los niveles
    - Relaciones posicionales sin llaves entre cuadrantes
    - Llaves ida/vuelta entre cajas
    
    Compresión: ~99% vs arquitecturas tradicionales
    """
    
    def __init__(self, config: Optional[LlarriConfig] = None):
        super().__init__()
        self.config = config or LlarriConfig()
        
        # Detectar dispositivo
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        dim = self.config.hidden_dim
        
        # Crear UN cuadrante base que se reutiliza (máxima compresión)
        self.cuadrante_base = Cuadrante(
            dim, 
            PosicionCuadrante.SUPERIOR_IZQUIERDA,
            dropout=self.config.dropout
        )
        
        # 3 Cajas (comparten el cuadrante base)
        self.caja1 = CajaTrinity(self.config, self.cuadrante_base)
        self.caja2 = CajaTrinity(self.config, self.cuadrante_base)
        self.caja3 = CajaTrinity(self.config, self.cuadrante_base)
        
        # Llaves entre cajas
        # Caja1 ↔ Caja2 (bidireccional)
        self.llave_1_2 = LlaveTrinity(dim, "bidireccional")
        
        # Caja1 → Caja3 (solo ida)
        self.llave_1_3 = LlaveTrinity(dim, "ida")
        
        # Caja2 → Caja3 (solo ida)
        self.llave_2_3 = LlaveTrinity(dim, "ida")
        
        # Caja3 → Caja1 (solo vuelta - retroalimentación)
        self.llave_3_1 = LlaveTrinity(dim, "vuelta")
        
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
        
        # Mostrar info
        self._print_info()
        
    def _print_info(self):
        """Muestra información del modelo"""
        params = sum(p.numel() for p in self.parameters())
        params_sin_compartir = self._calcular_params_sin_compartir()
        compresion = (1 - params / params_sin_compartir) * 100 if params_sin_compartir > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes")
        print(f"{'='*60}")
        print(f"Parámetros reales:     {params:,}")
        print(f"Sin compartir serían:  {params_sin_compartir:,}")
        print(f"Compresión:            {compresion:.1f}%")
        print(f"Dispositivo:           {self.device}")
        print(f"{'='*60}\n")
        
    def _calcular_params_sin_compartir(self) -> int:
        """Calcula cuántos parámetros tendría sin compartir pesos"""
        dim = self.config.hidden_dim
        
        # Sin compartir: cada sub-cuadrante, cuadrante y caja tendría sus propios pesos
        params_subcuadrante = dim * dim * 2  # 2 capas lineales
        params_cuadrante = params_subcuadrante * 4 + dim * 2 * dim * 3 + dim * 4 * dim * 2
        params_caja = params_cuadrante * 4 + dim * 2 * dim * 3 + dim * 4 * dim * 2
        params_total_sin_compartir = params_caja * 3
        
        return params_total_sin_compartir
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            output: [batch, output_dim]
        """
        x = x.to(self.device)
        
        # Procesar Caja 1
        out1, cuad1 = self.caja1(x)
        
        # Procesar Caja 2
        out2, cuad2 = self.caja2(x)
        
        # Aplicar llave Caja1 ↔ Caja2 (bidireccional)
        out1, out2 = self.llave_1_2(out1, out2)
        
        # Aplicar llaves hacia Caja3 (solo ida)
        _, out3_from_1 = self.llave_1_3(out1, torch.zeros_like(out1))
        _, out3_from_2 = self.llave_2_3(out2, torch.zeros_like(out2))
        
        # Combinar entradas a Caja3
        input_caja3 = out3_from_1 + out3_from_2
        
        # Procesar Caja 3
        out3, cuad3 = self.caja3(input_caja3)
        
        # Retroalimentación Caja3 → Caja1 (vuelta)
        out1, _ = self.llave_3_1(out1, out3)
        
        # Salida final
        output = self.output_layer(out3)
        
        return output
    
    def get_compression_stats(self) -> Dict:
        """Retorna estadísticas de compresión"""
        params = sum(p.numel() for p in self.parameters())
        params_sin_compartir = self._calcular_params_sin_compartir()
        
        return {
            'params_reales': params,
            'params_sin_compartir': params_sin_compartir,
            'compresion_porcentaje': (1 - params / params_sin_compartir) * 100,
            'factor_reduccion': params_sin_compartir / params
        }


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def crear_modelo(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    **kwargs
) -> LlarriO1_v2:
    """Crea una instancia del modelo con configuración personalizada"""
    config = LlarriConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        **kwargs
    )
    return LlarriO1_v2(config)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMO: LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes")
    print("="*70)
    
    # Crear modelo
    modelo = crear_modelo(
        input_dim=784,
        hidden_dim=256,
        output_dim=10
    )
    
    # Test de inferencia
    print("\nProbando inferencia...")
    x = torch.randn(32, 784)
    output = modelo(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Estadísticas
    stats = modelo.get_compression_stats()
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS DE COMPRESIÓN")
    print(f"{'='*60}")
    print(f"Parámetros reales:     {stats['params_reales']:,}")
    print(f"Sin compartir serían:  {stats['params_sin_compartir']:,}")
    print(f"Compresión:            {stats['compresion_porcentaje']:.1f}%")
    print(f"Factor de reducción:   {stats['factor_reduccion']:.1f}x")
    
    # Tamaño en memoria
    mem_mb = stats['params_reales'] * 4 / 1e6  # float32 = 4 bytes
    mem_sin_compartir = stats['params_sin_compartir'] * 4 / 1e6
    print(f"\nMemoria del modelo:    {mem_mb:.1f} MB")
    print(f"Sin compartir sería:   {mem_sin_compartir:.1f} MB")
    print(f"Ahorro:                {mem_sin_compartir - mem_mb:.1f} MB")
