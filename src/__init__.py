"""
LLARRI-O1 v3.0 - Trinity Fractal Recursivo Profundo
====================================================

Paquete principal de LLARRI-O1

La arquitectura de IA más comprimida del mundo.
Recursión fractal hasta el límite matemático.

Autor: Lucas Mella (Segunda Cabeza)
Licencia: Propietaria con atribución
"""

# Modelo v3.0 - Fractal Profundo (PRINCIPAL)
from .llarri_o1_fractal_profundo import (
    LlarriO1_FractalProfundo,
    LlarriFractalConfig,
    CuadranteFractal,
    CajaTrinityFractal,
    LlaveTrinity,
    PosicionCuadrante,
    crear_modelo_fractal,
    explorar_profundidades
)

# Modelo v2.0 - Legacy (compatibilidad)
from .llarri_o1_v2 import (
    LlarriO1_v2,
    LlarriConfig,
    Cuadrante,
    SubCuadrante,
    CajaTrinity,
    crear_modelo
)

from .utils import (
    contar_parametros,
    detectar_dispositivo,
    guardar_modelo,
    cargar_modelo,
    generar_reporte_modelo,
    MetricasEntrenamiento
)

# Entrenador v3.0
from .entrenador_fractal import (
    EntrenadorFractal,
    cargar_mnist,
    entrenar_modelo_completo
)

# Entrenador Progresivo (por cuadrantes)
from .entrenador_progresivo import (
    LlarriO1_EntrenableProgresivo,
    EntrenadorProgresivo,
    entrenar_modelo_progresivo,
    cargar_mnist_reducido
)

# Entrenador v2.0 - Legacy
from .entrenador import (
    Entrenador
)

__version__ = "3.0.0"
__author__ = "Lucas Mella"
__organization__ = "Segunda Cabeza"
__email__ = "lucas@segundacabeza.com"

__all__ = [
    # ========== MODELO v3.0 (PRINCIPAL) ==========
    'LlarriO1_FractalProfundo',
    'LlarriFractalConfig',
    'crear_modelo_fractal',
    'explorar_profundidades',
    
    # Componentes v3.0
    'CuadranteFractal',
    'CajaTrinityFractal',
    'LlaveTrinity',
    'PosicionCuadrante',
    
    # Entrenamiento v3.0
    'EntrenadorFractal',
    'entrenar_modelo_completo',
    'cargar_mnist',
    
    # ========== MODELO v2.0 (LEGACY) ==========
    'LlarriO1_v2',
    'LlarriConfig',
    'crear_modelo',
    'Cuadrante',
    'SubCuadrante', 
    'CajaTrinity',
    'Entrenador',
    
    # ========== UTILIDADES ==========
    'contar_parametros',
    'detectar_dispositivo',
    'guardar_modelo',
    'cargar_modelo',
    'generar_reporte_modelo',
    'MetricasEntrenamiento',
]
