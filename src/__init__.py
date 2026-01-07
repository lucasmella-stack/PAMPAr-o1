"""
LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes
==========================================

Paquete principal de LLARRI-O1

Autor: Lucas Mella (Segunda Cabeza)
Licencia: MIT + Attribution Required
"""

from .llarri_o1_v2 import (
    LlarriO1_v2,
    LlarriConfig,
    Cuadrante,
    SubCuadrante,
    CajaTrinity,
    LlaveTrinity,
    PosicionCuadrante,
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

from .entrenador import (
    Entrenador,
    cargar_mnist
)

__version__ = "2.0.0"
__author__ = "Lucas Mella"
__organization__ = "Segunda Cabeza"

__all__ = [
    # Modelo principal
    'LlarriO1_v2',
    'LlarriConfig',
    'crear_modelo',
    
    # Componentes
    'Cuadrante',
    'SubCuadrante', 
    'CajaTrinity',
    'LlaveTrinity',
    'PosicionCuadrante',
    
    # Entrenamiento
    'Entrenador',
    'cargar_mnist',
    
    # Utilidades
    'contar_parametros',
    'detectar_dispositivo',
    'guardar_modelo',
    'cargar_modelo',
    'generar_reporte_modelo',
    'MetricasEntrenamiento',
]
