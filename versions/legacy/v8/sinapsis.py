# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
#
# ⚠️ LEGACY CODE - For historical reference only
# Replaced by Fronteras (fronteras bidireccionales) in v9
#
"""
Sinapsis - Conexiones entre neuronas con reglas lógicas (LEGACY)

Las sinapsis:
- Conectan neuronas entre sí
- Tienen reglas de activación (cuándo transmitir)
- Pueden ser excitatorias o inhibitorias
- Son livianas (no agregan parámetros pesados)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class TipoSinapsis(Enum):
    """Tipos de conexión sináptica."""
    EXCITATORIA = "excitatoria"   # Aumenta activación del destino
    INHIBITORIA = "inhibitoria"   # Reduce activación del destino
    MODULADORA = "moduladora"     # Ajusta sin excitar/inhibir directamente


@dataclass
class ReglaSinaptica:
    """
    Regla que determina cuándo una sinapsis transmite.
    
    Ejemplo:
    - Si origen=matematicas y destino=patrones: transmitir al ver secuencias numéricas
    - Si origen=lenguaje y destino=contexto: transmitir siempre (alta correlación)
    """
    origen: str
    destino: str
    tipo: TipoSinapsis
    peso: float  # 0.0 a 1.0
    condicion: Optional[Callable] = None  # Función que evalúa si transmitir
    
    def evaluar(self, contexto: Dict) -> bool:
        """Evalúa si la sinapsis debe transmitir dado el contexto."""
        if self.condicion is None:
            return True  # Siempre transmite
        return self.condicion(contexto)


class Sinapsis(nn.Module):
    """
    Gestor de conexiones sinápticas entre neuronas.
    
    No tiene parámetros aprendibles pesados - solo reglas lógicas
    y una pequeña transformación lineal por conexión.
    """
    
    # Matriz de conexiones naturales entre módulos
    # Basada en cómo las áreas cerebrales realmente se conectan
    CONEXIONES_NATURALES = {
        # (origen, destino): (tipo, peso_base)
        ('lenguaje', 'contexto'): (TipoSinapsis.EXCITATORIA, 0.8),
        ('lenguaje', 'logica'): (TipoSinapsis.EXCITATORIA, 0.5),
        ('lenguaje', 'creatividad'): (TipoSinapsis.EXCITATORIA, 0.6),
        
        ('logica', 'matematicas'): (TipoSinapsis.EXCITATORIA, 0.7),
        ('logica', 'lenguaje'): (TipoSinapsis.EXCITATORIA, 0.4),
        ('logica', 'patrones'): (TipoSinapsis.EXCITATORIA, 0.6),
        
        ('matematicas', 'logica'): (TipoSinapsis.EXCITATORIA, 0.7),
        ('matematicas', 'patrones'): (TipoSinapsis.EXCITATORIA, 0.8),
        
        ('patrones', 'matematicas'): (TipoSinapsis.EXCITATORIA, 0.6),
        ('patrones', 'contexto'): (TipoSinapsis.EXCITATORIA, 0.5),
        ('patrones', 'creatividad'): (TipoSinapsis.EXCITATORIA, 0.4),
        
        ('contexto', 'lenguaje'): (TipoSinapsis.EXCITATORIA, 0.8),
        ('contexto', 'creatividad'): (TipoSinapsis.EXCITATORIA, 0.6),
        ('contexto', 'logica'): (TipoSinapsis.EXCITATORIA, 0.4),
        
        ('creatividad', 'lenguaje'): (TipoSinapsis.EXCITATORIA, 0.7),
        ('creatividad', 'contexto'): (TipoSinapsis.EXCITATORIA, 0.5),
        ('creatividad', 'patrones'): (TipoSinapsis.MODULADORA, 0.3),
    }
    
    def __init__(self, dim: int, modulos: List[str]):
        super().__init__()
        self.dim = dim
        self.modulos = modulos
        
        # Crear reglas sinápticas
        self.reglas: Dict[tuple, ReglaSinaptica] = {}
        self._inicializar_reglas()
        
        # Pequeñas transformaciones por conexión (livianas)
        self.transformaciones = nn.ModuleDict()
        for (origen, destino), (tipo, peso) in self.CONEXIONES_NATURALES.items():
            key = f"{origen}_a_{destino}"
            # Solo una capa lineal pequeña por conexión
            self.transformaciones[key] = nn.Linear(dim, dim, bias=False)
            # Inicializar cerca de identidad
            nn.init.eye_(self.transformaciones[key].weight)
            self.transformaciones[key].weight.data *= 0.1  # Escalar para que sea sutil
    
    def _inicializar_reglas(self):
        """Crea las reglas sinápticas basadas en conexiones naturales."""
        for (origen, destino), (tipo, peso) in self.CONEXIONES_NATURALES.items():
            self.reglas[(origen, destino)] = ReglaSinaptica(
                origen=origen,
                destino=destino,
                tipo=tipo,
                peso=peso,
            )
    
    def transmitir(
        self, 
        origen: str, 
        destino: str, 
        senal: torch.Tensor,
        contexto: Optional[Dict] = None
    ) -> Optional[torch.Tensor]:
        """
        Transmite señal de una neurona a otra.
        
        Args:
            origen: Nombre del módulo origen
            destino: Nombre del módulo destino
            senal: Tensor a transmitir (batch, seq, dim) o (batch, dim)
            contexto: Info contextual para evaluar reglas
            
        Returns:
            Señal transformada o None si no hay conexión
        """
        key = (origen, destino)
        if key not in self.reglas:
            return None
        
        regla = self.reglas[key]
        
        # Evaluar condición
        if contexto and not regla.evaluar(contexto):
            return None
        
        # Aplicar transformación
        transform_key = f"{origen}_a_{destino}"
        if transform_key in self.transformaciones:
            senal_transformada = self.transformaciones[transform_key](senal)
        else:
            senal_transformada = senal
        
        # Aplicar peso y tipo
        if regla.tipo == TipoSinapsis.EXCITATORIA:
            return senal_transformada * regla.peso
        elif regla.tipo == TipoSinapsis.INHIBITORIA:
            return -senal_transformada * regla.peso
        else:  # MODULADORA
            return senal_transformada * regla.peso * 0.5
    
    def obtener_peso(self, origen: str, destino: str) -> float:
        """Obtiene el peso de conexión entre dos módulos."""
        key = (origen, destino)
        if key in self.reglas:
            return self.reglas[key].peso
        return 0.0
    
    def obtener_conexiones_de(self, origen: str) -> List[str]:
        """Lista todos los destinos conectados desde un origen."""
        return [destino for (o, destino) in self.reglas.keys() if o == origen]
    
    def obtener_conexiones_a(self, destino: str) -> List[str]:
        """Lista todos los orígenes que conectan a un destino."""
        return [origen for (origen, d) in self.reglas.keys() if d == destino]
