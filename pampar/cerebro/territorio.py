# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Territorio - Agrupación funcional de módulos cerebrales

Los territorios son regiones cerebrales que agrupan módulos
relacionados funcionalmente. Dentro de un territorio, los módulos
comparten un buffer común y se comunican libremente.

Territorios de PampaR:
1. EXPRESIVO: Lenguaje + Creatividad (generar texto fluido)
2. CONTEXTUAL: Contexto (memoria de trabajo, coherencia)
3. FORMAL: Lógica (razonamiento, reglas)
4. ESTRUCTURAL: Patrones + Matemáticas (secuencias, números)

Filosofía:
- Intra-territorio: comunicación libre y barata (buffer compartido)
- Inter-territorio: comunicación selectiva via fronteras
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .neurona import Neurona
from .modulos.especializados import (
    NeuronaLenguaje, NeuronaLogica, NeuronaMatematicas,
    NeuronaPatrones, NeuronaContexto, NeuronaCreatividad
)


class TipoTerritorio(Enum):
    """Tipos de territorios cerebrales."""
    EXPRESIVO = "expresivo"       # Lenguaje + Creatividad
    CONTEXTUAL = "contextual"     # Contexto
    FORMAL = "formal"             # Lógica
    ESTRUCTURAL = "estructural"   # Patrones + Matemáticas


# Configuración de qué módulos pertenecen a cada territorio
CONFIGURACION_TERRITORIOS = {
    TipoTerritorio.EXPRESIVO: ['lenguaje', 'creatividad'],
    TipoTerritorio.CONTEXTUAL: ['contexto'],
    TipoTerritorio.FORMAL: ['logica'],
    TipoTerritorio.ESTRUCTURAL: ['patrones', 'matematicas'],
}


class Territorio(nn.Module):
    """
    Un territorio cerebral que agrupa módulos relacionados.
    
    Características:
    - Buffer compartido para comunicación intra-territorio
    - Los módulos leen y escriben al buffer común
    - Procesamiento eficiente (un solo espacio de trabajo)
    """
    
    def __init__(
        self, 
        tipo: TipoTerritorio,
        dim: int, 
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tipo = tipo
        self.dim = dim
        self.nombres_modulos = CONFIGURACION_TERRITORIOS[tipo]
        
        # Crear módulos según el territorio
        self.modulos = nn.ModuleDict()
        for nombre in self.nombres_modulos:
            self.modulos[nombre] = self._crear_modulo(nombre, dim, n_heads, dropout)
        
        # Buffer compartido del territorio
        # Permite comunicación libre entre módulos del mismo territorio
        self.buffer = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.buffer.weight)  # Inicializar como identidad
        
        # Gate para controlar cuánto del buffer usar
        self.buffer_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Normalización territorial
        self.norm = nn.LayerNorm(dim)
        
        # Proyección de salida del territorio
        self.salida = nn.Linear(dim, dim)
    
    def _crear_modulo(
        self, 
        nombre: str, 
        dim: int, 
        n_heads: int, 
        dropout: float
    ) -> Neurona:
        """Crea el módulo especializado correspondiente."""
        fabricas = {
            'lenguaje': NeuronaLenguaje,
            'creatividad': NeuronaCreatividad,
            'contexto': NeuronaContexto,
            'logica': NeuronaLogica,
            'patrones': NeuronaPatrones,
            'matematicas': NeuronaMatematicas,
        }
        return fabricas[nombre](dim, n_heads, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        pesos_modulos: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        senal_externa: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa entrada a través del territorio.
        
        Args:
            x: (batch, seq, dim) entrada
            pesos_modulos: dict con peso de cada módulo {nombre: (batch, seq, 1)}
            mask: máscara de atención
            senal_externa: señal de otros territorios via frontera
            
        Returns:
            salida: (batch, seq, dim) salida del territorio
            estado: (batch, seq, dim) estado del buffer (para fronteras)
        """
        batch, seq, dim = x.shape
        
        # Inicializar buffer con la entrada
        buffer = self.buffer(x)
        
        # Si hay señal externa (de otro territorio), integrarla
        if senal_externa is not None:
            # Gate para decidir cuánto de la señal externa aceptar
            contexto = torch.cat([buffer, senal_externa], dim=-1)
            gate = self.buffer_gate(contexto)
            buffer = buffer + gate * senal_externa
        
        # Procesar con cada módulo del territorio
        salida_acumulada = torch.zeros_like(x)
        
        for nombre in self.nombres_modulos:
            modulo = self.modulos[nombre]
            peso = pesos_modulos.get(nombre, torch.ones(batch, seq, 1, device=x.device) * 0.5)
            
            # El módulo procesa entrada + buffer
            entrada_modulo = x + buffer * 0.3  # Buffer influye sutilmente
            salida_modulo = modulo.procesar(entrada_modulo, mask)
            
            # Acumular salida pesada
            salida_acumulada = salida_acumulada + peso * salida_modulo
            
            # Actualizar buffer con la salida del módulo
            buffer = buffer + peso * salida_modulo * 0.1  # Actualización sutil
        
        # Normalizar y proyectar salida
        salida = self.norm(x + salida_acumulada)
        salida = self.salida(salida)
        
        # El estado del buffer se puede usar para fronteras
        estado_buffer = buffer
        
        return salida, estado_buffer
    
    def procesar_basal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesamiento basal (mínimo) cuando el territorio no es dominante.
        Solo aplica una transformación lineal rápida.
        """
        return self.buffer(x) * 0.1  # Actividad basal mínima


class GestorTerritorios(nn.Module):
    """
    Gestiona todos los territorios y coordina su procesamiento.
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # Crear los 4 territorios
        self.territorios = nn.ModuleDict({
            'expresivo': Territorio(TipoTerritorio.EXPRESIVO, dim, n_heads, dropout),
            'contextual': Territorio(TipoTerritorio.CONTEXTUAL, dim, n_heads, dropout),
            'formal': Territorio(TipoTerritorio.FORMAL, dim, n_heads, dropout),
            'estructural': Territorio(TipoTerritorio.ESTRUCTURAL, dim, n_heads, dropout),
        })
        
        # Mapeo de módulo a territorio
        self.modulo_a_territorio = {}
        for tipo, modulos in CONFIGURACION_TERRITORIOS.items():
            for modulo in modulos:
                self.modulo_a_territorio[modulo] = tipo.value
    
    def obtener_territorio(self, nombre: str) -> Territorio:
        """Obtiene un territorio por nombre."""
        return self.territorios[nombre]
    
    def obtener_territorio_de_modulo(self, nombre_modulo: str) -> str:
        """Obtiene el nombre del territorio al que pertenece un módulo."""
        return self.modulo_a_territorio.get(nombre_modulo, 'expresivo')
    
    @property
    def nombres_territorios(self) -> List[str]:
        """Lista de nombres de territorios."""
        return list(self.territorios.keys())
