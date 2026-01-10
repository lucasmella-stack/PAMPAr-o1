# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Frontera - Conexiones bidireccionales entre territorios

Las fronteras son los canales de comunicación entre territorios.
A diferencia de las sinapsis individuales (18 conexiones), las fronteras
conectan territorios completos (6 fronteras bidireccionales).

Características:
- Bidireccionales: información fluye en ambas direcciones
- Gateadas: el tálamo decide cuándo activar cada frontera
- Eficientes: solo se activan cuando 2+ territorios están activos

Fronteras de PampaR:
1. Expresivo ◄─► Contextual  (narrativa necesita contexto)
2. Expresivo ◄─► Formal      (argumentación lógica en texto)
3. Expresivo ◄─► Estructural (metáforas, ritmo, números en texto)
4. Contextual ◄─► Formal     (contexto valida lógica)
5. Contextual ◄─► Estructural (patrones en contexto)
6. Formal ◄─► Estructural    (lógica matemática)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConfigFrontera:
    """Configuración de una frontera entre territorios."""
    territorio_a: str
    territorio_b: str
    peso_base: float = 0.5  # Peso inicial de la conexión


# Definición de todas las fronteras del sistema
FRONTERAS_DEFINIDAS = [
    ConfigFrontera('expresivo', 'contextual', 0.8),    # Alta: narrativa + contexto
    ConfigFrontera('expresivo', 'formal', 0.5),        # Media: argumentación
    ConfigFrontera('expresivo', 'estructural', 0.6),   # Media-alta: números en texto
    ConfigFrontera('contextual', 'formal', 0.5),       # Media: validación
    ConfigFrontera('contextual', 'estructural', 0.6),  # Media-alta: patrones
    ConfigFrontera('formal', 'estructural', 0.7),      # Alta: lógica matemática
]


class FronteraBidireccional(nn.Module):
    """
    Conexión bidireccional entre dos territorios.
    
    Permite intercambio de información en ambas direcciones,
    con gates aprendidos que controlan el flujo.
    """
    
    def __init__(self, dim: int, config: ConfigFrontera):
        super().__init__()
        self.dim = dim
        self.config = config
        self.territorio_a = config.territorio_a
        self.territorio_b = config.territorio_b
        
        # Transformaciones para cada dirección
        self.trans_a_a_b = nn.Linear(dim, dim, bias=False)
        self.trans_b_a_a = nn.Linear(dim, dim, bias=False)
        
        # Inicializar cerca de identidad (conexión sutil al inicio)
        nn.init.eye_(self.trans_a_a_b.weight)
        nn.init.eye_(self.trans_b_a_a.weight)
        self.trans_a_a_b.weight.data *= 0.1
        self.trans_b_a_a.weight.data *= 0.1
        
        # Gates aprendidos para cada dirección
        self.gate_a_a_b = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.gate_b_a_a = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Peso base de la conexión (se puede modular)
        self.peso_base = nn.Parameter(torch.tensor(config.peso_base))
    
    def forward(
        self,
        estado_a: torch.Tensor,
        estado_b: torch.Tensor,
        activar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Intercambio bidireccional entre territorios.
        
        Args:
            estado_a: (batch, seq, dim) estado del territorio A
            estado_b: (batch, seq, dim) estado del territorio B
            activar: (batch, seq, 1) máscara de activación (del tálamo)
            
        Returns:
            senal_para_b: señal que A envía a B
            senal_para_a: señal que B envía a A
        """
        # Contexto combinado para decidir gates
        contexto_ab = torch.cat([estado_a, estado_b], dim=-1)
        contexto_ba = torch.cat([estado_b, estado_a], dim=-1)
        
        # Calcular gates
        gate_ab = self.gate_a_a_b(contexto_ab)  # (batch, seq, 1)
        gate_ba = self.gate_b_a_a(contexto_ba)  # (batch, seq, 1)
        
        # Transformar señales
        senal_para_b = self.trans_a_a_b(estado_a) * gate_ab * self.peso_base
        senal_para_a = self.trans_b_a_a(estado_b) * gate_ba * self.peso_base
        
        # Si hay máscara de activación (del tálamo), aplicarla
        if activar is not None:
            senal_para_b = senal_para_b * activar
            senal_para_a = senal_para_a * activar
        
        return senal_para_b, senal_para_a
    
    def obtener_peso_efectivo(self) -> float:
        """Retorna el peso efectivo actual de la frontera."""
        return self.peso_base.item()


class GestorFronteras(nn.Module):
    """
    Gestiona todas las fronteras entre territorios.
    
    Responsabilidades:
    - Crear y mantener las 6 fronteras bidireccionales
    - Decidir qué fronteras activar según activaciones del tálamo
    - Coordinar el intercambio de señales
    """
    
    def __init__(self, dim: int, umbral_activacion: float = 0.3):
        super().__init__()
        self.dim = dim
        self.umbral = umbral_activacion
        
        # Crear todas las fronteras
        self.fronteras = nn.ModuleDict()
        for config in FRONTERAS_DEFINIDAS:
            key = f"{config.territorio_a}_{config.territorio_b}"
            self.fronteras[key] = FronteraBidireccional(dim, config)
        
        # Índice inverso para búsqueda rápida
        self._indice_fronteras = {}
        for config in FRONTERAS_DEFINIDAS:
            self._indice_fronteras[(config.territorio_a, config.territorio_b)] = \
                f"{config.territorio_a}_{config.territorio_b}"
            self._indice_fronteras[(config.territorio_b, config.territorio_a)] = \
                f"{config.territorio_a}_{config.territorio_b}"
    
    def obtener_frontera(self, terr_a: str, terr_b: str) -> Optional[FronteraBidireccional]:
        """Obtiene la frontera entre dos territorios si existe."""
        key = self._indice_fronteras.get((terr_a, terr_b))
        if key:
            return self.fronteras[key]
        return None
    
    def decidir_cruces(
        self,
        pesos_territorios: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decide qué fronteras activar basado en activaciones de territorios.
        
        Regla: Activar frontera si AMBOS territorios tienen peso > umbral.
        
        Args:
            pesos_territorios: {nombre_territorio: (batch, seq, 1)}
            
        Returns:
            activaciones: {key_frontera: (batch, seq, 1) máscara}
        """
        activaciones = {}
        
        for config in FRONTERAS_DEFINIDAS:
            key = f"{config.territorio_a}_{config.territorio_b}"
            
            peso_a = pesos_territorios.get(config.territorio_a)
            peso_b = pesos_territorios.get(config.territorio_b)
            
            if peso_a is not None and peso_b is not None:
                # Activar si ambos superan umbral
                activo_a = (peso_a > self.umbral).float()
                activo_b = (peso_b > self.umbral).float()
                activacion = activo_a * activo_b  # AND lógico
                activaciones[key] = activacion
        
        return activaciones
    
    def intercambiar(
        self,
        estados_territorios: Dict[str, torch.Tensor],
        pesos_territorios: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Realiza el intercambio de señales entre todos los territorios.
        
        Args:
            estados_territorios: {nombre: (batch, seq, dim)} estados de cada territorio
            pesos_territorios: {nombre: (batch, seq, 1)} pesos del tálamo
            
        Returns:
            senales_entrantes: {nombre_territorio: (batch, seq, dim)} señales recibidas
        """
        # Decidir qué fronteras activar
        activaciones = self.decidir_cruces(pesos_territorios)
        
        # Acumular señales entrantes para cada territorio
        senales = {nombre: None for nombre in estados_territorios.keys()}
        
        for config in FRONTERAS_DEFINIDAS:
            key = f"{config.territorio_a}_{config.territorio_b}"
            frontera = self.fronteras[key]
            
            estado_a = estados_territorios.get(config.territorio_a)
            estado_b = estados_territorios.get(config.territorio_b)
            
            if estado_a is None or estado_b is None:
                continue
            
            # Obtener activación de esta frontera
            activar = activaciones.get(key)
            
            # Intercambio bidireccional
            senal_para_b, senal_para_a = frontera(estado_a, estado_b, activar)
            
            # Acumular señales
            if senales[config.territorio_a] is None:
                senales[config.territorio_a] = senal_para_a
            else:
                senales[config.territorio_a] = senales[config.territorio_a] + senal_para_a
            
            if senales[config.territorio_b] is None:
                senales[config.territorio_b] = senal_para_b
            else:
                senales[config.territorio_b] = senales[config.territorio_b] + senal_para_b
        
        return senales
    
    @property
    def n_fronteras(self) -> int:
        """Número total de fronteras."""
        return len(self.fronteras)
    
    def estado_fronteras(self) -> Dict[str, float]:
        """Retorna el peso efectivo de cada frontera (para debug)."""
        return {
            key: frontera.obtener_peso_efectivo()
            for key, frontera in self.fronteras.items()
        }
