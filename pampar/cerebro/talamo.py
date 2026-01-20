# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
Tálamo Territorial - El Orquestador Central v9

El tálamo cerebral es la estación de relevo que:
- Recibe TODA la información sensorial
- Decide QUÉ territorio debe procesarla
- Usa LLAVES (detectores de dominio) + atención aprendida

Filosofía: "No mezclar peras con manzanas"
- Cada módulo tiene tokens que SON SU DOMINIO
- El tálamo SABE de antemano qué va a cada módulo/territorio

Arquitectura v9:
- 4 Territorios: expresivo, contextual, formal, estructural
- 6 Módulos: lenguaje, creatividad, contexto, lógica, patrones, matemáticas
- LLAVES (70%) + Atención (30%) para routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LlaveModulo:
    """
    LLAVE: Define el dominio de un módulo.
    
    Una llave contiene:
    - Tokens específicos que activan este módulo
    - Patrones regex/prefijos que lo activan
    - Peso base de activación
    """
    nombre: str
    tokens_clave: List[int]  # IDs de tokens que activan este módulo
    patrones: List[str]      # Patrones de texto para detección
    peso_base: float = 1.0   # Peso cuando se detecta coincidencia
    peso_fondo: float = 0.1  # Peso mínimo (siempre algo de activación)


# ============================================================
# MAPEOS DE TERRITORIOS
# ============================================================

MODULO_A_TERRITORIO = {
    'lenguaje': 'expresivo',
    'creatividad': 'expresivo',
    'contexto': 'contextual',
    'logica': 'formal',
    'patrones': 'estructural',
    'matematicas': 'estructural',
}

TERRITORIOS = ['expresivo', 'contextual', 'formal', 'estructural']


class TalamoTerritorial(nn.Module):
    """
    Tálamo v9 - Orquestador que maneja territorios.
    
    Características:
    - Calcula pesos para TERRITORIOS y módulos
    - Eficiente: 4 territorios como unidad de routing principal
    - LLAVES (reglas explícitas) tienen 70% del peso
    - Atención aprendida 30% para casos ambiguos
    
    Flujo:
    1. LLAVES determinan pesos de módulos individuales
    2. Pesos de módulos se agregan a pesos de territorios
    3. Atención aprendida refina ambos niveles
    """
    
    def __init__(
        self,
        dim: int,
        vocab_size: int = 8000,
        peso_llaves: float = 0.7,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.peso_llaves = peso_llaves
        
        self.n_modulos = 6
        self.n_territorios = 4
        
        self.nombres_modulos = [
            'lenguaje', 'logica', 'matematicas',
            'patrones', 'contexto', 'creatividad'
        ]
        self.nombres_territorios = TERRITORIOS
        
        # ============================================
        # LLAVES - Reglas explícitas de dominio
        # ============================================
        self.llaves = self._crear_llaves()
        
        # Buffer para mapeo token -> módulo
        self.register_buffer('token_a_modulo', torch.zeros(vocab_size, self.n_modulos))
        
        # Matriz de agregación módulo -> territorio
        self.register_buffer('modulo_a_territorio', self._crear_matriz_agregacion())
        
        # ============================================
        # Atención aprendida (para casos ambiguos)
        # ============================================
        self.query_modulos = nn.Linear(dim, self.n_modulos)
        self.query_territorios = nn.Linear(dim, self.n_territorios)
        
        # Normalización
        self.norm = nn.LayerNorm(dim)
        
        # Estadísticas de activación
        self.register_buffer('act_territorios', torch.zeros(self.n_territorios))
        self.register_buffer('act_modulos', torch.zeros(self.n_modulos))
        self.register_buffer('n_llamadas', torch.tensor(0))
    
    def _crear_llaves(self) -> Dict[str, LlaveModulo]:
        """
        Crea las LLAVES para cada módulo.
        
        Estas son reglas EXPLÍCITAS basadas en conocimiento del dominio.
        """
        return {
            'lenguaje': LlaveModulo(
                nombre='lenguaje',
                tokens_clave=[],
                patrones=[
                    # Artículos, preposiciones, conectores
                    'el', 'la', 'los', 'las', 'un', 'una',
                    'de', 'en', 'por', 'para', 'con', 'sin',
                    'que', 'como', 'cuando', 'donde', 'quien',
                    'y', 'o', 'pero', 'aunque', 'porque',
                    # Verbos comunes
                    'es', 'son', 'fue', 'ser', 'estar',
                ],
                peso_base=1.0,
                peso_fondo=0.2,
            ),
            'logica': LlaveModulo(
                nombre='logica',
                tokens_clave=[],
                patrones=[
                    # Conectores lógicos
                    'si', 'entonces', 'porque', 'por lo tanto',
                    'sin embargo', 'aunque', 'mientras',
                    # Términos lógicos
                    'verdadero', 'falso', 'correcto', 'incorrecto',
                    'válido', 'inválido', 'necesario', 'suficiente',
                ],
                peso_base=1.2,
                peso_fondo=0.1,
            ),
            'matematicas': LlaveModulo(
                nombre='matematicas',
                tokens_clave=[],
                patrones=[
                    # Números
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    # Operadores
                    '+', '-', '*', '/', '=', '<', '>', '%',
                    # Términos matemáticos
                    'suma', 'resta', 'multiplica', 'divide',
                    'número', 'cantidad', 'valor', 'resultado',
                ],
                peso_base=1.5,
                peso_fondo=0.05,
            ),
            'patrones': LlaveModulo(
                nombre='patrones',
                tokens_clave=[],
                patrones=[
                    # Palabras de secuencia
                    'secuencia', 'patrón', 'serie', 'repetir',
                    'siguiente', 'anterior', 'primero', 'último',
                    # Palabras de orden
                    'orden', 'clasificar', 'agrupar',
                ],
                peso_base=1.0,
                peso_fondo=0.1,
            ),
            'contexto': LlaveModulo(
                nombre='contexto',
                tokens_clave=[],
                patrones=[
                    # Referencias temporales/espaciales
                    'antes', 'después', 'durante', 'mientras',
                    'aquí', 'allí', 'donde', 'cuando',
                    # Referencias
                    'este', 'ese', 'aquel', 'anterior', 'previo',
                ],
                peso_base=0.8,
                peso_fondo=0.15,
            ),
            'creatividad': LlaveModulo(
                nombre='creatividad',
                tokens_clave=[],
                patrones=[
                    # Palabras creativas
                    'imagina', 'inventa', 'crea', 'diseña',
                    'nuevo', 'original', 'único', 'diferente',
                    # Preguntas abiertas
                    'qué pasaría', 'y si', 'cómo sería',
                ],
                peso_base=0.9,
                peso_fondo=0.1,
            ),
        }
    
    def _crear_matriz_agregacion(self) -> torch.Tensor:
        """
        Crea matriz que mapea módulos a territorios.
        
        Shape: (n_modulos, n_territorios)
        Cada fila indica a qué territorio pertenece el módulo.
        """
        matriz = torch.zeros(self.n_modulos, self.n_territorios)
        
        for i, nombre_modulo in enumerate(self.nombres_modulos):
            territorio = MODULO_A_TERRITORIO[nombre_modulo]
            j = self.nombres_territorios.index(territorio)
            matriz[i, j] = 1.0
        
        return matriz
    
    def registrar_tokenizer(self, tokenizer, vocab_size: int):
        """
        Registra el tokenizer para mapear tokens a módulos.
        
        Debe llamarse después de cargar el tokenizer.
        """
        token_a_modulo = torch.zeros(vocab_size, self.n_modulos)
        
        for i, nombre in enumerate(self.nombres_modulos):
            if nombre not in self.llaves:
                continue
            
            llave = self.llaves[nombre]
            
            for patron in llave.patrones:
                try:
                    ids = tokenizer.encode(patron)
                    for token_id in ids:
                        if 0 <= token_id < vocab_size:
                            token_a_modulo[token_id, i] = llave.peso_base
                except:
                    pass
            
            token_a_modulo[:, i] = torch.clamp(
                token_a_modulo[:, i],
                min=llave.peso_fondo
            )
        
        # Normalizar
        sumas = token_a_modulo.sum(dim=1, keepdim=True).clamp(min=1e-6)
        token_a_modulo = token_a_modulo / sumas
        
        self.register_buffer('token_a_modulo', token_a_modulo)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Calcula pesos para territorios y módulos.
        
        Args:
            x: (batch, seq, dim) embeddings
            token_ids: (batch, seq) IDs de tokens
            
        Returns:
            pesos_territorios: {nombre: (batch, seq, 1)}
            pesos_modulos: {nombre: (batch, seq, 1)}
        """
        batch, seq, _ = x.shape
        device = x.device
        
        # ============================================
        # 1. Pesos de MÓDULOS por LLAVES
        # ============================================
        if token_ids is not None and self.token_a_modulo.shape[0] > 1:
            token_a_modulo = self.token_a_modulo.to(device)
            token_ids_clamped = token_ids.clamp(0, token_a_modulo.shape[0] - 1)
            pesos_modulos_llaves = token_a_modulo[token_ids_clamped]
        else:
            pesos_modulos_llaves = torch.ones(
                batch, seq, self.n_modulos, device=device
            ) / self.n_modulos
        
        # ============================================
        # 2. Pesos por ATENCIÓN aprendida
        # ============================================
        x_norm = self.norm(x)
        pesos_modulos_aten = F.softmax(self.query_modulos(x_norm), dim=-1)
        pesos_territorios_aten = F.softmax(self.query_territorios(x_norm), dim=-1)
        
        # ============================================
        # 3. Combinar LLAVES + ATENCIÓN para módulos
        # ============================================
        pesos_modulos_tensor = (
            self.peso_llaves * pesos_modulos_llaves +
            (1 - self.peso_llaves) * pesos_modulos_aten
        )
        pesos_modulos_tensor = pesos_modulos_tensor / (
            pesos_modulos_tensor.sum(dim=-1, keepdim=True) + 1e-6
        )
        
        # ============================================
        # 4. Agregar a TERRITORIOS
        # ============================================
        modulo_a_territorio = self.modulo_a_territorio.to(device)
        pesos_territorios_agregados = torch.matmul(
            pesos_modulos_tensor, modulo_a_territorio
        )
        
        # Combinar con atención directa de territorios
        pesos_territorios_tensor = (
            0.6 * pesos_territorios_agregados +
            0.4 * pesos_territorios_aten
        )
        pesos_territorios_tensor = pesos_territorios_tensor / (
            pesos_territorios_tensor.sum(dim=-1, keepdim=True) + 1e-6
        )
        
        # ============================================
        # 5. Convertir a diccionarios
        # ============================================
        pesos_territorios = {
            nombre: pesos_territorios_tensor[:, :, i:i+1]
            for i, nombre in enumerate(self.nombres_territorios)
        }
        
        pesos_modulos = {
            nombre: pesos_modulos_tensor[:, :, i:i+1]
            for i, nombre in enumerate(self.nombres_modulos)
        }
        
        # Estadísticas
        if self.training:
            self.act_territorios += pesos_territorios_tensor.mean(dim=(0, 1)).detach()
            self.act_modulos += pesos_modulos_tensor.mean(dim=(0, 1)).detach()
            self.n_llamadas += 1
        
        return pesos_territorios, pesos_modulos
    
    def obtener_estadisticas(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de activación."""
        if self.n_llamadas == 0:
            return {
                'territorios': {n: 0.0 for n in self.nombres_territorios},
                'modulos': {n: 0.0 for n in self.nombres_modulos},
            }
        
        prom_terr = self.act_territorios / self.n_llamadas
        prom_mod = self.act_modulos / self.n_llamadas
        
        return {
            'territorios': {
                n: prom_terr[i].item()
                for i, n in enumerate(self.nombres_territorios)
            },
            'modulos': {
                n: prom_mod[i].item()
                for i, n in enumerate(self.nombres_modulos)
            },
        }
    
    def reset_estadisticas(self):
        """Reinicia estadísticas."""
        self.act_territorios.zero_()
        self.act_modulos.zero_()
        self.n_llamadas.zero_()


# Alias para compatibilidad de imports
Talamo = TalamoTerritorial
