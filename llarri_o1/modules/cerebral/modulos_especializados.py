# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
"""
Módulos Especializados del Cerebro

Cada módulo tiene:
1. Una tarea ESPECÍFICA (no se mete en el dominio de otros)
2. Un tipo de input que SABE procesar
3. Un protocolo de comunicación claro

Módulos:
- Lenguaje: sintaxis, gramática, semántica
- Lógica: inferencias, deducciones, reglas
- Matemáticas: números, operaciones, magnitudes
- Patrones: reconocimiento de estructuras repetidas
- Contexto: visión global, "de qué se trata esto"
- Creatividad: asociaciones libres, combinaciones nuevas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class ModuloBase(nn.Module):
    """
    Clase base para módulos especializados.
    
    Cada módulo:
    - Procesa SOLO su dominio
    - Tiene un identificador de dominio
    - Puede calcular su propia relevancia para un input
    """
    
    dominio: str = "base"
    descripcion: str = "Módulo base"
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Cada módulo tiene attention propio (especializado)
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # FFN específica del módulo
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        # Normalización
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Detector de relevancia (qué tan apropiado es este módulo para el input)
        self.detector_relevancia = nn.Linear(dim, 1)
        
    def calcular_relevancia(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula qué tan relevante es este módulo para el input."""
        return torch.sigmoid(self.detector_relevancia(x.mean(dim=1)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Procesa el input según la especialización del módulo."""
        # Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class ModuloLenguaje(ModuloBase):
    """
    Procesa estructura lingüística: sintaxis, gramática, semántica.
    
    Especialización:
    - Reconoce patrones gramaticales
    - Entiende relaciones sintácticas
    - Procesa significado de palabras en contexto
    
    NO hace:
    - Cálculos matemáticos
    - Inferencias lógicas formales
    """
    
    dominio = "lenguaje"
    descripcion = "Procesa sintaxis, gramática y semántica"
    
    def __init__(self, dim: int, n_heads: int = 4, n_roles: int = 8):
        super().__init__(dim, n_heads)
        
        # Roles gramaticales (sujeto, verbo, objeto, etc.)
        self.roles = nn.Parameter(torch.randn(n_roles, dim))
        self.asignador_roles = nn.Linear(dim, n_roles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x = super().forward(x)
        
        # Asignar roles gramaticales
        pesos_roles = F.softmax(self.asignador_roles(x), dim=-1)  # (batch, seq, n_roles)
        info_roles = pesos_roles @ self.roles  # (batch, seq, dim)
        
        # Añadir información de roles
        return x + 0.1 * info_roles


class ModuloLogica(ModuloBase):
    """
    Procesa inferencias y deducciones lógicas.
    
    Especialización:
    - Relaciones: si A entonces B
    - Transitividad: A > B, B > C → A > C
    - Consistencia: detectar contradicciones
    
    NO hace:
    - Procesar gramática
    - Cálculos numéricos
    """
    
    dominio = "logica"
    descripcion = "Procesa inferencias y relaciones lógicas"
    
    def __init__(self, dim: int, n_heads: int = 4, n_relaciones: int = 8):
        super().__init__(dim, n_heads)
        
        # Relaciones lógicas aprendibles
        # Cada relación es una transformación del espacio
        self.relaciones = nn.Parameter(torch.randn(n_relaciones, dim, dim) * 0.1)
        
        # Detector de qué relación aplicar
        self.detector_relacion = nn.Linear(dim * 2, n_relaciones)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x = super().forward(x)
        
        # Para cada par de posiciones, detectar relación
        batch, seq, dim = x.shape
        
        # Simplificación: usar relación entre token actual y contexto
        contexto = x.mean(dim=1, keepdim=True).expand(-1, seq, -1)
        pares = torch.cat([x, contexto], dim=-1)  # (batch, seq, dim*2)
        
        # Qué relación aplica
        pesos_rel = F.softmax(self.detector_relacion(pares), dim=-1)  # (batch, seq, n_rel)
        
        # Aplicar relaciones ponderadas
        transformado = torch.zeros_like(x)
        for i in range(self.relaciones.shape[0]):
            # x @ R_i para cada relación
            t = torch.einsum('bsd,de->bse', x, self.relaciones[i])
            transformado += pesos_rel[..., i:i+1] * t
            
        return x + 0.1 * transformado


class ModuloMatematicas(ModuloBase):
    """
    Procesa números, cantidades y operaciones.
    
    Especialización:
    - Representación de magnitudes
    - Operaciones básicas (conceptualmente)
    - Comparaciones numéricas
    
    NO hace:
    - Análisis gramatical
    - Inferencias no numéricas
    """
    
    dominio = "matematicas"
    descripcion = "Procesa números y operaciones"
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__(dim, n_heads)
        
        # Línea numérica interna (representación de magnitudes)
        self.linea_numerica = nn.Parameter(torch.randn(100, dim) * 0.1)  # 0-99
        
        # Operaciones como bilineares
        self.suma = nn.Bilinear(dim, dim, dim)
        self.producto = nn.Bilinear(dim, dim, dim)
        
        # Detector de tipo de operación
        self.detector_op = nn.Linear(dim, 3)  # suma, producto, comparación
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x = super().forward(x)
        
        # Proyectar a espacio numérico
        similitud = x @ self.linea_numerica.T  # (batch, seq, 100)
        magnitud = similitud.softmax(dim=-1) @ self.linea_numerica  # (batch, seq, dim)
        
        # Mezclar información numérica
        return x + 0.1 * magnitud


class ModuloPatrones(ModuloBase):
    """
    Reconoce estructuras repetidas y regularidades.
    
    Especialización:
    - Detectar repeticiones
    - Reconocer ritmos y secuencias
    - Extrapolar patrones
    
    NO hace:
    - Análisis semántico
    - Cálculos precisos
    """
    
    dominio = "patrones"
    descripcion = "Reconoce estructuras repetidas"
    
    def __init__(self, dim: int, n_heads: int = 4, n_escalas: int = 4):
        super().__init__(dim, n_heads)
        
        # Convoluciones a diferentes escalas
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=2**i, padding=2**(i-1))
            for i in range(1, n_escalas + 1)
        ])
        
        # Fusión de escalas
        self.fusion = nn.Linear(dim * n_escalas, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x_att = super().forward(x)
        
        # Detectar patrones a múltiples escalas
        x_conv = x.transpose(1, 2)  # (batch, dim, seq)
        escalas = []
        for conv in self.convs:
            e = conv(x_conv)
            # Ajustar tamaño si es necesario
            if e.shape[2] != x.shape[1]:
                e = F.interpolate(e, size=x.shape[1])
            escalas.append(e)
        
        # Concatenar escalas
        multi_escala = torch.cat(escalas, dim=1).transpose(1, 2)  # (batch, seq, dim*n)
        patrones = self.fusion(multi_escala)
        
        return x_att + 0.1 * patrones


class ModuloContexto(ModuloBase):
    """
    Captura el panorama global: "de qué se trata esto".
    
    Especialización:
    - Resumen del contenido
    - Tema principal
    - Coherencia global
    
    NO hace:
    - Análisis detallado token por token
    - Cálculos específicos
    """
    
    dominio = "contexto"
    descripcion = "Captura el panorama global"
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__(dim, n_heads)
        
        # Pooling global con attention
        self.query_global = nn.Parameter(torch.randn(1, 1, dim))
        self.attention_global = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x = super().forward(x)
        
        batch = x.shape[0]
        
        # Query global atiende a toda la secuencia
        query = self.query_global.expand(batch, -1, -1)
        contexto_global, _ = self.attention_global(query, x, x)  # (batch, 1, dim)
        
        # Distribuir contexto a todos los tokens
        contexto_broadcast = contexto_global.expand(-1, x.shape[1], -1)
        
        return x + 0.1 * contexto_broadcast


class ModuloCreatividad(ModuloBase):
    """
    Genera asociaciones libres y combinaciones nuevas.
    
    Especialización:
    - Conexiones inesperadas
    - Exploración del espacio latente
    - Combinaciones novedosas
    
    NO hace:
    - Seguir reglas estrictas
    - Cálculos precisos
    """
    
    dominio = "creatividad"
    descripcion = "Genera asociaciones y combinaciones nuevas"
    
    def __init__(self, dim: int, n_heads: int = 4, ruido: float = 0.1):
        super().__init__(dim, n_heads)
        
        self.ruido_base = ruido
        
        # Proyecciones a espacios "laterales"
        self.proyeccion_lateral = nn.Linear(dim, dim)
        self.mezcla = nn.Linear(dim * 2, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Procesar con attention base
        x = super().forward(x)
        
        # Generar variación "creativa"
        if self.training:
            ruido = torch.randn_like(x) * self.ruido_base
        else:
            ruido = 0
            
        # Proyección lateral (explorar espacio)
        lateral = torch.tanh(self.proyeccion_lateral(x + ruido))
        
        # Mezclar original con lateral
        combinado = torch.cat([x, lateral], dim=-1)
        creativo = self.mezcla(combinado)
        
        return x + 0.1 * creativo
