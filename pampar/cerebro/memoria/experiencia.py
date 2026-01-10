# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Memoria de Experiencia - Aprendizaje a partir de resultados

Este módulo implementa:
1. Memoria de éxitos: qué configuraciones funcionaron bien
2. Memoria de fracasos: qué evitar
3. Reflexión: analizar por qué algo funcionó o no

Filosofía: "Aprender de la experiencia, no solo de los datos"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class Experiencia:
    """
    Registro de una experiencia (éxito o fracaso).
    
    Atributos:
        contexto: Vector que resume el contexto de entrada
        accion: Vector que resume la acción/salida tomada
        resultado: Score de qué tan bien salió (0 a 1)
        timestamp: Cuándo ocurrió (para decay temporal)
        metadata: Info adicional (tokens, perplexity, etc.)
    """
    contexto: torch.Tensor
    accion: torch.Tensor
    resultado: float
    timestamp: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def es_exito(self, umbral: float = 0.5) -> bool:
        return self.resultado >= umbral
    
    def es_fracaso(self, umbral: float = 0.3) -> bool:
        return self.resultado < umbral


class MemoriaExperiencia(nn.Module):
    """
    Almacén de experiencias para aprendizaje práctico.
    
    Mantiene:
    - Buffer de experiencias recientes (limitado por memoria)
    - Prototipos de éxitos/fracasos (aprendidos)
    - Mecanismo de consulta para guiar decisiones futuras
    
    La memoria es diferenciable para que pueda aprenderse.
    """
    
    def __init__(
        self, 
        dim: int, 
        capacidad: int = 1000,
        n_prototipos: int = 32,
        umbral_exito: float = 0.6,
        umbral_fracaso: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.capacidad = capacidad
        self.umbral_exito = umbral_exito
        self.umbral_fracaso = umbral_fracaso
        
        # Buffer de experiencias (no diferenciable, solo almacenamiento)
        self.experiencias: deque = deque(maxlen=capacidad)
        self.contador = 0
        
        # ============================================
        # Prototipos APRENDIBLES
        # ============================================
        # Prototipos de contextos exitosos
        self.prototipos_exito = nn.Parameter(
            torch.randn(n_prototipos, dim) * 0.1
        )
        # Prototipos de contextos fracasados
        self.prototipos_fracaso = nn.Parameter(
            torch.randn(n_prototipos, dim) * 0.1
        )
        
        # Atención para consultar prototipos
        self.query_exito = nn.Linear(dim, dim)
        self.query_fracaso = nn.Linear(dim, dim)
        
        # ============================================
        # Predictor de resultado
        # ============================================
        self.predictor_resultado = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # Refinador: modifica entrada basado en memoria
        # ============================================
        self.refinador = nn.Sequential(
            nn.Linear(dim * 3, dim),  # entrada + info_exito + info_fracaso
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        # Peso de cuánto influye la memoria
        self.peso_memoria = nn.Parameter(torch.tensor(0.3))
    
    def registrar(
        self, 
        contexto: torch.Tensor, 
        accion: torch.Tensor, 
        resultado: float,
        metadata: Optional[Dict] = None
    ):
        """
        Registra una nueva experiencia.
        
        Args:
            contexto: Vector de contexto (detached de gradientes)
            accion: Vector de acción/salida
            resultado: Score de 0 a 1
            metadata: Info adicional
        """
        exp = Experiencia(
            contexto=contexto.detach().cpu(),
            accion=accion.detach().cpu(),
            resultado=resultado,
            timestamp=self.contador,
            metadata=metadata or {},
        )
        self.experiencias.append(exp)
        self.contador += 1
        
        # Actualizar prototipos basado en la experiencia
        self._actualizar_prototipos(exp)
    
    def _actualizar_prototipos(self, exp: Experiencia):
        """Actualiza prototipos con la nueva experiencia (sin gradientes)."""
        with torch.no_grad():
            ctx = exp.contexto.to(self.prototipos_exito.device)
            
            if exp.es_exito(self.umbral_exito):
                # Encontrar prototipo de éxito más cercano y acercarlo
                dists = torch.cdist(ctx.unsqueeze(0), self.prototipos_exito)
                idx = dists.argmin()
                self.prototipos_exito.data[idx] = (
                    0.9 * self.prototipos_exito.data[idx] + 
                    0.1 * ctx
                )
            elif exp.es_fracaso(self.umbral_fracaso):
                # Similar para fracasos
                dists = torch.cdist(ctx.unsqueeze(0), self.prototipos_fracaso)
                idx = dists.argmin()
                self.prototipos_fracaso.data[idx] = (
                    0.9 * self.prototipos_fracaso.data[idx] + 
                    0.1 * ctx
                )
    
    def consultar_exitos(self, x: torch.Tensor) -> torch.Tensor:
        """
        Consulta la memoria de éxitos.
        
        Args:
            x: (batch, seq, dim) o (batch, dim)
            
        Returns:
            info_exitos: Información relevante de experiencias exitosas
        """
        shape_original = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch, seq, dim = x.shape
        
        # Query
        q = self.query_exito(x)  # (batch, seq, dim)
        
        # Atención sobre prototipos de éxito
        scores = torch.matmul(q, self.prototipos_exito.T)  # (batch, seq, n_proto)
        attn = F.softmax(scores / (dim ** 0.5), dim=-1)
        
        # Información agregada
        info = torch.matmul(attn, self.prototipos_exito)  # (batch, seq, dim)
        
        if len(shape_original) == 2:
            info = info.squeeze(1)
        
        return info
    
    def consultar_fracasos(self, x: torch.Tensor) -> torch.Tensor:
        """
        Consulta la memoria de fracasos (para evitarlos).
        """
        shape_original = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch, seq, dim = x.shape
        
        q = self.query_fracaso(x)
        scores = torch.matmul(q, self.prototipos_fracaso.T)
        attn = F.softmax(scores / (dim ** 0.5), dim=-1)
        info = torch.matmul(attn, self.prototipos_fracaso)
        
        if len(shape_original) == 2:
            info = info.squeeze(1)
        
        return info
    
    def predecir_resultado(
        self, 
        contexto: torch.Tensor, 
        accion_propuesta: torch.Tensor
    ) -> torch.Tensor:
        """
        Predice qué tan bien resultará una acción dado un contexto.
        
        Útil para guiar la generación: preferir acciones con alto score predicho.
        """
        combinado = torch.cat([contexto, accion_propuesta], dim=-1)
        return self.predictor_resultado(combinado)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refina la entrada basándose en memoria de experiencias.
        
        Args:
            x: (batch, seq, dim) entrada a refinar
            
        Returns:
            x_refinado: entrada modificada por la memoria
        """
        # Consultar memorias
        info_exito = self.consultar_exitos(x)
        info_fracaso = self.consultar_fracasos(x)
        
        # Combinar información
        combinado = torch.cat([x, info_exito, info_fracaso], dim=-1)
        refinamiento = self.refinador(combinado)
        
        # Aplicar con peso
        peso = torch.sigmoid(self.peso_memoria)
        x_refinado = x + peso * refinamiento
        
        return x_refinado
    
    def similitud_con_fracasos(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula qué tan similar es x a patrones de fracaso.
        Útil para penalizar generaciones similares a fracasos pasados.
        """
        if x.dim() == 3:
            x_flat = x.mean(dim=1)  # (batch, dim)
        else:
            x_flat = x
        
        # Distancia a prototipos de fracaso
        dists = torch.cdist(x_flat.unsqueeze(0), self.prototipos_fracaso)
        min_dist = dists.min(dim=-1).values.squeeze(0)
        
        # Convertir distancia a similitud (más cercano = más similar)
        similitud = 1.0 / (1.0 + min_dist)
        return similitud
    
    def guardar(self, path: str):
        """Guarda las experiencias a disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'experiencias': [
                {
                    'contexto': exp.contexto.tolist(),
                    'accion': exp.accion.tolist(),
                    'resultado': exp.resultado,
                    'timestamp': exp.timestamp,
                    'metadata': exp.metadata,
                }
                for exp in self.experiencias
            ],
            'contador': self.contador,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def cargar(self, path: str):
        """Carga experiencias de disco."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.experiencias.clear()
        for exp_data in data['experiencias']:
            exp = Experiencia(
                contexto=torch.tensor(exp_data['contexto']),
                accion=torch.tensor(exp_data['accion']),
                resultado=exp_data['resultado'],
                timestamp=exp_data['timestamp'],
                metadata=exp_data['metadata'],
            )
            self.experiencias.append(exp)
        
        self.contador = data['contador']
    
    def obtener_estadisticas(self) -> Dict:
        """Obtiene estadísticas de la memoria."""
        if not self.experiencias:
            return {
                'total': 0,
                'exitos': 0,
                'fracasos': 0,
                'promedio_resultado': 0.0,
            }
        
        resultados = [e.resultado for e in self.experiencias]
        return {
            'total': len(self.experiencias),
            'exitos': sum(1 for r in resultados if r >= self.umbral_exito),
            'fracasos': sum(1 for r in resultados if r < self.umbral_fracaso),
            'promedio_resultado': sum(resultados) / len(resultados),
        }
