# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 - Utilidades
======================

Funciones de utilidad para el modelo LLARRI-O1 v2.0

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


def contar_parametros(modelo: nn.Module) -> Dict[str, int]:
    """
    Cuenta los parámetros del modelo.
    
    Returns:
        Dict con parámetros totales, entrenables y no entrenables
    """
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    no_entrenables = total - entrenables
    
    return {
        'total': total,
        'entrenables': entrenables,
        'no_entrenables': no_entrenables
    }


def detectar_dispositivo(preferir_gpu: bool = True) -> torch.device:
    """
    Detecta el mejor dispositivo disponible.
    
    Args:
        preferir_gpu: Si preferir GPU cuando esté disponible
        
    Returns:
        torch.device
    """
    if preferir_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU detectada: {gpu_name} ({vram:.1f} GB)")
        return device
    else:
        print("💻 Usando CPU")
        return torch.device("cpu")


def guardar_modelo(
    modelo: nn.Module,
    ruta: str,
    config: Optional[dict] = None,
    metadata: Optional[dict] = None
):
    """
    Guarda el modelo con su configuración y metadata.
    
    Args:
        modelo: Modelo a guardar
        ruta: Ruta del archivo .pt
        config: Configuración del modelo
        metadata: Información adicional
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': modelo.state_dict(),
        'config': config,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, ruta)
    print(f"✓ Modelo guardado en {ruta}")


def cargar_modelo(
    modelo: nn.Module,
    ruta: str,
    strict: bool = True
) -> Tuple[nn.Module, dict]:
    """
    Carga un modelo desde un checkpoint.
    
    Args:
        modelo: Instancia del modelo
        ruta: Ruta del archivo .pt
        strict: Si requerir coincidencia exacta de llaves
        
    Returns:
        modelo cargado, metadata
    """
    checkpoint = torch.load(ruta, map_location='cpu')
    modelo.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    metadata = checkpoint.get('metadata', {})
    print(f"✓ Modelo cargado desde {ruta}")
    
    return modelo, metadata


def formatear_numero(num: int) -> str:
    """Formatea un número con separadores de miles"""
    return f"{num:,}"


def calcular_memoria_modelo(modelo: nn.Module, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Calcula la memoria que ocupa el modelo.
    
    Args:
        modelo: Modelo
        dtype: Tipo de dato (float32, float16, etc.)
        
    Returns:
        Dict con memoria en diferentes unidades
    """
    params = sum(p.numel() for p in modelo.parameters())
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }.get(dtype, 4)
    
    bytes_total = params * bytes_per_param
    
    return {
        'bytes': bytes_total,
        'KB': bytes_total / 1024,
        'MB': bytes_total / (1024 ** 2),
        'GB': bytes_total / (1024 ** 3)
    }


def generar_reporte_modelo(modelo) -> str:
    """
    Genera un reporte completo del modelo.
    
    Args:
        modelo: Instancia de LlarriO1_v2
        
    Returns:
        String con el reporte
    """
    params = contar_parametros(modelo)
    mem = calcular_memoria_modelo(modelo)
    
    if hasattr(modelo, 'get_compression_stats'):
        stats = modelo.get_compression_stats()
    else:
        stats = {'compresion_porcentaje': 0, 'factor_reduccion': 1}
    
    reporte = f"""
{'='*60}
REPORTE LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes
{'='*60}

📊 PARÁMETROS
   Total:        {formatear_numero(params['total'])}
   Entrenables:  {formatear_numero(params['entrenables'])}
   Congelados:   {formatear_numero(params['no_entrenables'])}

💾 MEMORIA (float32)
   Tamaño:       {mem['MB']:.2f} MB

🗜️ COMPRESIÓN
   Ahorro:       {stats['compresion_porcentaje']:.1f}%
   Factor:       {stats['factor_reduccion']:.1f}x

🏗️ ARQUITECTURA
   - 3 Cajas Trinity
   - 4 Cuadrantes por caja (A, B, C, D)
   - 4 Sub-cuadrantes por cuadrante (a1, a2, a3, a4)
   - Pesos compartidos en todos los niveles
   - Llaves bidireccionales entre cajas

{'='*60}
Diseñado por Lucas Mella - Segunda Cabeza
{'='*60}
"""
    return reporte


class MetricasEntrenamiento:
    """Clase para trackear métricas durante el entrenamiento"""
    
    def __init__(self):
        self.historial = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
    def registrar(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):
        """Registra métricas de una época"""
        self.historial['epoch'].append(epoch)
        self.historial['train_loss'].append(train_loss)
        self.historial['train_acc'].append(train_acc)
        self.historial['val_loss'].append(val_loss)
        self.historial['val_acc'].append(val_acc)
        self.historial['lr'].append(lr)
        
    def guardar(self, ruta: str):
        """Guarda el historial en JSON"""
        with open(ruta, 'w') as f:
            json.dump(self.historial, f, indent=2)
            
    def cargar(self, ruta: str):
        """Carga el historial desde JSON"""
        with open(ruta, 'r') as f:
            self.historial = json.load(f)
            
    @property
    def mejor_val_acc(self) -> float:
        """Retorna la mejor accuracy de validación"""
        return max(self.historial['val_acc']) if self.historial['val_acc'] else 0
    
    @property
    def mejor_epoca(self) -> int:
        """Retorna la época con mejor accuracy"""
        if not self.historial['val_acc']:
            return 0
        idx = self.historial['val_acc'].index(max(self.historial['val_acc']))
        return self.historial['epoch'][idx]
