# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 - Detector y Gestor de Recursos
==========================================

Gestión inteligente de CPU/GPU/RAM.

Principio: "No usar topadora para levantar botellas"
- GPU: Operaciones pesadas (matmul, conv)
- CPU: Operaciones ligeras (dropout, norm, activaciones)

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import gc
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ResourceInfo:
    """Información de recursos del sistema"""
    gpu_disponible: bool
    gpu_nombre: str
    vram_total_gb: float
    vram_libre_gb: float
    vram_usado_gb: float
    ram_total_gb: float
    ram_libre_gb: float
    cuda_version: str
    pytorch_version: str


class ResourceDetector:
    """
    Detecta recursos de hardware disponibles.
    
    Uso:
        detector = ResourceDetector()
        detector.print_info()
        
        if detector.puede_usar_gpu(modelo_params=1000000):
            device = torch.device("cuda")
    """
    
    def __init__(self):
        self.gpu_disponible = torch.cuda.is_available()
        self._update()
    
    def _update(self):
        """Actualiza la información de recursos"""
        self._vram_total = 0
        self._vram_libre = 0
        self._vram_usado = 0
        self._gpu_nombre = "N/A"
        self._cuda_version = "N/A"
        
        if self.gpu_disponible:
            props = torch.cuda.get_device_properties(0)
            self._gpu_nombre = props.name
            self._vram_total = props.total_memory
            self._vram_usado = torch.cuda.memory_allocated(0)
            self._vram_libre = self._vram_total - self._vram_usado
            self._cuda_version = torch.version.cuda or "N/A"
        
        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            self._ram_total = mem.total
            self._ram_libre = mem.available
        except ImportError:
            self._ram_total = 16 * 1024**3  # Asumir 16GB
            self._ram_libre = 8 * 1024**3
    
    def get_info(self) -> ResourceInfo:
        """Retorna información de recursos"""
        self._update()
        return ResourceInfo(
            gpu_disponible=self.gpu_disponible,
            gpu_nombre=self._gpu_nombre,
            vram_total_gb=self._vram_total / 1e9,
            vram_libre_gb=self._vram_libre / 1e9,
            vram_usado_gb=self._vram_usado / 1e9,
            ram_total_gb=self._ram_total / 1e9,
            ram_libre_gb=self._ram_libre / 1e9,
            cuda_version=self._cuda_version,
            pytorch_version=torch.__version__
        )
    
    def get_vram_libre(self) -> int:
        """VRAM libre en bytes"""
        self._update()
        return self._vram_libre
    
    def get_vram_total(self) -> int:
        """VRAM total en bytes"""
        return self._vram_total
    
    def get_ram_libre(self) -> int:
        """RAM libre en bytes"""
        self._update()
        return self._ram_libre
    
    def puede_usar_gpu(self, modelo_params: int, bytes_per_param: int = 4, factor_seguridad: float = 2.5) -> bool:
        """
        Verifica si un modelo cabe en GPU.
        
        Args:
            modelo_params: Número de parámetros
            bytes_per_param: Bytes por parámetro (4 para float32)
            factor_seguridad: Multiplicador para gradientes/optimizador
        """
        if not self.gpu_disponible:
            return False
        
        memoria_requerida = modelo_params * bytes_per_param * factor_seguridad
        return memoria_requerida < self.get_vram_libre() * 0.8
    
    def decidir_modo(self, modelo_params: int) -> str:
        """
        Decide el modo óptimo de ejecución.
        
        Returns:
            "gpu": Todo en GPU
            "hibrido": Pesado en GPU, ligero en CPU
            "cpu": Todo en CPU
        """
        if not self.gpu_disponible:
            return "cpu"
        
        memoria_modelo = modelo_params * 4 * 3  # float32 * (pesos + grads + optim)
        vram_libre = self.get_vram_libre()
        
        if memoria_modelo < vram_libre * 0.5:
            return "gpu"
        elif memoria_modelo < vram_libre * 0.9:
            return "hibrido"
        else:
            return "cpu"
    
    def print_info(self):
        """Imprime información de recursos"""
        info = self.get_info()
        
        print(f"\n{'='*50}")
        print("RECURSOS DEL SISTEMA")
        print(f"{'='*50}")
        print(f"PyTorch: {info.pytorch_version}")
        print(f"\nGPU:")
        print(f"  Disponible: {info.gpu_disponible}")
        if info.gpu_disponible:
            print(f"  Modelo: {info.gpu_nombre}")
            print(f"  CUDA: {info.cuda_version}")
            print(f"  VRAM Total: {info.vram_total_gb:.2f} GB")
            print(f"  VRAM Libre: {info.vram_libre_gb:.2f} GB")
            print(f"  VRAM Usado: {info.vram_usado_gb:.2f} GB")
        print(f"\nRAM:")
        print(f"  Total: {info.ram_total_gb:.1f} GB")
        print(f"  Libre: {info.ram_libre_gb:.1f} GB")
        print(f"{'='*50}\n")


class HybridMemoryManager:
    """
    Gestor de memoria híbrida CPU/GPU.
    
    Implementa el principio de "no usar topadora para botellas":
    - Mantiene tensores pesados en GPU
    - Mueve operaciones ligeras a CPU cuando hay presión de memoria
    
    Uso:
        manager = HybridMemoryManager(umbral_vram_mb=500)
        tensor_gpu = manager.to_device_pesado(tensor)
        tensor_cpu = manager.to_device_ligero(tensor)
    """
    
    def __init__(self, umbral_vram_mb: int = 500, modo: str = "auto"):
        self.detector = ResourceDetector()
        self.umbral_vram = umbral_vram_mb * 1024 * 1024
        self.modo_forzado = modo if modo != "auto" else None
        
        self._setup_devices()
    
    def _setup_devices(self):
        """Configura los dispositivos según disponibilidad"""
        if self.modo_forzado:
            if self.modo_forzado == "cpu":
                self.device_pesado = torch.device("cpu")
                self.device_ligero = torch.device("cpu")
            elif self.modo_forzado == "gpu":
                self.device_pesado = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device_ligero = self.device_pesado
            else:  # hibrido
                self.device_pesado = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device_ligero = torch.device("cpu")
        else:
            # Auto
            if torch.cuda.is_available():
                self.device_pesado = torch.device("cuda")
                # Decidir si usar híbrido
                if self.detector.get_vram_libre() < self.umbral_vram:
                    self.device_ligero = torch.device("cpu")
                else:
                    self.device_ligero = self.device_pesado
            else:
                self.device_pesado = torch.device("cpu")
                self.device_ligero = torch.device("cpu")
    
    def get_modo(self) -> str:
        """Retorna el modo actual"""
        if self.device_pesado.type == "cpu":
            return "cpu"
        elif self.device_ligero.type == "cpu":
            return "hibrido"
        else:
            return "gpu"
    
    def to_device_pesado(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mueve tensor al dispositivo para operaciones pesadas"""
        return tensor.to(self.device_pesado)
    
    def to_device_ligero(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mueve tensor al dispositivo para operaciones ligeras"""
        return tensor.to(self.device_ligero)
    
    def liberar_vram(self):
        """Libera memoria VRAM no usada"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def check_memory_pressure(self) -> bool:
        """Verifica si hay presión de memoria"""
        if not torch.cuda.is_available():
            return False
        return self.detector.get_vram_libre() < self.umbral_vram
    
    def __repr__(self) -> str:
        return f"HybridMemoryManager(modo={self.get_modo()}, pesado={self.device_pesado}, ligero={self.device_ligero})"


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def limpiar_memoria():
    """Limpia memoria GPU y ejecuta garbage collector"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memoria_usada() -> Dict[str, float]:
    """Retorna memoria usada en GB"""
    result = {"ram": 0.0, "vram": 0.0}
    
    try:
        import psutil
        result["ram"] = psutil.Process().memory_info().rss / 1e9
    except:
        pass
    
    if torch.cuda.is_available():
        result["vram"] = torch.cuda.memory_allocated() / 1e9
    
    return result


def profile_memoria(func):
    """Decorador para perfilar uso de memoria de una función"""
    def wrapper(*args, **kwargs):
        antes = get_memoria_usada()
        result = func(*args, **kwargs)
        despues = get_memoria_usada()
        
        print(f"\n[Memoria] {func.__name__}:")
        print(f"  RAM:  {antes['ram']:.2f} → {despues['ram']:.2f} GB (Δ{despues['ram']-antes['ram']:+.2f})")
        print(f"  VRAM: {antes['vram']:.2f} → {despues['vram']:.2f} GB (Δ{despues['vram']-antes['vram']:+.2f})")
        
        return result
    return wrapper


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMO: Detector de Recursos")
    print("="*50)
    
    detector = ResourceDetector()
    detector.print_info()
    
    # Test decisión de modo
    for params in [100_000, 1_000_000, 10_000_000, 100_000_000]:
        modo = detector.decidir_modo(params)
        print(f"Modelo con {params:,} params → Modo: {modo}")
    
    print("\n" + "="*50)
    print("DEMO: Hybrid Memory Manager")
    print("="*50)
    
    manager = HybridMemoryManager()
    print(manager)
    
    # Test con tensor
    x = torch.randn(1000, 1000)
    x_pesado = manager.to_device_pesado(x)
    x_ligero = manager.to_device_ligero(x)
    
    print(f"\nTensor original: {x.device}")
    print(f"En device pesado: {x_pesado.device}")
    print(f"En device ligero: {x_ligero.device}")
