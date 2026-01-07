# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v3.0 - Entrenador Avanzado
=====================================

Sistema de entrenamiento optimizado para la arquitectura
Trinity Fractal Recursivo Profundo.

Características:
- Entrenamiento con métricas detalladas
- Guardado de checkpoints
- Visualización de progreso
- Soporte para múltiples datasets

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Dict, Tuple, List
import time
from pathlib import Path
import json

from llarri_o1_fractal_profundo import (
    LlarriO1_FractalProfundo, 
    LlarriFractalConfig,
    crear_modelo_fractal
)


class EntrenadorFractal:
    """
    Entrenador optimizado para LLARRI-O1 v3.0 Fractal Profundo
    
    Características:
    - AdamW con weight decay
    - Cosine Annealing con warm restarts
    - Gradient clipping
    - Early stopping
    - Logging detallado
    """
    
    def __init__(
        self,
        modelo: LlarriO1_FractalProfundo,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5
    ):
        self.modelo = modelo
        self.device = modelo.device
        self.warmup_epochs = warmup_epochs
        
        # Optimizador AdamW
        self.optimizer = optim.AdamW(
            modelo.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler con warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=lr * 0.01
        )
        
        # Loss con label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Historial
        self.historial = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        
        # Mejor modelo
        self.mejor_acc = 0
        self.mejor_epoca = 0
        self.epochs_sin_mejora = 0
    
    def entrenar_epoca(self, dataloader: DataLoader, epoca: int) -> Tuple[float, float]:
        """Entrena una época"""
        self.modelo.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Flatten para MNIST
            data = data.view(data.size(0), -1).to(self.device)
            target = target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.modelo(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluar(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evalúa el modelo"""
        self.modelo.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in dataloader:
            data = data.view(data.size(0), -1).to(self.device)
            target = target.to(self.device)
            
            output = self.modelo(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def entrenar(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        guardar_mejor: bool = True,
        ruta_guardado: str = "checkpoints",
        early_stopping: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Ciclo de entrenamiento completo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número máximo de épocas
            guardar_mejor: Si guardar el mejor modelo
            ruta_guardado: Carpeta de guardado
            early_stopping: Épocas sin mejora para parar (0 = desactivado)
            verbose: Mostrar progreso
            
        Returns:
            historial: Dict con métricas
        """
        Path(ruta_guardado).mkdir(parents=True, exist_ok=True)
        
        if verbose:
            self._print_header(epochs)
        
        for epoch in range(epochs):
            inicio = time.time()
            
            # Entrenar
            train_loss, train_acc = self.entrenar_epoca(train_loader, epoch)
            
            # Evaluar
            val_loss, val_acc = self.evaluar(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Tiempo
            tiempo = time.time() - inicio
            
            # Guardar métricas
            self.historial['train_loss'].append(train_loss)
            self.historial['train_acc'].append(train_acc)
            self.historial['val_loss'].append(val_loss)
            self.historial['val_acc'].append(val_acc)
            self.historial['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.historial['epoch_time'].append(tiempo)
            
            # ¿Mejor modelo?
            es_mejor = val_acc > self.mejor_acc
            if es_mejor:
                self.mejor_acc = val_acc
                self.mejor_epoca = epoch
                self.epochs_sin_mejora = 0
                
                if guardar_mejor:
                    self._guardar_modelo(ruta_guardado, epoch, val_acc)
            else:
                self.epochs_sin_mejora += 1
            
            # Log
            if verbose:
                self._print_epoch(epoch, epochs, train_loss, train_acc, 
                                val_loss, val_acc, tiempo, es_mejor)
            
            # Early stopping
            if early_stopping > 0 and self.epochs_sin_mejora >= early_stopping:
                if verbose:
                    print(f"\n  ⚠️  Early stopping: {early_stopping} épocas sin mejora")
                break
        
        if verbose:
            self._print_footer()
        
        # Guardar historial
        self._guardar_historial(ruta_guardado)
        
        return self.historial
    
    def _guardar_modelo(self, ruta: str, epoch: int, val_acc: float):
        """Guarda el mejor modelo"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.modelo.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.modelo.config,
            'historial': self.historial,
            'compression_stats': self.modelo.get_compression_stats()
        }, f"{ruta}/mejor_modelo_fractal.pt")
    
    def _guardar_historial(self, ruta: str):
        """Guarda el historial en JSON"""
        with open(f"{ruta}/historial_entrenamiento.json", 'w') as f:
            json.dump(self.historial, f, indent=2)
    
    def _print_header(self, epochs: int):
        """Imprime header del entrenamiento"""
        stats = self.modelo.get_compression_stats()
        
        print(f"\n{'='*80}")
        print(f"  ENTRENAMIENTO LLARRI-O1 v3.0 - TRINITY FRACTAL RECURSIVO PROFUNDO")
        print(f"{'='*80}")
        print(f"  Épocas:              {epochs}")
        print(f"  Dispositivo:         {self.device}")
        print(f"  Parámetros:          {stats['parametros_reales']:,}")
        print(f"  Compresión:          {stats['compresion_porcentaje']:.1f}%")
        print(f"  Profundidad fractal: {stats['profundidad_fractal']}")
        print(f"{'='*80}")
        print(f"\n  {'Época':^8} │ {'Train Loss':^11} │ {'Train Acc':^10} │ "
              f"{'Val Loss':^10} │ {'Val Acc':^10} │ {'Tiempo':^7} │")
        print(f"  {'─'*8}─┼─{'─'*11}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*7}─┤")
    
    def _print_epoch(self, epoch, total, train_loss, train_acc, val_loss, val_acc, tiempo, mejor):
        """Imprime información de época"""
        marca = " ★" if mejor else "  "
        print(f"  {epoch+1:3d}/{total:3d}   │ {train_loss:11.4f} │ {train_acc:9.2f}% │ "
              f"{val_loss:10.4f} │ {val_acc:9.2f}% │ {tiempo:6.1f}s │{marca}")
    
    def _print_footer(self):
        """Imprime footer del entrenamiento"""
        print(f"  {'─'*8}─┴─{'─'*11}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*7}─┘")
        print(f"\n{'='*80}")
        print(f"  ENTRENAMIENTO COMPLETADO")
        print(f"  Mejor accuracy: {self.mejor_acc:.2f}% (época {self.mejor_epoca + 1})")
        print(f"{'='*80}\n")


def cargar_mnist(batch_size: int = 128, data_dir: str = "./data") -> Tuple[DataLoader, DataLoader]:
    """
    Carga el dataset MNIST.
    
    Args:
        batch_size: Tamaño del batch
        data_dir: Directorio de datos
        
    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    
    val_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def entrenar_modelo_completo(
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    profundidad: int = -1
) -> Tuple[LlarriO1_FractalProfundo, Dict]:
    """
    Función principal para entrenar el modelo completo.
    
    Args:
        hidden_dim: Dimensión oculta
        epochs: Número de épocas
        batch_size: Tamaño de batch
        lr: Learning rate
        profundidad: Profundidad fractal (-1 = auto)
        
    Returns:
        modelo, historial
    """
    print("\n" + "="*80)
    print("  PREPARANDO ENTRENAMIENTO")
    print("="*80)
    
    # Cargar datos
    print("\n  Cargando MNIST...")
    train_loader, val_loader = cargar_mnist(batch_size)
    print(f"  ✓ Train: {len(train_loader.dataset):,} imágenes")
    print(f"  ✓ Val:   {len(val_loader.dataset):,} imágenes")
    
    # Crear modelo
    print("\n  Creando modelo...")
    modelo = crear_modelo_fractal(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        profundidad=profundidad
    )
    
    # Crear entrenador
    entrenador = EntrenadorFractal(modelo, lr=lr)
    
    # Entrenar
    historial = entrenador.entrenar(
        train_loader,
        val_loader,
        epochs=epochs,
        guardar_mejor=True,
        early_stopping=15
    )
    
    return modelo, historial


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Entrenar modelo
    modelo, historial = entrenar_modelo_completo(
        hidden_dim=256,
        epochs=30,
        batch_size=128,
        lr=1e-3
    )
    
    # Mostrar estadísticas finales
    stats = modelo.get_compression_stats()
    print("\n" + "="*80)
    print("  ESTADÍSTICAS FINALES")
    print("="*80)
    print(f"\n  Mejor accuracy:    {max(historial['val_acc']):.2f}%")
    print(f"  Parámetros:        {stats['parametros_reales']:,}")
    print(f"  Compresión:        {stats['compresion_porcentaje']:.1f}%")
    print(f"  Factor reducción:  {stats['factor_reduccion']:.1f}x")
    print(f"  Profundidad:       {stats['profundidad_fractal']} niveles")
