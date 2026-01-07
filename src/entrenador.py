# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v2.0 - Entrenador
===========================

Sistema de entrenamiento optimizado para la arquitectura
Trinity Fractal Cuadrantes.

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Dict, Tuple
import time
from pathlib import Path

from llarri_o1_v2 import LlarriO1_v2, LlarriConfig, crear_modelo


class Entrenador:
    """Entrenador optimizado para LLARRI-O1 v2.0"""
    
    def __init__(
        self,
        modelo: LlarriO1_v2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.modelo = modelo
        self.device = modelo.device
        
        # Optimizador AdamW
        self.optimizer = optim.AdamW(
            modelo.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Métricas
        self.historial = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def entrenar_epoca(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Entrena una época"""
        self.modelo.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(self.device)
            target = target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.modelo(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)
            
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
        epochs: int = 20,
        guardar_mejor: bool = True,
        ruta_guardado: str = "checkpoints"
    ) -> Dict:
        """
        Ciclo de entrenamiento completo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            guardar_mejor: Si guardar el mejor modelo
            ruta_guardado: Carpeta donde guardar
            
        Returns:
            historial: Dict con métricas de entrenamiento
        """
        Path(ruta_guardado).mkdir(parents=True, exist_ok=True)
        mejor_acc = 0
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO LLARRI-O1 v2.0")
        print(f"{'='*60}")
        print(f"Épocas: {epochs}")
        print(f"Dispositivo: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            inicio = time.time()
            
            # Entrenar
            train_loss, train_acc = self.entrenar_epoca(train_loader)
            
            # Evaluar
            val_loss, val_acc = self.evaluar(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Guardar métricas
            self.historial['train_loss'].append(train_loss)
            self.historial['train_acc'].append(train_acc)
            self.historial['val_loss'].append(val_loss)
            self.historial['val_acc'].append(val_acc)
            
            # Guardar mejor modelo
            if guardar_mejor and val_acc > mejor_acc:
                mejor_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.modelo.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.modelo.config
                }, f"{ruta_guardado}/mejor_modelo.pt")
                print(f"  ✓ Nuevo mejor modelo guardado (acc: {val_acc:.2f}%)")
            
            tiempo = time.time() - inicio
            print(f"Época {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"Tiempo: {tiempo:.1f}s")
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"Mejor accuracy: {mejor_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.historial


def cargar_mnist(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Carga el dataset MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENTRENAMIENTO LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes")
    print("="*70)
    
    # Crear modelo
    modelo = crear_modelo(
        input_dim=784,
        hidden_dim=256,
        output_dim=10
    )
    
    # Cargar datos
    print("\nCargando MNIST...")
    train_loader, test_loader = cargar_mnist(batch_size=128)
    print(f"Train: {len(train_loader.dataset)} muestras")
    print(f"Test: {len(test_loader.dataset)} muestras")
    
    # Crear entrenador
    entrenador = Entrenador(modelo, lr=1e-3)
    
    # Entrenar
    historial = entrenador.entrenar(
        train_loader,
        test_loader,
        epochs=20,
        guardar_mejor=True,
        ruta_guardado="checkpoints"
    )
    
    # Evaluación final
    print("\nEvaluación final en test set:")
    test_loss, test_acc = entrenador.evaluar(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Estadísticas de compresión
    stats = modelo.get_compression_stats()
    print(f"\n{'='*60}")
    print("RESUMEN DE COMPRESIÓN")
    print(f"{'='*60}")
    print(f"Parámetros:   {stats['params_reales']:,}")
    print(f"Compresión:   {stats['compresion_porcentaje']:.1f}%")
    print(f"Factor:       {stats['factor_reduccion']:.1f}x")
