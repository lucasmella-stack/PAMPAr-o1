# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

"""
Trainer para LLARRI-O1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, Any
import time
import os

from llarri_o1.model import LlarriO1


class Trainer:
    """
    Entrenador para LLARRI-O1.
    
    Soporta:
        - Mixed precision (AMP)
        - Gradient accumulation
        - Checkpointing
        - Early stopping básico
    
    Example:
        >>> from llarri_o1 import LlarriO1, Config
        >>> from llarri_o1.training import Trainer
        >>> from llarri_o1.utils import get_mnist_loaders
        >>> 
        >>> model = LlarriO1(Config())
        >>> train_loader, test_loader = get_mnist_loaders(batch_size=32)
        >>> trainer = Trainer(model, train_loader, test_loader)
        >>> trainer.train(epochs=10)
    """
    
    def __init__(
        self,
        model: LlarriO1,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        accumulation_steps: int = 1,
    ):
        """
        Args:
            model: Modelo LLARRI-O1
            train_loader: DataLoader de entrenamiento
            test_loader: DataLoader de evaluación
            lr: Learning rate
            weight_decay: Weight decay para AdamW
            checkpoint_dir: Directorio para guardar checkpoints
            use_amp: Usar mixed precision
            accumulation_steps: Pasos de acumulación de gradientes
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        self.accumulation_steps = accumulation_steps
        
        # Optimizador y scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Loss y AMP
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_acc = 0.0
        self.history: Dict[str, list] = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
        }
        
        # Crear directorio de checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> tuple:
        """Entrena una época."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward con AMP
            if self.use_amp:
                with autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evalúa el modelo."""
        self.model.eval()
        correct = 0
        total = 0
        
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    output = self.model(data)
            else:
                output = self.model(data)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return 100.0 * correct / total
    
    def save_checkpoint(self, filename: str, epoch: int, accuracy: float):
        """Guarda un checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.model.config,
        }, path)
        print(f"Checkpoint guardado: {path}")
    
    def load_checkpoint(self, path: str):
        """Carga un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint.get('accuracy', 0.0)
        print(f"Checkpoint cargado: {path} (acc: {self.best_acc:.2f}%)")
        return checkpoint.get('epoch', 0)
    
    def train(self, epochs: int = 10, save_every: int = 5):
        """
        Entrena el modelo.
        
        Args:
            epochs: Número de épocas
            save_every: Guardar checkpoint cada N épocas
        """
        print(f"\nEntrenando en {self.device}")
        print(f"AMP: {'Sí' if self.use_amp else 'No'}")
        print(f"Acumulación: {self.accumulation_steps} pasos")
        print(f"{'='*50}\n")
        
        for epoch in range(1, epochs + 1):
            start = time.time()
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            
            # Evaluar
            test_acc = self.evaluate()
            
            elapsed = time.time() - start
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            # Logging
            print(f"Época {epoch}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.2f}% | "
                  f"Test: {test_acc:.2f}% | "
                  f"Time: {elapsed:.1f}s")
            
            # Guardar mejor modelo
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_model.pt', epoch, test_acc)
            
            # Checkpoint periódico
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, test_acc)
        
        print(f"\n{'='*50}")
        print(f"Entrenamiento completado. Mejor accuracy: {self.best_acc:.2f}%")
