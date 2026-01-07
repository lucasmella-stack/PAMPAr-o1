"""
LLARRI-O1 - Entrenador Principal
=================================

Entrenador optimizado con soporte híbrido CPU/GPU.

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
import time
import os
from pathlib import Path


@dataclass
class EntrenadorConfig:
    """Configuración del entrenador"""
    
    # Épocas
    epochs: int = 20
    
    # Optimización
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Scheduler
    usar_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 7
    min_delta: float = 0.001
    
    # Checkpoints
    guardar_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    guardar_mejor: bool = True
    
    # Logging
    log_interval: int = 100
    verbose: bool = True
    
    # Mixed precision
    usar_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0


class Entrenador:
    """
    Entrenador para LLARRI-O1.
    
    Características:
    - Soporte híbrido CPU/GPU
    - Mixed precision (AMP)
    - Early stopping
    - Checkpoints automáticos
    - Logging detallado
    
    Uso:
        entrenador = Entrenador(modelo, config)
        historia = entrenador.entrenar(train_loader, test_loader)
    """
    
    def __init__(
        self, 
        modelo: nn.Module,
        config: Optional[EntrenadorConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.modelo = modelo
        self.config = config or EntrenadorConfig()
        
        # Device
        if device is not None:
            self.device = device
        elif hasattr(modelo, 'rm') and hasattr(modelo.rm, 'device_pesado'):
            self.device = modelo.rm.device_pesado
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mover modelo al device
        self.modelo.to(self.device)
        
        # Optimizador
        self.optimizer = optim.AdamW(
            self.modelo.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        if self.config.usar_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=self.config.verbose
            )
        else:
            self.scheduler = None
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda') if self.config.usar_amp and self.device.type == 'cuda' else None
        
        # Historia
        self.historia: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        
        # Early stopping
        self.mejor_val_acc = 0.0
        self.epochs_sin_mejora = 0
        
        # Checkpoints
        if self.config.guardar_checkpoints:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _preparar_batch(self, batch):
        """Prepara un batch para el entrenamiento"""
        x, y = batch
        
        # Aplanar si es necesario
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return x.to(self.device), y.to(self.device)
    
    def _entrenar_epoch(self, train_loader: DataLoader) -> tuple:
        """Entrena una época"""
        self.modelo.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x, y = self._preparar_batch(batch)
            
            self.optimizer.zero_grad()
            
            # Forward con AMP si está habilitado
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.modelo(x)
                    loss = self.criterion(outputs, y)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modelo.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.modelo(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modelo.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Log intermedio
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - "
                      f"Acc: {100.*correct/total:.2f}%", end='\r')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _evaluar(self, test_loader: DataLoader) -> tuple:
        """Evalúa el modelo"""
        self.modelo.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in test_loader:
            x, y = self._preparar_batch(batch)
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.modelo(x)
                    loss = self.criterion(outputs, y)
            else:
                outputs = self.modelo(x)
                loss = self.criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _guardar_checkpoint(self, epoch: int, val_acc: float, es_mejor: bool = False):
        """Guarda un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.modelo.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'historia': self.historia,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Guardar checkpoint regular
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        
        # Guardar mejor modelo
        if es_mejor and self.config.guardar_mejor:
            mejor_path = os.path.join(self.config.checkpoint_dir, "mejor_modelo.pt")
            torch.save(checkpoint, mejor_path)
            if self.config.verbose:
                print(f"    ✓ Nuevo mejor modelo guardado: {val_acc:.2f}%")
    
    def cargar_checkpoint(self, path: str):
        """Carga un checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.modelo.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'historia' in checkpoint:
            self.historia = checkpoint['historia']
        
        return checkpoint.get('epoch', 0), checkpoint.get('val_acc', 0)
    
    def entrenar(
        self, 
        train_loader: DataLoader, 
        test_loader: DataLoader,
        callback: Optional[Callable[[int, Dict], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            test_loader: DataLoader de validación
            callback: Función opcional llamada después de cada época
        
        Returns:
            Historia del entrenamiento
        """
        print(f"\n{'='*60}")
        print("INICIANDO ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Épocas: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"AMP: {'Sí' if self.scaler else 'No'}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Entrenar
            train_loss, train_acc = self._entrenar_epoch(train_loader)
            
            # Evaluar
            val_loss, val_acc = self._evaluar(test_loader)
            
            # Tiempo
            epoch_time = time.time() - epoch_start
            
            # Guardar historia
            self.historia['train_loss'].append(train_loss)
            self.historia['train_acc'].append(train_acc)
            self.historia['val_loss'].append(val_loss)
            self.historia['val_acc'].append(val_acc)
            self.historia['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.historia['epoch_time'].append(epoch_time)
            
            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_acc)
            
            # Check mejor modelo
            es_mejor = val_acc > self.mejor_val_acc
            if es_mejor:
                self.mejor_val_acc = val_acc
                self.epochs_sin_mejora = 0
            else:
                self.epochs_sin_mejora += 1
            
            # Guardar checkpoint
            if self.config.guardar_checkpoints:
                self._guardar_checkpoint(epoch, val_acc, es_mejor)
            
            # Log
            if self.config.verbose:
                print(f"Época {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
                print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
                print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                if es_mejor:
                    print(f"  ★ Nueva mejor accuracy: {val_acc:.2f}%")
                print()
            
            # Callback
            if callback is not None:
                callback(epoch, {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
            
            # Early stopping
            if self.config.early_stopping and self.epochs_sin_mejora >= self.config.patience:
                if self.config.verbose:
                    print(f"\n⚠ Early stopping: {self.config.patience} épocas sin mejora")
                break
        
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"Mejor accuracy: {self.mejor_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.historia


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def entrenar_rapido(
    modelo: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Entrena un modelo rápidamente con configuración por defecto.
    
    Útil para pruebas rápidas.
    """
    config = EntrenadorConfig(
        epochs=epochs,
        learning_rate=lr,
        verbose=verbose,
        guardar_checkpoints=False,
        early_stopping=False
    )
    
    entrenador = Entrenador(modelo, config)
    return entrenador.entrenar(train_loader, test_loader)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMO: Entrenador")
    print("="*50)
    
    # Modelo simple para demo
    class ModeloDemo(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    modelo = ModeloDemo()
    
    # Config
    config = EntrenadorConfig(
        epochs=2,
        verbose=True,
        guardar_checkpoints=False
    )
    
    entrenador = Entrenador(modelo, config)
    
    print(f"Entrenador creado")
    print(f"Device: {entrenador.device}")
    print(f"Config: {config}")
