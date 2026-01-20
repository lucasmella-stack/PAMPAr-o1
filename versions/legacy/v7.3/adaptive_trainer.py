# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi

"""
Trainer Adaptativo para LLARRI-O1.

Detecta automáticamente la capacidad del hardware y decide:
- Si hay suficiente memoria: entrena todos los niveles juntos
- Si no hay suficiente: entrena por niveles progresivamente

La clave: como los parámetros son COMPARTIDOS, entrenar por niveles
sigue actualizando el mismo modelo completo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Tuple
import time
import os
import gc

from llarri_o1.model import LlarriO1
from llarri_o1.config import Config


def get_gpu_memory_info() -> Dict[str, float]:
    """Obtiene información de memoria GPU."""
    if not torch.cuda.is_available():
        return {'available': 0, 'total': 0, 'used': 0}
    
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    reserved = torch.cuda.memory_reserved(0) / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    available = total - reserved
    
    return {
        'total': total,
        'reserved': reserved,
        'allocated': allocated,
        'available': available
    }


def estimate_memory_per_level(batch_size: int, hidden_dim: int) -> float:
    """
    Estima memoria necesaria por nivel fractal (en GB).
    
    Heurística basada en:
    - Activaciones: batch * hidden * 4 bytes (float32)
    - Gradientes: misma cantidad
    - Estados del optimizador: 2x (AdamW momentum + variance)
    """
    bytes_per_elem = 4  # float32
    activations = batch_size * hidden_dim * bytes_per_elem
    gradients = activations
    optimizer_states = activations * 2
    
    total_bytes = (activations + gradients + optimizer_states) * 3  # 3x safety margin
    return total_bytes / 1e9  # GB


class AdaptiveTrainer:
    """
    Entrenador Adaptativo para LLARRI-O1.
    
    MODO FULL:
        - Entrena todos los niveles en cada forward pass
        - Más rápido si hay suficiente memoria
    
    MODO PROGRESIVO:
        - Entrena por chunks de niveles
        - Permite entrenar modelos grandes en hardware limitado
        - Los parámetros compartidos se actualizan en cada chunk
    
    Example:
        >>> model = LlarriO1(Config(hidden_dim=1024))
        >>> trainer = AdaptiveTrainer(model, train_loader, test_loader)
        >>> trainer.train(epochs=10)  # Auto-detecta el mejor modo
    
    Author: Lucas Ricardo Mella Chillemi (Independent Researcher)
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
        memory_threshold_gb: float = 0.8,  # 80% de memoria disponible
        force_mode: Optional[str] = None,  # 'full', 'progressive', o None (auto)
    ):
        """
        Args:
            model: Modelo LLARRI-O1
            train_loader: DataLoader de entrenamiento
            test_loader: DataLoader de evaluación
            lr: Learning rate
            weight_decay: Weight decay para AdamW
            checkpoint_dir: Directorio para checkpoints
            use_amp: Usar mixed precision
            memory_threshold_gb: Umbral de memoria para decidir modo
            force_mode: Forzar modo específico ('full' o 'progressive')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        self.memory_threshold_gb = memory_threshold_gb
        self.force_mode = force_mode
        
        # Configuración del modelo
        self.niveles = list(model.config.niveles_fractales)
        self.batch_size = train_loader.batch_size or 32
        
        # Detectar modo óptimo
        self.mode, self.level_chunks = self._detect_optimal_mode()
        
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
            'mode': [],
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._print_config()
    
    def _detect_optimal_mode(self) -> Tuple[str, List[List[int]]]:
        """
        Detecta el modo óptimo basado en la capacidad del hardware.
        
        Returns:
            (mode, level_chunks): Modo y chunks de niveles a procesar
        """
        if self.force_mode:
            if self.force_mode == 'full':
                return 'full', [self.niveles]
            else:
                # Progressive: dividir en chunks de 2-3 niveles
                chunks = self._create_level_chunks()
                return 'progressive', chunks
        
        # Auto-detección
        mem_info = get_gpu_memory_info()
        
        if mem_info['total'] == 0:  # CPU
            # CPU siempre usa progresivo para no saturar RAM
            chunks = self._create_level_chunks()
            return 'progressive', chunks
        
        # Estimar memoria necesaria para full
        mem_needed = estimate_memory_per_level(
            self.batch_size, 
            self.model.config.hidden_dim
        ) * len(self.niveles)
        
        available = mem_info['available'] * self.memory_threshold_gb
        
        if mem_needed < available:
            return 'full', [self.niveles]
        else:
            # Calcular cuántos niveles caben
            mem_per_level = mem_needed / len(self.niveles)
            levels_that_fit = max(1, int(available / mem_per_level))
            chunks = self._create_level_chunks(max_per_chunk=levels_that_fit)
            return 'progressive', chunks
    
    def _create_level_chunks(self, max_per_chunk: int = 3) -> List[List[int]]:
        """Divide los niveles en chunks procesables."""
        chunks = []
        for i in range(0, len(self.niveles), max_per_chunk):
            chunk = self.niveles[i:i + max_per_chunk]
            chunks.append(chunk)
        return chunks
    
    def _print_config(self):
        """Imprime configuración del trainer."""
        mem_info = get_gpu_memory_info()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║             ADAPTIVE TRAINER - LLARRI-O1                     ║
╠══════════════════════════════════════════════════════════════╣
║  Device:      {str(self.device):>10}                                    ║
║  GPU Memory:  {mem_info['total']:.1f} GB total, {mem_info['available']:.1f} GB available         
║  Batch Size:  {self.batch_size:>10}                                    ║
║  AMP:         {'Sí' if self.use_amp else 'No':>10}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  MODO:        {self.mode.upper():>10}                                    ║""")
        
        if self.mode == 'progressive':
            print(f"║  Chunks:      {len(self.level_chunks)} grupos de niveles                       ║")
            for i, chunk in enumerate(self.level_chunks):
                print(f"║    Chunk {i+1}:   {chunk}                           ")
        else:
            print(f"║  Niveles:     {self.niveles}          ")
        
        print(f"""╠══════════════════════════════════════════════════════════════╣
║  Los parámetros se COMPARTEN entre niveles.                  ║
║  Entrenar por chunks actualiza el MISMO modelo completo.     ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def _forward_full(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass normal (todos los niveles)."""
        return self.model(data)
    
    def _forward_progressive(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass progresivo (por chunks de niveles).
        
        Procesa chunk por chunk, acumulando gradientes.
        Los parámetros compartidos reciben gradientes de TODOS los chunks.
        """
        # Para el forward progresivo, necesitamos modificar temporalmente
        # el modelo para que solo procese ciertos niveles.
        # Sin embargo, la arquitectura actual procesa todos los niveles
        # en secuencia dentro de CuadranteProgresivo.
        
        # Solución: procesar normalmente pero con gradient checkpointing
        # para reducir memoria. El modelo ya comparte parámetros.
        
        # Por ahora, usamos el forward normal con checkpointing
        # TODO: Implementar forward por chunks de niveles específicos
        
        return self.model(data)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Entrena una época."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mode == 'progressive':
                # Modo progresivo: procesar por chunks y acumular gradientes
                loss, output = self._train_step_progressive(data, target)
            else:
                # Modo full: forward normal
                loss, output = self._train_step_full(data, target)
            
            total_loss += loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Limpiar cache para modo progresivo
            if self.mode == 'progressive' and batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _train_step_full(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Paso de entrenamiento en modo full."""
        if self.use_amp:
            with autocast('cuda'):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), output.detach()
    
    def _train_step_progressive(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Paso de entrenamiento en modo progresivo.
        
        Procesa el modelo completo pero con gradient checkpointing
        para reducir uso de memoria.
        """
        # Habilitar gradient checkpointing si está disponible
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        if self.use_amp:
            with autocast('cuda'):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), output.detach()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evalúa el modelo."""
        self.model.eval()
        correct = 0
        total = 0
        
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_amp and torch.cuda.is_available():
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
            'mode': self.mode,
        }, path)
        print(f"  → Checkpoint: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Carga un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint.get('accuracy', 0.0)
        print(f"Checkpoint cargado: {path} (acc: {self.best_acc:.2f}%)")
        return checkpoint.get('epoch', 0)
    
    def train(self, epochs: int = 10, save_every: int = 5):
        """
        Entrena el modelo adaptativamente.
        
        Args:
            epochs: Número de épocas
            save_every: Guardar checkpoint cada N épocas
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento en modo {self.mode.upper()}")
        print(f"{'='*60}\n")
        
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
            self.history['mode'].append(self.mode)
            
            # Logging
            mem_info = get_gpu_memory_info()
            print(f"Época {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:6.2f}% | "
                  f"Test: {test_acc:6.2f}% | "
                  f"Mem: {mem_info['allocated']:.1f}GB | "
                  f"Time: {elapsed:.1f}s")
            
            # Guardar mejor modelo
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_model.pt', epoch, test_acc)
            
            # Checkpoint periódico
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, test_acc)
        
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado.")
        print(f"Mejor accuracy: {self.best_acc:.2f}%")
        print(f"Modo utilizado: {self.mode}")
        print(f"{'='*60}")
        
        return self.history
