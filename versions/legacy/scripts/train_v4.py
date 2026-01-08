# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
Script de entrenamiento para LLARRI v4 - 9 Cajas.

Arquitectura:
- Cajas 1-6: Procesamiento fractal (percepción)
- Cajas 7-9: Compositor (cognición/razonamiento)
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset

# Añadir path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llarri_o1.models.language_model_v4 import LLARRIv4, LLARRIv4Config


class TinyStoriesDataset(Dataset):
    """Dataset de TinyStories para entrenamiento byte-level."""
    
    def __init__(self, texts, seq_len=256):
        self.texts = texts
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convertir a bytes
        bytes_data = list(text.encode('utf-8', errors='ignore'))
        
        # Padding o truncate
        if len(bytes_data) < self.seq_len + 1:
            bytes_data = bytes_data + [0] * (self.seq_len + 1 - len(bytes_data))
        else:
            bytes_data = bytes_data[:self.seq_len + 1]
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


def load_tinystories(max_samples=None, split='train'):
    """Carga TinyStories desde HuggingFace."""
    print(f"Cargando TinyStories ({split})...")
    
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    
    texts = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        texts.append(item['text'])
        if (i + 1) % 10000 == 0:
            print(f"  Cargados {i+1} textos...")
    
    print(f"  Total: {len(texts)} textos")
    return texts


class LLARRIv4Trainer:
    """Trainer para LLARRI v4."""
    
    def __init__(
        self,
        model: LLARRIv4,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer con weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Scheduler cosine
        total_steps = len(train_loader) * 30  # estimado
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=lr * 0.01
        )
        
        # AMP
        self.use_amp = device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum = 2
        
        self.best_val_loss = float('inf')
        
        print("\n" + "=" * 60)
        print("LLARRI v4 - 9 CAJAS TRAINER")
        print("=" * 60)
        print(f"  Parámetros: {model.get_num_params():,}")
        print(f"  Batch Size: {train_loader.batch_size} (×{self.grad_accum} = {train_loader.batch_size * self.grad_accum})")
        print(f"  AMP: {'Sí' if self.use_amp else 'No'}")
        print(f"  Device: {device}")
        print("=" * 60)
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, labels=labels)
                loss = output['loss'] / self.grad_accum
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum
            num_batches += 1
            
            # Gradient accumulation step
            if (i + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Log
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {i+1:5d}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, labels=labels)
                total_loss += output['loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs: int, save_every: int = 2):
        """Entrenamiento completo."""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO - {epochs} épocas")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\n📘 Época {epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n  ✓ Train Loss: {train_loss:.4f}")
            print(f"  ✓ Val Loss:   {val_loss:.4f}")
            print(f"  ✓ Perplexity: {perplexity:.2f}")
            print(f"  ⏱️  Tiempo: {epoch_time:.1f}s")
            
            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('checkpoints/llarri_v4_best.pt', val_loss)
                print(f"  💾 Guardado: checkpoints/llarri_v4_best.pt")
            
            # Save periodic
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoints/llarri_v4_epoch_{epoch}.pt', val_loss)
                print(f"  💾 Guardado: checkpoints/llarri_v4_epoch_{epoch}.pt")
        
        total_time = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Tiempo total: {total_time:.1f} minutos")
        print(f"   Mejor loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        # Test generación
        print("\n📝 Generando texto de muestra...")
        self.model.eval()
        text = self.model.generate("Once upon a time", max_new_tokens=100, temperatura=0.8)
        print(f"   {text}")
    
    def save_checkpoint(self, path: str, val_loss: float):
        """Guarda checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Train LLARRI v4 - 9 Cajas')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠 LLARRI v4 - 9 CAJAS                                        ║
║                                                                  ║
║   Cajas 1-6: Procesamiento Fractal (percepción)                 ║
║   Cajas 7-9: Compositor (cognición/razonamiento)                ║
║                                                                  ║
║   Author: Lucas Ricardo Mella Chillemi                           ║
║   Organization: Segunda Cabeza                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Config
    config = LLARRIv4Config(
        embed_dim=args.embed_dim,
        niveles=[2, 4, 8, 16],
        max_length=args.seq_len + 1,
    )
    
    print(f"📊 Configuración:")
    print(f"   embed_dim: {args.embed_dim}")
    print(f"   niveles: [2, 4, 8, 16]")
    print(f"   max_length: {args.seq_len + 1}")
    
    # Model
    print("\n📦 Creando modelo...")
    model = LLARRIv4(config)
    
    # Data
    print("\n📚 Cargando dataset TinyStories...")
    train_texts = load_tinystories(max_samples=args.max_samples, split='train')
    val_texts = load_tinystories(max_samples=args.max_samples // 10, split='validation')
    
    train_dataset = TinyStoriesDataset(train_texts, seq_len=args.seq_len)
    val_dataset = TinyStoriesDataset(val_texts, seq_len=args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🖥️  Dispositivo: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Trainer
    trainer = LLARRIv4Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=device
    )
    
    # Train
    trainer.train(epochs=args.epochs, save_every=2)


if __name__ == "__main__":
    main()
