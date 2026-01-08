# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
# Coordinator: Alvaro (alvaro@segundacabeza.com)
"""
Script de entrenamiento para LLARRI v5 - Híbrido Neural + Matemático.

Las cajas 1-6 (neuronales) se entrenan.
Las cajas 7-9 (matemáticas) NO se entrenan - son reglas fijas.

Esto significa que el modelo tiene MENOS parámetros pero
GARANTÍAS matemáticas de no colapsar.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, List
from datasets import load_dataset

# Agregar path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llarri_o1.models.language_model_v5 import LLARRIv5, LLARRIv5Config
from llarri_o1.utils.device import get_device, print_device_info


class TinyStoriesDataset(Dataset):
    """Dataset de TinyStories para entrenamiento."""
    
    def __init__(
        self, 
        split: str = "train",
        max_length: int = 128,
        max_samples: Optional[int] = None
    ):
        self.max_length = max_length
        
        print(f"Cargando TinyStories ({split})...")
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
        self.texts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            self.texts.append(item['text'])
            if (i + 1) % 10000 == 0:
                print(f"  Cargados {i+1} textos...")
        
        print(f"  Total: {len(self.texts)} textos")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convertir a bytes
        tokens = list(text.encode('utf-8', errors='ignore'))
        
        # Truncar o pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


class LLARRIv5Trainer:
    """Trainer para LLARRI v5."""
    
    def __init__(
        self,
        model: LLARRIv5,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        grad_accum_steps: int = 2,
        use_amp: bool = True
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Optimizer y Scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * 10  # Asumir 10 epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        print("\n" + "=" * 60)
        print("LLARRI v5 - TRAINER HÍBRIDO")
        print("=" * 60)
        print(f"  Parámetros ENTRENABLES: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Cajas 1-6: Neuronales (se entrenan)")
        print(f"  Cajas 7-9: Matemáticas (reglas fijas)")
        print(f"  Batch Size: {batch_size} (×{grad_accum_steps} = {batch_size * grad_accum_steps})")
        print(f"  AMP: {'Sí' if self.use_amp else 'No'}")
        print(f"  Device: {self.device}")
        print("=" * 60)
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, labels)
                loss = output['loss'] / self.grad_accum_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1:5}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Valida el modelo."""
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, labels)
            
            total_loss += output['loss'].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "checkpoints",
        save_every: int = 2
    ):
        """Entrena el modelo."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        print(f"\n🚀 INICIANDO ENTRENAMIENTO - {num_epochs} épocas")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\n📘 Época {epoch}/{num_epochs}")
            print("-" * 40)
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            
            print(f"\n  ✓ Train Loss: {train_loss:.4f}")
            print(f"  ✓ Val Loss:   {val_loss:.4f}")
            print(f"  ✓ Perplexity: {perplexity:.2f}")
            print(f"  ⏱️  Tiempo: {epoch_time:.1f}s")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.model.config,
                }, f"{save_dir}/llarri_v5_best.pt")
                print(f"  💾 Guardado: {save_dir}/llarri_v5_best.pt")
            
            # Guardar checkpoint periódico
            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                }, f"{save_dir}/llarri_v5_epoch_{epoch}.pt")
                print(f"  💾 Guardado: {save_dir}/llarri_v5_epoch_{epoch}.pt")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Tiempo total: {total_time/60:.1f} minutos")
        print(f"   Mejor loss: {best_val_loss:.4f}")
        print("=" * 60)
        
        # Generar texto de muestra
        print("\n📝 Generando texto de muestra...")
        self.model.eval()
        sample = self.model.generate("Once upon a time", max_new_tokens=100)
        print(f"   {sample}")


def main():
    parser = argparse.ArgumentParser(description="Entrenar LLARRI v5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠 LLARRI v5 - HÍBRIDO NEURAL + MATEMÁTICO                    ║
║                                                                  ║
║   Cajas 1-6: Redes Neuronales (aprenden)                        ║
║   Cajas 7-9: Matemáticas Blindadas (garantizan)                 ║
║                                                                  ║
║   Author: Lucas Ricardo Mella Chillemi                           ║
║   Organization: Segunda Cabeza                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    print(f"📊 Configuración:")
    print(f"   embed_dim: {args.embed_dim}")
    print(f"   niveles: [2, 4, 8, 16]")
    print(f"   max_length: {args.seq_len + 1}")
    
    # Crear modelo
    print("\n📦 Creando modelo...")
    config = LLARRIv5Config(
        embed_dim=args.embed_dim,
        niveles=[2, 4, 8, 16],
        max_length=args.seq_len + 1
    )
    model = LLARRIv5(config)
    
    # Dataset
    print("\n📚 Cargando dataset TinyStories...")
    train_dataset = TinyStoriesDataset(
        split="train",
        max_length=args.seq_len + 1,
        max_samples=args.max_samples
    )
    val_dataset = TinyStoriesDataset(
        split="validation",
        max_length=args.seq_len + 1,
        max_samples=5000
    )
    
    # Device
    device = get_device()
    print_device_info()
    model = model.to(device)
    
    # Trainer
    trainer = LLARRIv5Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_accum_steps=args.grad_accum
    )
    
    # Entrenar
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
