# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6 - Training Script

Entrena el modelo de 27 cajas con reflexión y early exit.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v6 import LLARRILanguageModelV6, LLARRIConfigV6


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠 LLARRI v6 - 27 CAJAS CON REFLEXIÓN Y EARLY EXIT            ║
║                                                                  ║
║   Cajas 1-3:   PUERTA (gates rápidos)                           ║
║   Cajas 4-12:  NEURAL 1 (aprende)                               ║
║   Cajas 13-15: REFLEXIÓN 1 (¿early exit?)                       ║
║   Cajas 16-24: NEURAL 2 (profundiza)                            ║
║   Cajas 25-26: REFLEXIÓN 2 (corrección final)                   ║
║   Caja 27:     OUTPUT                                           ║
║                                                                  ║
║   Escalas: 2→256 (8 niveles fractales)                          ║
║                                                                  ║
║   Author: Lucas Ricardo Mella Chillemi                           ║
║   Organization: Segunda Cabeza                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_device_info():
    """Imprime información del dispositivo."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"\n{'='*50}")
        print(f"Dispositivo: cuda")
        print(f"GPU: {props.name}")
        print(f"VRAM Total: {props.total_memory / 1024**3:.1f} GB")
        print(f"VRAM Usado: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"{'='*50}\n")
    else:
        print("\n⚠️  CUDA no disponible, usando CPU\n")


class TinyStoriesDataset(Dataset):
    """Dataset para TinyStories con byte-level tokenization."""
    
    def __init__(self, texts, seq_len=128):
        self.texts = texts
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Byte-level encoding
        bytes_data = list(text.encode('utf-8'))[:self.seq_len + 1]
        
        # Pad si es necesario
        if len(bytes_data) < self.seq_len + 1:
            bytes_data = bytes_data + [0] * (self.seq_len + 1 - len(bytes_data))
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


def load_tinystories(max_samples=50000):
    """Carga TinyStories desde HuggingFace."""
    try:
        from datasets import load_dataset
        
        print("Cargando TinyStories (train)...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            texts.append(item['text'])
            if (i + 1) % 10000 == 0:
                print(f"  Cargados {i+1} textos...")
        
        print(f"  Total: {len(texts)} textos")
        
        # Validación
        print("Cargando TinyStories (validation)...")
        val_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        val_texts = []
        for i, item in enumerate(val_dataset):
            if i >= max_samples // 10:
                break
            val_texts.append(item['text'])
        print(f"  Total: {len(val_texts)} textos")
        
        return texts, val_texts
        
    except Exception as e:
        print(f"Error cargando TinyStories: {e}")
        print("Generando datos sintéticos...")
        
        # Datos sintéticos de respaldo
        templates = [
            "Once upon a time, there was a little {}.",
            "The {} was very happy today.",
            "A small {} lived in a big house.",
            "One day, the {} went to the park.",
        ]
        words = ["cat", "dog", "bird", "mouse", "rabbit", "girl", "boy"]
        
        texts = [t.format(w) for t in templates for w in words] * (max_samples // 28 + 1)
        texts = texts[:max_samples]
        val_texts = texts[:max_samples // 10]
        
        return texts, val_texts


class TrainerV6:
    """Trainer para LLARRI v6."""
    
    def __init__(
        self,
        model: LLARRILanguageModelV6,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        grad_accum_steps: int = 2
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,
            eta_min=lr * 0.1
        )
        
        # AMP Scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Best loss tracking
        self.best_val_loss = float('inf')
        
        # Stats
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"""
{'='*60}
LLARRI v6 - TRAINER
{'='*60}
  Parámetros TOTALES: {self.total_params:,}
  Parámetros ENTRENABLES: {self.trainable_params:,}
  Batch Size: {train_loader.batch_size} (×{grad_accum_steps} = {train_loader.batch_size * grad_accum_steps})
  AMP: {'Sí' if self.use_amp else 'No'}
  Device: {device}
  Escalas: {model.config.escalas}
{'='*60}
""")
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        n_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward con AMP
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, labels)
                loss = loss / self.grad_accum_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.grad_accum_steps == 0:
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
            
            total_loss += loss.item() * self.grad_accum_steps
            
            # Log
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1:5d}/{n_batches} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evalúa en validation set."""
        self.model.eval()
        self.model.reset_early_exit_stats()
        
        total_loss = 0
        n_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, labels)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        
        # Early exit stats
        ee_rate = self.model.early_exit_count / max(1, self.model.total_forward_count)
        print(f"  📊 Early Exit Rate: {ee_rate:.2%}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Guarda checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': {
                'vocab_size': self.model.config.vocab_size,
                'embed_dim': self.model.config.embed_dim,
                'n_heads': self.model.config.n_heads,
                'max_length': self.model.config.max_length,
                'dropout': self.model.config.dropout,
                'escalas': self.model.config.escalas,
                'threshold_early_exit': self.model.config.threshold_early_exit,
                'threshold_correccion': self.model.config.threshold_correccion,
            }
        }
        
        # Guardar por época
        path = self.checkpoint_dir / f"llarri_v6_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.checkpoint_dir / "llarri_v6_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  💾 Mejor modelo guardado: {best_path}")
    
    def train(self, n_epochs: int):
        """Loop principal de entrenamiento."""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO - {n_epochs} épocas")
        print("=" * 60)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\n📘 Época {epoch}/{n_epochs}")
            print("-" * 40)
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            # Evaluate
            val_loss = self.evaluate()
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save
            if epoch % 2 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log
            print(f"\n  📊 Resumen Época {epoch}:")
            print(f"     Train Loss: {train_loss:.4f}")
            print(f"     Val Loss:   {val_loss:.4f} {'⭐ BEST!' if is_best else ''}")
            print(f"     Perplexity: {perplexity:.2f}")
            print(f"     Tiempo:     {train_time/60:.1f} min")
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Mejor Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train LLARRI v6")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()
    
    print_banner()
    
    print(f"📊 Configuración:")
    print(f"   embed_dim: {args.embed_dim}")
    print(f"   n_heads: {args.n_heads}")
    print(f"   seq_len: {args.seq_len}")
    print(f"   escalas: (2, 4, 8, 16, 32, 64, 128, 256)")
    
    # Crear modelo
    print("\n📦 Creando modelo...")
    config = LLARRIConfigV6(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        max_length=args.seq_len + 1,
        dropout=0.1
    )
    model = LLARRILanguageModelV6(config)
    
    # Cargar datos
    print("\n📚 Cargando dataset TinyStories...")
    train_texts, val_texts = load_tinystories(args.max_samples)
    
    train_dataset = TinyStoriesDataset(train_texts, args.seq_len)
    val_dataset = TinyStoriesDataset(val_texts, args.seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_device_info()
    
    # Trainer
    trainer = TrainerV6(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=device,
        grad_accum_steps=args.grad_accum
    )
    
    # Train
    trainer.train(args.epochs)
    
    # Test generación final
    print("\n📝 Test de generación final:")
    model.eval()
    prompts = ["Once upon a time", "The little girl", "A magical forest"]
    for prompt in prompts:
        generated = model.generate_text(prompt, max_length=80, temperature=0.8)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {generated[:150]}...")


if __name__ == "__main__":
    main()
