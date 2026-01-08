# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7 - Entrenamiento de Arquitectura Cerebral
===================================================

Modelo con 6 módulos especializados inspirados en el cerebro:
- Tálamo: router con modulación continua (15%-100%)
- Módulos: Lenguaje, Lógica, Matemáticas, Patrones, Contexto, Creatividad
- Hipocampo: memoria con LSH
- Integrador: combina outputs respetando modulación

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v7 import (
    LLARRIv7Cerebral,
    LLARRIv7Mini,
    LLARRIv7Base,
    LLARRIv7Large,
)


class WikiTextDataset(Dataset):
    """Dataset para WikiText-103."""
    
    def __init__(self, data_path: str, seq_len: int = 128, vocab_size: int = 50257):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Cargar datos
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenización simple (byte-level)
        self.tokens = torch.tensor(
            [min(ord(c), vocab_size - 1) for c in text],
            dtype=torch.long
        )
        
        self.n_samples = (len(self.tokens) - 1) // seq_len
        print(f"   Tokens: {len(self.tokens):,}")
        print(f"   Samples: {self.n_samples:,}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


def get_dataloaders(data_dir: str, seq_len: int, batch_size: int, vocab_size: int):
    """Crear dataloaders para train y val."""
    train_path = Path(data_dir) / "wikitext-103-raw" / "wiki.train.raw"
    val_path = Path(data_dir) / "wikitext-103-raw" / "wiki.valid.raw"
    
    print("📚 Cargando datos...")
    print("   Train:")
    train_dataset = WikiTextDataset(str(train_path), seq_len, vocab_size)
    print("   Val:")
    val_dataset = WikiTextDataset(str(val_path), seq_len, vocab_size)
    
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


def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    """Entrenar una época."""
    model.train()
    total_loss = 0
    total_tokens = 0
    total_consenso = 0
    total_iteraciones = 0
    n_batches = 0
    
    start_time = time.time()
    log_interval = 100
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward con mixed precision
        with autocast():
            outputs = model(inputs, targets=targets)
            loss = outputs['loss']
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Stats
        batch_tokens = inputs.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_consenso += outputs['stats'].get('consenso', 0)
        total_iteraciones += outputs['stats'].get('iteraciones', 1)
        n_batches += 1
        
        # Log
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, 20))
            avg_consenso = total_consenso / n_batches
            avg_iter = total_iteraciones / n_batches
            
            # Modulaciones
            modulaciones = outputs['stats'].get('modulaciones', {})
            mod_str = " ".join([f"{k[:3]}:{v:.2f}" for k, v in list(modulaciones.items())[:3]])
            
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | Ppl: {ppl:.2f} | "
                  f"Consenso: {avg_consenso:.3f} | Iter: {avg_iter:.1f} | "
                  f"{mod_str} | {tokens_per_sec:,.0f} tok/s")
    
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluar modelo."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_consenso = 0
    n_batches = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with autocast():
            outputs = model(inputs, targets=targets)
        
        batch_tokens = inputs.numel()
        total_loss += outputs['loss'].item() * batch_tokens
        total_tokens += batch_tokens
        total_consenso += outputs['stats'].get('consenso', 0)
        n_batches += 1
    
    avg_loss = total_loss / total_tokens
    avg_consenso = total_consenso / n_batches
    return avg_loss, avg_consenso


def main():
    parser = argparse.ArgumentParser(description='Entrenar LLARRI v7 Cerebral')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default='base', 
                        choices=['mini', 'base', 'large'])
    parser.add_argument('--data_dir', type=str, default='data/wikitext-103')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    print("=" * 70)
    print("      LLARRI v7 - ARQUITECTURA CEREBRAL")
    print("            ENTRENAMIENTO")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Vocab size (byte-level + special)
    vocab_size = 50257
    
    # Crear modelo
    print(f"\n🧠 Creando modelo {args.model}...")
    if args.model == 'mini':
        model = LLARRIv7Mini(vocab_size=vocab_size)
    elif args.model == 'base':
        model = LLARRIv7Base(vocab_size=vocab_size)
    else:
        model = LLARRIv7Large(vocab_size=vocab_size)
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {n_params:,}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.seq_len, args.batch_size, vocab_size
    )
    
    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(2000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n📂 Cargando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumiendo desde epoch {start_epoch}")
    
    # Checkpoints dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO: {args.epochs} epochs")
    print(f"{'='*70}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'─'*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'─'*70}")
        
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        
        # Eval
        val_loss, val_consenso = evaluate(model, val_loader, device)
        
        epoch_time = (time.time() - epoch_start) / 60
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        
        print(f"\n📊 Epoch {epoch} completada en {epoch_time:.1f} min")
        print(f"   Train Loss: {train_loss:.4f} | Train Ppl: {train_ppl:.2f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Ppl: {val_ppl:.2f}")
        print(f"   Val Consenso: {val_consenso:.3f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"   🏆 Nuevo mejor modelo! Val Loss: {val_loss:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }
        
        # Guardar cada 2 epochs
        if epoch % 2 == 0:
            path = os.path.join(args.checkpoint_dir, f'llarri_v7_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            print(f"   💾 Guardado: {path}")
        
        # Guardar mejor
        if is_best:
            path = os.path.join(args.checkpoint_dir, 'llarri_v7_best.pt')
            torch.save(checkpoint, path)
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Mejor Val Loss: {best_val_loss:.4f}")
    print(f"   Mejor Val Ppl: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
