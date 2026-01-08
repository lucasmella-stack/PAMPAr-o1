# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi
"""
LLARRI v6b Training Script - Reflexión V2 (Early Exit Inteligente)

CAMBIO CLAVE: Early exit solo cuando token_predicho == target
- Ya no sale "siempre" como v6
- Deberíamos ver early_exit_rate mucho más bajo
- Neural 2 debería usarse más frecuentemente
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import argparse
import math

from llarri_o1.models.language_model_v6b import LLARRILanguageModelV6b, LLARRIConfigV6b


class TextDataset(Dataset):
    """Dataset byte-level para texto."""
    
    def __init__(self, text: str, seq_length: int = 128):
        self.data = torch.tensor(
            list(text.encode('utf-8')),
            dtype=torch.long
        )
        self.seq_length = seq_length
        
    def __len__(self):
        return max(1, len(self.data) - self.seq_length - 1)
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_wikitext():
    """Cargar datos de WikiText-103."""
    data_path = Path("data/wikitext-103")
    
    if not data_path.exists():
        print("📥 Descargando WikiText-103...")
        import urllib.request
        import zipfile
        
        data_path.mkdir(parents=True, exist_ok=True)
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        zip_path = data_path / "wikitext.zip"
        
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        zip_path.unlink()
        print("✓ WikiText-103 descargado")
    
    # Cargar train y valid
    train_path = data_path / "wikitext-103-raw" / "wiki.train.raw"
    valid_path = data_path / "wikitext-103-raw" / "wiki.valid.raw"
    
    train_text = train_path.read_text(encoding='utf-8')[:5_000_000]  # 5MB
    valid_text = valid_path.read_text(encoding='utf-8')[:500_000]    # 500KB
    
    return train_text, valid_text


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, log_interval=100):
    """Entrena una época."""
    model.train()
    model.reset_early_exit_stats()
    total_loss = 0
    n_batches = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward con stats de early exit
        logits, loss, stats = model(x, y, return_early_exit_stats=True)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_idx * x.shape[0] * x.shape[1]) / elapsed
            
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Ppl: {math.exp(avg_loss):.2f} | "
                  f"EarlyExit: {stats['early_exit_rate']*100:.1f}% | "
                  f"Neural2: {stats['neural2_usage']*100:.1f}% | "
                  f"TokenMatch: {stats['coincide_target']*100:.1f}% | "
                  f"{tokens_per_sec:.0f} tok/s")
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, valid_loader, device):
    """Evalúa el modelo."""
    model.eval()
    model.reset_early_exit_stats()
    total_loss = 0
    n_batches = 0
    
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        logits, loss, stats = model(x, y, return_early_exit_stats=True)
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    return avg_loss, stats


def main():
    parser = argparse.ArgumentParser(description="Entrenar LLARRI v6b")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENTRENAMIENTO LLARRI v6b - REFLEXIÓN V2 (EARLY EXIT INTELIGENTE)")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sequence length: {args.seq_len}")
    print()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Datos
    print("📚 Cargando datos...")
    train_text, valid_text = load_wikitext()
    
    train_dataset = TextDataset(train_text, args.seq_len)
    valid_dataset = TextDataset(valid_text, args.seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Valid: {len(valid_dataset):,} samples")
    print()
    
    # Modelo
    config = LLARRIConfigV6b(
        vocab_size=256,
        embed_dim=128,
        n_heads=4,
        max_length=args.seq_len,
        dropout=0.1
    )
    
    model = LLARRILanguageModelV6b(config).to(device)
    print()
    
    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    # Entrenamiento
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    
    print("\n" + "=" * 70)
    print("COMENZANDO ENTRENAMIENTO")
    print("=" * 70)
    
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'─' * 70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'─' * 70}")
        
        # Entrenar
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # Evaluar
        val_loss, val_stats = evaluate(model, valid_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch} completada en {epoch_time/60:.1f} min")
        print(f"   Train Loss: {train_loss:.4f} | Train Ppl: {math.exp(train_loss):.2f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Ppl: {math.exp(val_loss):.2f}")
        print(f"   Early Exit Rate: {val_stats['early_exit_rate']*100:.1f}%")
        print(f"   Neural2 Usage: {val_stats['neural2_usage']*100:.1f}%")
        
        # Guardar checkpoint
        if epoch % 2 == 0:
            ckpt_path = checkpoint_dir / f"llarri_v6b_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, ckpt_path)
            print(f"   💾 Checkpoint: {ckpt_path}")
        
        # Mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "llarri_v6b_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_path)
            print(f"   🏆 Nuevo mejor modelo! Val Loss: {val_loss:.4f}")
    
    total_time = time.time() - training_start
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
    print(f"🏆 Mejor Val Loss: {best_val_loss:.4f}")
    print(f"📊 Mejor Perplexity: {math.exp(best_val_loss):.2f}")
    
    # Test de generación
    print("\n" + "=" * 70)
    print("TEST DE GENERACIÓN")
    print("=" * 70)
    
    model.eval()
    prompts = [
        "The history of",
        "In the year 2025,",
        "Science has proven that"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = model.generate_text(prompt, max_length=100, temperature=0.8)
        print(f"Generated: {generated}")
        print("-" * 50)


if __name__ == "__main__":
    main()
