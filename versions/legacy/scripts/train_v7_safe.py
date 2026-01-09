# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7 - Entrenamiento SEGURO para GTX 1650
==============================================

Version con diagnóstico de NaN y configuración conservadora.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v7 import (
    LLARRIv7Cerebral,
)


def check_nan(tensor, name):
    """Verifica NaN/Inf y reporta."""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"⚠️  {name}: NaN={has_nan}, Inf={has_inf}, shape={tensor.shape}")
        print(f"    stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        return True
    return False


def check_gradients(model):
    """Verifica gradientes."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"⚠️  Gradiente malo en: {name}")
                return True
    return False


class WikiTextDataset(Dataset):
    """Dataset para WikiText-103."""
    
    def __init__(self, data_path: str, seq_len: int = 64, vocab_size: int = 256, max_tokens: int = 5_000_000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        tokens_list = []
        total_read = 0
        chunk_size = 1_000_000
        
        print(f"   Cargando {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            while total_read < max_tokens:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Byte-level tokenization (0-255)
                chunk_tokens = [ord(c) % vocab_size for c in chunk]
                tokens_list.extend(chunk_tokens)
                total_read += len(chunk_tokens)
                if total_read >= max_tokens:
                    tokens_list = tokens_list[:max_tokens]
                    break
        
        self.tokens = torch.tensor(tokens_list, dtype=torch.long)
        del tokens_list
        
        self.n_samples = (len(self.tokens) - 1) // seq_len
        print(f"   Tokens: {len(self.tokens):,} | Samples: {self.n_samples:,}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


def train_epoch(model, loader, optimizer, device, epoch, check_nan_mode=False):
    """Entrenar una época."""
    model.train()
    total_loss = 0
    total_tokens = 0
    n_batches = 0
    
    start_time = time.time()
    log_interval = 50
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward (sin mixed precision para estabilidad)
        outputs = model(inputs, targets=targets)
        loss = outputs['loss']
        
        # Check NaN en loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf en loss en batch {batch_idx}")
            if check_nan_mode:
                # Diagnóstico detallado
                print("📊 Diagnóstico:")
                check_nan(model.embedding.weight, "embedding.weight")
                check_nan(outputs['logits'], "logits")
                with torch.no_grad():
                    x = model.embedding(inputs)
                    check_nan(x, "post-embedding")
                    positions = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
                    x = x + model.pos_encoding(positions)
                    check_nan(x, "post-pos-encoding")
                    mod = model.talamo(x)
                    check_nan(mod, "modulación tálamo")
            return float('nan')
        
        # Backward
        loss.backward()
        
        # Gradient clipping ANTES de step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Check gradientes
        if check_nan_mode and check_gradients(model):
            print(f"❌ Gradientes NaN en batch {batch_idx}")
            return float('nan')
        
        optimizer.step()
        
        # Stats
        batch_tokens = inputs.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1
        
        # Log
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, 20))
            
            consenso = outputs['stats'].get('consenso_medio', 0)
            
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | Ppl: {ppl:.2f} | "
                  f"Consenso: {consenso:.3f} | "
                  f"GradNorm: {grad_norm:.2f} | "
                  f"{tokens_per_sec:,.0f} tok/s")
    
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluar modelo."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs, targets=targets)
        loss = outputs['loss']
        
        if torch.isnan(loss):
            return float('nan')
        
        batch_tokens = inputs.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description='Entrenar LLARRI v7 - Version Segura')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)  # LR conservador
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default='data/wikitext-103')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--debug', action='store_true', help='Modo diagnóstico de NaN')
    args = parser.parse_args()
    
    print("=" * 70)
    print("      LLARRI v7 - ENTRENAMIENTO SEGURO")
    print("            (Sin mixed precision)")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {mem_gb:.1f} GB")
    
    # Vocab size byte-level
    vocab_size = 256
    
    # Crear modelo con dimensión pequeña
    print(f"\n🧠 Creando modelo dim={args.dim}...")
    model = LLARRIv7Cerebral(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=2 if args.dim <= 64 else 4,
        usar_hipocampo=False,  # Sin hipocampo para ahorrar memoria
        max_iteraciones=1,     # Solo una iteración para empezar
        actividad_basal=0.2,   # Un poco más de actividad basal
    )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {n_params:,}")
    
    # Verificar pesos iniciales
    print("\n🔍 Verificando pesos iniciales...")
    has_bad_weights = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"   ⚠️ Peso malo: {name}")
            has_bad_weights = True
    
    if not has_bad_weights:
        print("   ✅ Todos los pesos OK")
    
    # Data
    train_path = Path(args.data_dir) / "wikitext-103-raw" / "wiki.train.raw"
    val_path = Path(args.data_dir) / "wikitext-103-raw" / "wiki.valid.raw"
    
    if not train_path.exists():
        print(f"\n❌ No se encontró: {train_path}")
        return
    
    print("\n📚 Cargando datos...")
    print("   Train:")
    train_dataset = WikiTextDataset(str(train_path), args.seq_len, vocab_size, max_tokens=2_000_000)
    print("   Val:")
    val_dataset = WikiTextDataset(str(val_path), args.seq_len, vocab_size, max_tokens=500_000)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Optimizer con LR bajo
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # Test forward
    print("\n🧪 Test forward...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_input = test_batch[0][:2].to(device)  # Solo 2 samples
        test_target = test_batch[1][:2].to(device)
        
        outputs = model(test_input, targets=test_target)
        test_loss = outputs['loss'].item()
        print(f"   Test loss: {test_loss:.4f}")
        
        if math.isnan(test_loss):
            print("   ❌ NaN en test forward!")
            print("   📊 Diagnóstico:")
            check_nan(outputs['logits'], "logits")
            
            # Verificar paso a paso
            x = model.embedding(test_input)
            check_nan(x, "embedding output")
            
            positions = torch.arange(test_input.shape[1], device=device).unsqueeze(0)
            x = x + model.pos_encoding(positions)
            check_nan(x, "with pos encoding")
            
            mod = model.talamo(x)
            check_nan(mod, "modulación")
            
            # Probar cada módulo
            for nombre, modulo in model.modulos.items():
                out = modulo(x)
                if check_nan(out, f"módulo {nombre}"):
                    break
            
            return
        else:
            print("   ✅ Forward OK")
    
    # Training
    print("\n" + "=" * 70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    best_val_loss = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            check_nan_mode=args.debug
        )
        
        if math.isnan(train_loss):
            print("\n❌ Training divergió (NaN)")
            break
        
        val_loss = evaluate(model, val_loader, device)
        
        if math.isnan(val_loss):
            print("\n❌ Validation divergió (NaN)")
            break
        
        val_ppl = math.exp(min(val_loss, 20))
        print(f"\n📊 Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val PPL = {val_ppl:.2f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.checkpoint_dir) / "llarri_v7_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"   💾 Mejor modelo guardado: {save_path}")
        
        # Save checkpoint cada 2 epochs
        if epoch % 2 == 0:
            save_path = Path(args.checkpoint_dir) / f"llarri_v7_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"   💾 Checkpoint: {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Mejor Val Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
