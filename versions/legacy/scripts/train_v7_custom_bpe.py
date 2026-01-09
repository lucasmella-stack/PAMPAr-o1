# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucas
"""
LLARRI v7.2 - Entrenamiento con BPE Personalizado (8K vocab)
============================================================

Configuración optimizada para GTX 1650 (4.3GB VRAM):
- vocab_size: 8,000 (vs 50,257 de GPT-2 que crasheó)
- d_model: 128 
- n_heads: 4
- Parámetros estimados: ~5M (cabe en VRAM)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sentencepiece as spm

# Añadir path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


class WikiTextBPEDataset(Dataset):
    """Dataset usando nuestro tokenizer BPE personalizado."""
    
    def __init__(self, file_path: str, tokenizer_path: str, seq_length: int = 256, max_lines: int = 100000):
        self.seq_length = seq_length
        
        # Cargar tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # Tokens especiales
        self.pad_id = self.sp.pad_id()  # 0
        self.bos_id = self.sp.bos_id()  # 2
        self.eos_id = self.sp.eos_id()  # 3
        
        # Leer y tokenizar por líneas para evitar crash
        print(f"   Cargando {file_path}...")
        self.tokens = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Limitar líneas para evitar memoria excesiva
        lines = lines[:max_lines]
        print(f"   Procesando {len(lines):,} líneas...")
        
        # Tokenizar línea por línea
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 10:  # Ignorar líneas muy cortas
                tokens = self.sp.encode(line)
                self.tokens.extend(tokens)
            
            if (i + 1) % 20000 == 0:
                print(f"      {i+1:,} líneas procesadas...")
        
        print(f"   Total tokens: {len(self.tokens):,}")
        
        # Calcular número de secuencias
        self.n_sequences = max(1, (len(self.tokens) - 1) // seq_length)
        print(f"   Secuencias de {seq_length}: {self.n_sequences:,}")
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        
        chunk = self.tokens[start:end]
        
        # Padding si es necesario
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [self.pad_id] * (self.seq_length + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def count_parameters(model):
    """Cuenta parámetros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, 
                grad_accum_steps=2, max_grad_norm=1.0):
    """Entrena una época con gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward - modelo retorna dict
        output = model(x)
        logits = output['logits']
        
        # Loss con label smoothing
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / grad_accum_steps
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        n_batches += 1
        
        # Progress
        if batch_idx % 200 == 0:
            print(f"      Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item() * grad_accum_steps:.4f}")
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        logits = output['logits']
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = math.exp(min(avg_loss, 20))  # Cap para evitar overflow
    
    return avg_loss, perplexity


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device, max_tokens=50, temperature=0.8, top_p=0.9):
    """Genera texto de muestra."""
    model.eval()
    
    # Tokenizar prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    generated = tokens.copy()
    
    for _ in range(max_tokens):
        # Solo usar últimos 256 tokens
        curr_input = input_ids[:, -256:]
        
        output = model(curr_input)
        logits = output['logits']
        next_logits = logits[0, -1, :] / temperature
        
        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        # Stop en EOS
        if next_token == tokenizer.eos_id():
            break
        
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    
    return tokenizer.decode(generated)


def main():
    print("=" * 60)
    print("   LLARRI v7.2 - BPE Personalizado (8K vocab)")
    print("=" * 60)
    
    # Configuración para GTX 1650
    config = {
        'vocab_size': 8000,       # Nuestro tokenizer
        'd_model': 128,           # Mantener 128
        'n_heads': 4,
        'n_layers': 4,
        'dropout': 0.15,
        'seq_length': 256,
        'batch_size': 16,         # Pequeño para VRAM
        'grad_accum': 4,          # Effective batch = 64
        'epochs': 15,
        'lr': 3e-4,
        'warmup_steps': 500,
        'label_smoothing': 0.1,
        'patience': 4,            # Early stopping
    }
    
    # Paths
    base_path = Path(__file__).parent.parent
    tokenizer_path = base_path / "data" / "tokenizer" / "llarri_bpe.model"
    train_path = base_path / "data" / "wikitext-103" / "wikitext-103-raw" / "wiki.train.raw"
    val_path = base_path / "data" / "wikitext-103" / "wikitext-103-raw" / "wiki.valid.raw"
    checkpoint_dir = base_path / "checkpoints"
    
    # Verificar tokenizer
    if not tokenizer_path.exists():
        print("❌ Tokenizer no encontrado! Ejecuta train_tokenizer.py primero")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Optimizaciones de memoria
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # Cargar tokenizer
    print(f"\n📖 Cargando tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    print(f"   Vocab size: {sp.vocab_size()}")
    
    # Datasets
    print(f"\n📚 Cargando datasets...")
    train_dataset = WikiTextBPEDataset(str(train_path), str(tokenizer_path), config['seq_length'], max_lines=100000)
    val_dataset = WikiTextBPEDataset(str(val_path), str(tokenizer_path), config['seq_length'], max_lines=10000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Modelo
    print(f"\n🧠 Creando modelo LLARRI v7.2...")
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        usar_hipocampo=True,
        capacidad_memoria=2000,  # Reducido para VRAM
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    print(f"   Tamaño estimado: {total_params * 4 / 1e6:.1f} MB")
    
    # Memory check
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM usado (modelo): {allocated:.2f} GB")
    
    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * config['epochs'] // config['grad_accum']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps,
        pct_start=config['warmup_steps'] / total_steps,
        anneal_strategy='cos'
    )
    
    # Loss con label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=sp.pad_id(),
        label_smoothing=config['label_smoothing']
    )
    
    # Training
    print(f"\n🏋️ Iniciando entrenamiento...")
    print(f"   Épocas: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['grad_accum']})")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Patience: {config['patience']}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"📅 Época {epoch}/{config['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            grad_accum_steps=config['grad_accum']
        )
        
        # Eval
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        
        print(f"\n   📊 Train Loss: {train_loss:.4f}")
        print(f"   📊 Val Loss: {val_loss:.4f}")
        print(f"   📊 Val PPL: {val_ppl:.2f}")
        
        # Generar muestra
        print(f"\n   📝 Muestra generada:")
        sample = generate_sample(model, sp, "The ", device, max_tokens=40, temperature=0.8)
        print(f"   \"{sample}\"")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / "llarri_v7.2_bpe_best.pt")
            print(f"   ✅ Nuevo mejor modelo guardado! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ Sin mejora ({patience_counter}/{config['patience']})")
            
            if patience_counter >= config['patience']:
                print(f"\n   🛑 Early stopping!")
                break
        
        # Checkpoint periódico
        if epoch % 3 == 0:
            torch.save(checkpoint, checkpoint_dir / f"llarri_v7.2_bpe_epoch_{epoch}.pt")
        
        # Memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"   ✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Mejor Val Loss: {best_val_loss:.4f}")
    print(f"   Mejor Val PPL: {math.exp(best_val_loss):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
