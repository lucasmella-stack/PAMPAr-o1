# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.2 - Entrenamiento con BPE Tokenizer

Mejora principal: Usa tokenización BPE (como GPT-2) en lugar de bytes.
Esto permite al modelo aprender palabras y subpalabras reales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


class WikiTextBPEDataset(Dataset):
    """Dataset con tokenización BPE."""
    
    def __init__(self, data_path: str, seq_len: int = 128, max_tokens: int = None):
        self.seq_len = seq_len
        
        # Usar tokenizer GPT-2 (50257 tokens)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        print(f'  📖 Cargando {data_path}...')
        
        # Cargar y tokenizar en chunks
        tokens = []
        with open(data_path, 'r', encoding='utf-8') as f:
            while len(tokens) < (max_tokens or float('inf')):
                chunk = f.read(100000)
                if not chunk:
                    break
                chunk_tokens = self.tokenizer.encode(chunk)
                tokens.extend(chunk_tokens)
        
        self.tokens = tokens[:max_tokens] if max_tokens else tokens
        print(f'  📊 Tokens BPE: {len(self.tokens):,}')
        print(f'  📊 Vocab size: {self.tokenizer.n_vocab:,}')
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class LabelSmoothingLoss(nn.Module):
    """Cross entropy con label smoothing."""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


def main():
    print('=' * 70)
    print('   🧠 LLARRI v7.2 - ENTRENAMIENTO CON BPE')
    print('   Tokenización de subpalabras (como GPT-2)')
    print('=' * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n📱 Device: {device}')
    
    if device.type == 'cuda':
        print(f'   GPU: {torch.cuda.get_device_name()}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # === CONFIGURACIÓN ===
    # GPT-2 tiene vocab_size=50257, pero usaremos modelo más pequeño
    config = {
        'vocab_size': 50257,  # Vocabulario GPT-2
        'dim': 128,
        'n_heads': 4,
        'dropout': 0.1,
        'usar_hipocampo': False,
        'max_iteraciones': 1,
        'actividad_basal': 0.2,
        
        'seq_len': 128,
        'batch_size': 16,  # Reducido por vocab grande
        'grad_accum_steps': 4,
        'epochs': 10,
        'lr': 3e-4,
        'label_smoothing': 0.1,
        'gradient_clip': 1.0,
        'train_tokens': 500_000,  # Menos tokens BPE = más texto real
        'val_tokens': 50_000,
        
        'patience': 3,
    }
    
    # === CREAR MODELO ===
    print('\n🧠 Creando modelo LLARRI v7.2 (BPE)...')
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        usar_hipocampo=config['usar_hipocampo'],
        max_iteraciones=config['max_iteraciones'],
        actividad_basal=config['actividad_basal'],
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'   Parámetros: {num_params:,}')
    print(f'   Vocab: {config["vocab_size"]:,} (BPE GPT-2)')
    print(f'   Dim: {config["dim"]}')
    
    # === CARGAR DATOS ===
    print('\n📚 Cargando WikiText-103 con BPE...')
    data_dir = Path('data/wikitext-103/wikitext-103-raw')
    
    train_dataset = WikiTextBPEDataset(
        data_dir / 'wiki.train.raw',
        seq_len=config['seq_len'],
        max_tokens=config['train_tokens'],
    )
    
    val_dataset = WikiTextBPEDataset(
        data_dir / 'wiki.valid.raw',
        seq_len=config['seq_len'],
        max_tokens=config['val_tokens'],
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=0,
    )
    
    print(f'   Batches train: {len(train_loader):,}')
    print(f'   Batches val: {len(val_loader):,}')
    
    # === OPTIMIZACIÓN ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    
    total_steps = (len(train_loader) // config['grad_accum_steps']) * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps + 100,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    criterion = LabelSmoothingLoss(
        vocab_size=config['vocab_size'],
        smoothing=config['label_smoothing'],
    )
    
    # === ENTRENAMIENTO ===
    print('\n' + '=' * 70)
    print('   INICIANDO ENTRENAMIENTO BPE')
    print('=' * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    grad_accum = config['grad_accum_steps']
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}')
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            result = model(x, targets=y)
            logits = result['logits']
            
            loss = criterion(logits, y) / grad_accum
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * grad_accum
            train_steps += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
        
        train_loss /= max(train_steps, 1)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                result = model(x, targets=y)
                loss = F.cross_entropy(
                    result['logits'].view(-1, config['vocab_size']),
                    y.view(-1),
                )
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_steps += 1
        
        val_loss /= max(val_steps, 1)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        # --- STATS ---
        print(f'\n📊 Epoch {epoch}:')
        print(f'   Train Loss: {train_loss:.4f}')
        print(f'   Val Loss:   {val_loss:.4f}')
        print(f'   Val PPL:    {val_ppl:.2f}')
        print(f'   LR:         {scheduler.get_last_lr()[0]:.2e}')
        
        # --- EARLY STOPPING ---
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': config,
            }, 'checkpoints/llarri_v7_bpe_best.pt')
            print(f'   💾 Mejor modelo guardado!')
        else:
            patience_counter += 1
            print(f'   ⏳ Sin mejora ({patience_counter}/{config["patience"]})')
            
            if patience_counter >= config['patience']:
                print(f'\n🛑 Early stopping!')
                break
        
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, f'checkpoints/llarri_v7_bpe_epoch_{epoch}.pt')
    
    # === RESUMEN ===
    print('\n' + '=' * 70)
    print('   ✅ ENTRENAMIENTO BPE COMPLETADO')
    print(f'   Mejor Val Loss: {best_val_loss:.4f}')
    print(f'   Mejor Val PPL:  {torch.exp(torch.tensor(best_val_loss)).item():.2f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
