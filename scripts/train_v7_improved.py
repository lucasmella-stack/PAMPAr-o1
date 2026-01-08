# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.1 - Entrenamiento Mejorado

Mejoras implementadas:
1. Dropout para regularización (evita sobreajuste)
2. Más datos de entrenamiento (5M tokens)
3. Early stopping basado en val_loss plateau
4. Label smoothing para mejor generalización
5. Learning rate scheduling con warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


class WikiTextDataset(Dataset):
    """Dataset para WikiText-103."""
    
    def __init__(self, data_path: str, seq_len: int = 128, max_tokens: int = None):
        self.seq_len = seq_len
        
        # Cargar texto de manera eficiente
        print(f'  📖 Cargando {data_path}...')
        tokens = []
        with open(data_path, 'r', encoding='utf-8') as f:
            while len(tokens) < (max_tokens or float('inf')):
                chunk = f.read(100000)  # Leer en chunks
                if not chunk:
                    break
                tokens.extend([ord(c) % 256 for c in chunk])
        
        if max_tokens:
            self.tokens = tokens[:max_tokens]
        else:
            self.tokens = tokens
            
        print(f'  📊 Tokens cargados: {len(self.tokens):,}')
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class LabelSmoothingLoss(nn.Module):
    """Cross entropy con label smoothing para mejor generalización."""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch * seq, vocab_size)
            targets: (batch * seq,)
        """
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # NLL loss con confidence
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Smooth loss (distribución uniforme)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combinar
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


def main():
    print('=' * 70)
    print('   LLARRI v7.1 - ENTRENAMIENTO MEJORADO')
    print('   Con dropout, label smoothing, y más datos')
    print('=' * 70)
    
    # === CONFIGURACIÓN MEJORADA ===
    config = {
        'vocab_size': 256,
        'dim': 128,  # AUMENTADO para más capacidad
        'n_heads': 4,  # AUMENTADO
        'dropout': 0.15,  
        'usar_hipocampo': False,
        'max_iteraciones': 1,
        'actividad_basal': 0.2,
        
        # Entrenamiento mejorado
        'seq_len': 128,
        'batch_size': 32,  # Reducido por más VRAM
        'grad_accum_steps': 4,
        'epochs': 15,  
        'lr': 3e-4,  # Un poco más bajo para modelo más grande
        'warmup_steps': 500,
        'label_smoothing': 0.1,
        'gradient_clip': 0.5,
        'train_tokens': 2_000_000,
        'val_tokens': 200_000,
        
        # Early stopping
        'patience': 3,  # Epochs sin mejora antes de parar
        'min_delta': 0.001,  # Mejora mínima significativa
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n📱 Device: {device}')
    
    if device.type == 'cuda':
        print(f'   GPU: {torch.cuda.get_device_name()}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # === CREAR MODELO CON DROPOUT ===
    print('\n🧠 Creando modelo LLARRI v7.1...')
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],  # NUEVO
        usar_hipocampo=config['usar_hipocampo'],
        max_iteraciones=config['max_iteraciones'],
        actividad_basal=config['actividad_basal'],
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'   Parámetros: {num_params:,}')
    print(f'   Dropout: {config["dropout"]}')
    print(f'   Label Smoothing: {config["label_smoothing"]}')
    
    # === CARGAR DATOS ===
    print('\n📚 Cargando WikiText-103...')
    data_dir = Path('data/wikitext-103/wikitext-103-raw')
    
    train_dataset = WikiTextDataset(
        data_dir / 'wiki.train.raw',
        seq_len=config['seq_len'],
        max_tokens=config['train_tokens'],
    )
    
    val_dataset = WikiTextDataset(
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
        shuffle=False,
        num_workers=0,
        pin_memory=True,
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
    
    # Learning rate scheduler con warmup y cosine decay
    # Ajustar para gradient accumulation
    total_steps = (len(train_loader) // config['grad_accum_steps']) * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps + 100,  # Buffer extra
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
    )
    
    # Label smoothing loss
    criterion = LabelSmoothingLoss(
        vocab_size=config['vocab_size'],
        smoothing=config['label_smoothing'],
    )
    
    # === EARLY STOPPING ===
    best_val_loss = float('inf')
    patience_counter = 0
    
    # === ENTRENAMIENTO ===
    print('\n' + '=' * 70)
    print('   INICIANDO ENTRENAMIENTO')
    print('=' * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    grad_accum = config['grad_accum_steps']
    
    for epoch in range(1, config['epochs'] + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}')
        
        optimizer.zero_grad()  # Mover fuera del loop
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            result = model(x, targets=y)
            logits = result['logits']
            
            # Loss con label smoothing (normalizado por accum steps)
            loss = criterion(logits, y) / grad_accum
            
            # Verificar NaN
            if torch.isnan(loss):
                print(f'\n⚠️ NaN detectado en batch {batch_idx}')
                continue
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * grad_accum
            train_steps += 1
            
            # Actualizar barra
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum:.4f}',
                'lr': f'{current_lr:.2e}',
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
        stats = result.get('stats', {})
        consenso = stats.get('consenso_mean', 0) - stats.get('conflicto_mean', 0)
        
        print(f'\n📊 Epoch {epoch}:')
        print(f'   Train Loss: {train_loss:.4f}')
        print(f'   Val Loss:   {val_loss:.4f}')
        print(f'   Val PPL:    {val_ppl:.2f}')
        print(f'   Consenso:   {consenso:+.3f}')
        print(f'   LR:         {scheduler.get_last_lr()[0]:.2e}')
        
        # --- EARLY STOPPING ---
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Guardar mejor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': config,
            }, 'checkpoints/llarri_v7_improved_best.pt')
            print(f'   💾 Mejor modelo guardado!')
        else:
            patience_counter += 1
            print(f'   ⏳ Sin mejora ({patience_counter}/{config["patience"]})')
            
            if patience_counter >= config['patience']:
                print(f'\n🛑 Early stopping! No hay mejora en {config["patience"]} epochs.')
                break
        
        # Checkpoint periódico
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, f'checkpoints/llarri_v7_improved_epoch_{epoch}.pt')
    
    # === RESUMEN FINAL ===
    print('\n' + '=' * 70)
    print('   ✅ ENTRENAMIENTO COMPLETADO')
    print(f'   Mejor Val Loss: {best_val_loss:.4f}')
    print(f'   Mejor Val PPL:  {torch.exp(torch.tensor(best_val_loss)).item():.2f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
