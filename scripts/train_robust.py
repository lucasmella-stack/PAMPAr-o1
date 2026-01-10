# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v8 - Entrenamiento Robusto y Escalable

Este script entrena el modelo LLARRI v8 con:
- Mixed precision (FP16) para ahorrar VRAM
- Gradient checkpointing opcional
- Gradient accumulation para batch sizes efectivos mayores
- Configuración automática según VRAM disponible
- Logging detallado
- Checkpoints frecuentes
- Early stopping
- Learning rate scheduling (cosine warmup)

Uso:
    python scripts/train_robust.py                    # Auto-detecta VRAM
    python scripts/train_robust.py --preset local     # Forzar 4GB config
    python scripts/train_robust.py --preset server_8  # Forzar 8GB config
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import sentencepiece as spm

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.config import (
    ConfigLLARRI, 
    LOCAL_4GB, SERVER_8GB, SERVER_24GB, SERVER_80GB,
    get_config_for_vram
)
from llarri_o1.cerebro.model import LLARRIv8


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Dataset para texto tokenizado con soporte multi-archivo."""
    
    def __init__(
        self, 
        data_paths: list,
        tokenizer_path: str, 
        seq_len: int = 256, 
        max_tokens: int = 5_000_000,
        encoding: str = 'utf-8'
    ):
        self.seq_len = seq_len
        
        # Cargar tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Cargar y tokenizar todos los archivos
        self.tokens = []
        
        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"  ⚠️ Archivo no encontrado: {data_path}")
                continue
                
            print(f"  📄 Cargando: {data_path}")
            
            try:
                # Leer línea por línea para ahorrar memoria
                buffer = []
                buffer_size = 0
                chunk_size = 50000  # chars por chunk
                
                with open(data_path, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('='):
                            continue
                        
                        buffer.append(line)
                        buffer_size += len(line)
                        
                        # Tokenizar cuando el buffer es grande
                        if buffer_size >= chunk_size:
                            text_chunk = ' '.join(buffer)
                            self.tokens.extend(self.tokenizer.Encode(text_chunk))
                            buffer = []
                            buffer_size = 0
                            
                            if len(self.tokens) >= max_tokens:
                                break
                    
                    # Tokenizar lo que quede en el buffer
                    if buffer and len(self.tokens) < max_tokens:
                        text_chunk = ' '.join(buffer)
                        self.tokens.extend(self.tokenizer.Encode(text_chunk))
                        
            except Exception as e:
                print(f"  ⚠️ Error leyendo {data_path}: {e}")
                continue
            
            if len(self.tokens) >= max_tokens:
                self.tokens = self.tokens[:max_tokens]
                break
        
        print(f"  ✅ Total tokens: {len(self.tokens):,}")
        
        # Calcular número de ejemplos
        self.n_ejemplos = max(1, (len(self.tokens) - 1) // seq_len)
    
    def __len__(self):
        return self.n_ejemplos
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        tokens = self.tokens[start:end]
        
        # Pad si es necesario
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y


# =============================================================================
# TRAINING
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine scheduler con warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    scheduler,
    scaler,
    config,
    device,
    epoch,
    accumulation_steps=1,
    log_every=50
):
    """Entrena una época completa."""
    model.train()
    total_loss = 0
    n_batches = len(dataloader)
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Mixed precision forward
        with autocast(enabled=config.use_mixed_precision):
            outputs = model(x, labels=y, return_info=True)
            loss = outputs['loss'] / accumulation_steps
        
        # Backward con gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Logging
        if batch_idx % log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            
            # Estadísticas de módulos
            stats = model.obtener_estadisticas_modulos()
            stats_str = " | ".join([
                f"{k[:3]}:{v*100:.1f}%" 
                for k, v in stats.items()
            ])
            
            print(f"  Batch {batch_idx}/{n_batches} | "
                  f"Loss: {loss.item()*accumulation_steps:.4f} | "
                  f"Avg: {avg_loss:.4f} | "
                  f"LR: {lr:.2e}")
            print(f"    Módulos: {stats_str}")
            
            model.reset_estadisticas()
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    """Evalúa el modelo en un dataset."""
    model.eval()
    total_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        with autocast(enabled=config.use_mixed_precision):
            outputs = model(x, labels=y)
            total_loss += outputs['loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_samples(model, tokenizer, device, prompts):
    """Genera muestras de texto."""
    model.eval()
    
    samples = []
    for prompt in prompts:
        tokens = tokenizer.Encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        
        text = tokenizer.Decode(output[0].tolist())
        samples.append((prompt, text))
    
    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entrenar LLARRI v8')
    parser.add_argument('--preset', type=str, default='auto',
                       choices=['auto', 'local', 'server_8', 'server_24', 'server_80'],
                       help='Preset de configuración')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas (override preset)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directorio de datos')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint para continuar entrenamiento')
    parser.add_argument('--accumulation', type=int, default=2,
                       help='Gradient accumulation steps')
    args = parser.parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("LLARRI v8 - ENTRENAMIENTO ROBUSTO")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram_gb:.1f} GB")
    else:
        vram_gb = 0
    
    # Configuración
    if args.preset == 'auto':
        config = get_config_for_vram(vram_gb)
        preset_name = f"AUTO ({vram_gb:.0f}GB)"
    elif args.preset == 'local':
        config = LOCAL_4GB
        preset_name = "LOCAL_4GB"
    elif args.preset == 'server_8':
        config = SERVER_8GB
        preset_name = "SERVER_8GB"
    elif args.preset == 'server_24':
        config = SERVER_24GB
        preset_name = "SERVER_24GB"
    else:
        config = SERVER_80GB
        preset_name = "SERVER_80GB"
    
    # Override epochs si se especifica
    if args.epochs:
        config.max_epochs = args.epochs
    
    print(f"\n⚙️ Configuración: {preset_name}")
    print(f"   dim={config.dim}, capas={config.n_capas}, heads={config.n_heads}")
    print(f"   vocab={config.vocab_size}, seq_len={config.max_seq_len}")
    print(f"   batch={config.batch_size}, accumulation={args.accumulation}")
    print(f"   effective_batch={config.batch_size * args.accumulation}")
    print(f"   mixed_precision={config.use_mixed_precision}")
    print(f"   gradient_checkpointing={config.use_gradient_checkpointing}")
    
    # =========================================================================
    # DATOS
    # =========================================================================
    
    print("\n📚 Cargando datos...")
    
    tokenizer_path = os.path.join(args.data_dir, 'tokenizer', 'llarri_bpe.model')
    
    # Buscar archivos de datos
    data_files_train = []
    data_files_valid = []
    
    # WikiText
    wikitext_dir = os.path.join(args.data_dir, 'wikitext-103', 'wikitext-103-raw')
    if os.path.exists(wikitext_dir):
        train_file = os.path.join(wikitext_dir, 'wiki.train.raw')
        valid_file = os.path.join(wikitext_dir, 'wiki.valid.raw')
        if os.path.exists(train_file):
            data_files_train.append(train_file)
        if os.path.exists(valid_file):
            data_files_valid.append(valid_file)
    
    # Corpus español (si existe)
    spanish_dir = os.path.join(args.data_dir, 'spanish')
    if os.path.exists(spanish_dir):
        for f in os.listdir(spanish_dir):
            if f.endswith('.txt'):
                data_files_train.append(os.path.join(spanish_dir, f))
    
    if not data_files_train:
        print("❌ No se encontraron archivos de datos!")
        print(f"   Buscando en: {args.data_dir}")
        return
    
    print(f"   Archivos train: {len(data_files_train)}")
    print(f"   Archivos valid: {len(data_files_valid)}")
    
    # Crear datasets
    # Ajustar max_tokens según VRAM
    max_tokens = min(5_000_000, int(vram_gb * 1_000_000)) if vram_gb > 0 else 2_000_000
    
    train_dataset = TextDataset(
        data_paths=data_files_train,
        tokenizer_path=tokenizer_path,
        seq_len=config.max_seq_len,
        max_tokens=max_tokens,
    )
    
    if data_files_valid:
        valid_dataset = TextDataset(
            data_paths=data_files_valid,
            tokenizer_path=tokenizer_path,
            seq_len=config.max_seq_len,
            max_tokens=500_000,
        )
    else:
        valid_dataset = None
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        valid_loader = None
    
    # =========================================================================
    # MODELO
    # =========================================================================
    
    print("\n🧠 Inicializando modelo...")
    
    model = LLARRIv8(config).to(device)
    
    # Registrar tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    model.registrar_tokenizer(tokenizer)
    
    # Contar parámetros
    params = model.contar_parametros()
    print(f"\n   Parámetros totales: {params['total']:,}")
    for k, v in params.items():
        if k != 'total' and v > 0:
            print(f"     {k}: {v:,}")
    
    # Cargar checkpoint si existe
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n📥 Cargando checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Manejar diferentes formatos de checkpoint
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        start_epoch = ckpt.get('epoch', 0) + 1 if isinstance(ckpt, dict) else 0
        best_val_loss = ckpt.get('val_loss', float('inf')) if isinstance(ckpt, dict) else float('inf')
        print(f"   Continuando desde época {start_epoch}")
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    total_steps = len(train_loader) * config.max_epochs // args.accumulation
    warmup_steps = min(config.warmup_steps, total_steps // 10)
    
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=config.use_mixed_precision)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    print(f"\n   Épocas: {config.max_epochs}")
    print(f"   Batches/época: {len(train_loader)}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # Directorio de checkpoints
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    
    # Stats para guardar
    training_stats = {
        'config': {
            'preset': preset_name,
            'dim': config.dim,
            'n_capas': config.n_capas,
            'vocab_size': config.vocab_size,
            'params': params['total'],
        },
        'epochs': [],
    }
    
    # Prompts para generación
    test_prompts = ['The', 'In the', 'Science is']
    
    no_improvement = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(start_epoch, config.max_epochs):
        print(f"\n{'='*20} ÉPOCA {epoch+1}/{config.max_epochs} {'='*20}")
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, epoch,
            accumulation_steps=args.accumulation,
            log_every=100
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\n  Época {epoch+1} completada en {epoch_time:.1f}s")
        print(f"  Loss promedio: {train_loss:.4f}")
        
        # Validation
        if valid_loader:
            val_loss, val_ppl = evaluate(model, valid_loader, config, device)
            print(f"  Validación | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                
                torch.save({
                    'model': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, ckpt_dir / 'llarri_v8_best.pt')
                print(f"  ✅ Nuevo mejor modelo guardado!")
            else:
                no_improvement += 1
                print(f"  ⚠️ Sin mejora por {no_improvement} épocas")
        else:
            val_loss, val_ppl = None, None
        
        # Generar muestras
        print("\n  Muestras de generación:")
        samples = generate_samples(model, tokenizer, device, test_prompts)
        for prompt, text in samples:
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    '{prompt}' → {text_preview}")
        
        # Checkpoint periódico
        if (epoch + 1) % 3 == 0:
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, ckpt_dir / f'llarri_v8_epoch_{epoch+1}.pt')
        
        # Stats
        training_stats['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'time': epoch_time,
        })
        
        # Guardar stats
        with open(ckpt_dir / 'training_stats_robust.json', 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Early stopping
        if no_improvement >= patience:
            print(f"\n⛔ Early stopping después de {patience} épocas sin mejora")
            break
    
    # =========================================================================
    # FINAL
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    # Guardar modelo final
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'epoch': epoch,
        'val_loss': val_loss,
        'train_loss': train_loss,
    }, ckpt_dir / 'llarri_v8_final.pt')
    
    print(f"\n✅ Modelo final guardado en: {ckpt_dir / 'llarri_v8_final.pt'}")
    print(f"✅ Mejor modelo guardado en: {ckpt_dir / 'llarri_v8_best.pt'}")
    print(f"✅ Stats guardadas en: {ckpt_dir / 'training_stats_robust.json'}")


if __name__ == '__main__':
    main()
