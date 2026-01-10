#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
PampaR - Script de Entrenamiento Unificado
==========================================

Entrena el modelo PampaR con corpus WikiText-103.
Optimizado para GPUs de 4GB+ VRAM.

Uso:
    python scripts/train.py                     # Entrenamiento básico
    python scripts/train.py --epochs 10         # 10 épocas
    python scripts/train.py --resume            # Continuar desde checkpoint
    python scripts/train.py --tokens 10M        # Limitar tokens (10M, 50M, etc)
    python scripts/train.py --batch-size 32     # Batch size personalizado

Autor: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import sentencepiece as spm

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))
from pampar.config import ConfigPampaR, LOCAL_4GB, LOCAL_4GB_MAX
from pampar.cerebro.model import PampaR


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

class TrainConfig:
    """Configuración de entrenamiento"""
    def __init__(self):
        # Paths
        self.corpus_path = "data/wikitext-103/wikitext-103-raw/wiki.train.raw"
        self.val_path = "data/wikitext-103/wikitext-103-raw/wiki.valid.raw"
        self.tokenizer_path = "data/tokenizer/llarri_bpe.model"
        self.checkpoint_dir = "checkpoints"
        
        # Entrenamiento
        self.batch_size = 16
        self.gradient_accumulation = 4
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.max_grad_norm = 1.0
        self.use_mixed_precision = True
        
        # Checkpoints
        self.save_every_steps = 2000
        self.eval_every_steps = 500
        
        # Límites
        self.max_tokens = None  # None = todo el corpus


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Dataset simple para tokenizar texto"""
    
    def __init__(self, corpus_path: str, tokenizer_path: str, 
                 seq_len: int = 256, max_tokens: Optional[int] = None):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        
        print(f"📖 Cargando: {corpus_path}")
        self.tokens = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if len(line) > 20:  # Solo líneas con contenido
                    ids = self.sp.Encode(line)
                    self.tokens.extend(ids)
                
                # Progreso
                if (i + 1) % 50000 == 0:
                    print(f"    Líneas: {i+1:,} | Tokens: {len(self.tokens):,}")
                
                # Límite de tokens
                if max_tokens and len(self.tokens) >= max_tokens:
                    self.tokens = self.tokens[:max_tokens]
                    print(f"    ✓ Límite alcanzado: {max_tokens:,} tokens")
                    break
        
        # Crear secuencias
        n_seqs = len(self.tokens) // seq_len
        self.tokens = self.tokens[:n_seqs * seq_len]
        print(f"    ✓ Total: {len(self.tokens):,} tokens, {n_seqs:,} secuencias")
    
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = self.tokens[start:start + self.seq_len]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# =============================================================================
# ENTRENADOR
# =============================================================================

class Trainer:
    """Entrenador simple y eficiente"""
    
    def __init__(self, config: TrainConfig, model_config: ConfigPampaR):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear directorio de checkpoints
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Modelo
        print("\n🧠 Inicializando modelo...")
        self.model = PampaR(model_config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"    Parámetros: {n_params:,}")
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Estado
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def load_checkpoint(self, path: str) -> bool:
        """Carga un checkpoint si existe"""
        if not os.path.exists(path):
            return False
        
        print(f"📂 Cargando checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Cargar modelo (soporta diferentes formatos)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict'))
        if state_dict:
            self.model.load_state_dict(state_dict)
        
        # Cargar optimizador si existe
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.global_step = checkpoint.get('step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"    ✓ Step: {self.global_step}, Best Loss: {self.best_val_loss:.4f}")
        return True
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Guarda checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.model_config.__dict__,
        }
        torch.save(checkpoint, path)
        if is_best:
            print(f"    ✅ Nuevo mejor modelo guardado!")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int):
        """Entrena una época"""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward con mixed precision
            with autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                output = self.model(x)
                logits = output['logits'] if isinstance(output, dict) else output
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1)
                )
                loss = loss / self.config.gradient_accumulation
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation
            
            # Actualizar pesos
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log cada 100 steps
                if self.global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (batch_idx + 1) * self.config.batch_size * (x.size(1)) / elapsed
                    vram = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                    avg_loss = total_loss / (batch_idx + 1)
                    
                    print(f"  Step {self.global_step:,} | Loss: {avg_loss:.4f} | "
                          f"{tokens_per_sec/1000:.1f}K tok/s | VRAM: {vram:.0f}MB")
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evalúa el modelo"""
        self.model.eval()
        total_loss = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                output = self.model(x)
                logits = output['logits'] if isinstance(output, dict) else output
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1)
                )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity
    
    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """Loop principal de entrenamiento"""
        print("\n" + "="*60)
        print("🏋️ INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            print(f"\n📖 ÉPOCA {epoch}/{epochs}")
            print("-" * 40)
            
            # Entrenar
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            
            # Evaluar
            val_loss, val_ppl = self.evaluate(val_loader)
            print(f"\n  📊 Validación: Loss={val_loss:.4f} | PPL={val_ppl:.1f}")
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/pampar_best.pt",
                    is_best=True
                )
            
            # Guardar checkpoint de época
            self.save_checkpoint(f"{self.config.checkpoint_dir}/pampar_epoch_{epoch}.pt")
            
            # Limpiar memoria
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Guardar modelo final
        self.save_checkpoint(f"{self.config.checkpoint_dir}/pampar_final.pt")
        print("\n✅ Entrenamiento completado!")
        print(f"   Mejor Val Loss: {self.best_val_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def parse_tokens(value: str) -> int:
    """Parsea valores como '10M', '50M', '1B'"""
    value = value.upper().strip()
    multipliers = {'K': 1000, 'M': 1_000_000, 'B': 1_000_000_000}
    
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[:-1]) * mult)
    
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="PampaR Training")
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: según config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: según config)')
    parser.add_argument('--tokens', type=str, default=None, help='Límite de tokens (ej: 10M, 50M)')
    parser.add_argument('--resume', action='store_true', help='Continuar desde checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path a checkpoint específico')
    parser.add_argument('--max', action='store_true', help='Usar config MAX (más parámetros, más riesgo OOM)')
    parser.add_argument('--accum', type=int, default=8, help='Gradient accumulation steps')
    args = parser.parse_args()
    
    print("="*60)
    print("🦙 PampaR - Entrenamiento")
    print("="*60)
    
    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n📱 GPU: {gpu_name}")
        print(f"   VRAM: {vram:.1f} GB")
    else:
        print("\n⚠️ No se detectó GPU, usando CPU")
    
    # Configuración
    train_config = TrainConfig()
    
    # Seleccionar config del modelo
    if args.max:
        model_config = LOCAL_4GB_MAX
        print("\n⚡ Modo MAX activado - configuración agresiva")
    else:
        model_config = LOCAL_4GB
    
    # Usar valores del config si no se especifican
    train_config.batch_size = args.batch_size or model_config.batch_size
    train_config.learning_rate = args.lr or model_config.learning_rate
    train_config.gradient_accumulation = args.accum
    
    if args.tokens:
        train_config.max_tokens = parse_tokens(args.tokens)
    
    print(f"\n🎯 Configuración:")
    print(f"   Modelo: {'LOCAL_4GB_MAX' if args.max else 'LOCAL_4GB'}")
    print(f"   Dim: {model_config.dim}, Capas: {model_config.n_capas}, Heads: {model_config.n_heads}")
    print(f"   Épocas: {args.epochs}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Gradient accumulation: {train_config.gradient_accumulation}")
    print(f"   Effective batch: {train_config.batch_size * train_config.gradient_accumulation}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Max tokens: {train_config.max_tokens or 'Todo el corpus'}")
    
    # Cargar datos
    print("\n" + "="*60)
    print("📚 CARGANDO DATOS")
    print("="*60)
    
    train_dataset = TextDataset(
        train_config.corpus_path,
        train_config.tokenizer_path,
        seq_len=model_config.max_seq_len,
        max_tokens=train_config.max_tokens
    )
    
    val_dataset = TextDataset(
        train_config.val_path,
        train_config.tokenizer_path,
        seq_len=model_config.max_seq_len,
        max_tokens=500_000  # 500K para validación
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Crear entrenador
    trainer = Trainer(train_config, model_config)
    
    # Cargar checkpoint si existe
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint or f"{train_config.checkpoint_dir}/pampar_best.pt"
        trainer.load_checkpoint(checkpoint_path)
    
    # Entrenar
    trainer.train(args.epochs, train_loader, val_loader)


if __name__ == "__main__":
    main()
