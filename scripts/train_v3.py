#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Segunda Cabeza
# Author: Lucas Ricardo Mella Chillemi (lucas@segundacabeza.com)
"""
Entrenamiento LLARRI v3 Multiescala.

Entrena el modelo con procesamiento simultáneo en múltiples niveles:
- Nivel 2: caracteres (256 tokens)
- Nivel 4: bigramas (64 tokens)
- Nivel 8: cuadrantes (16 tokens)  
- Nivel 16: contexto (4 tokens)

Los embeddings son COMPARTIDOS entre niveles para economía.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Agregar path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v3 import LLARRIv3, LLARRIv3Config
from llarri_o1.utils.device import get_device_info


def clear_memory():
    """Limpia memoria GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TinyStoriesDataset(Dataset):
    """Dataset TinyStories para entrenamiento."""
    
    def __init__(self, split='train', seq_len=128, max_samples=50000):
        try:
            from datasets import load_dataset
            print(f"Cargando TinyStories ({split})...")
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            
            self.texts = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                text = item.get('text', '')
                if len(text) > 20:
                    self.texts.append(text)
                if (i + 1) % 10000 == 0:
                    print(f"  Cargados {len(self.texts)} textos...")
            
            print(f"  Total: {len(self.texts)} textos")
        except Exception as e:
            print(f"Error cargando TinyStories: {e}")
            self.texts = self._synthetic_data(max_samples)
        
        self.seq_len = seq_len
    
    def _synthetic_data(self, n=1000):
        """Datos sintéticos de respaldo."""
        stories = [
            "Once upon a time there was a little girl named Lucy who loved to play.",
            "The cat sat on the mat and watched the birds fly by the window.",
            "Tom went to the park with his friends to play soccer and have fun.",
        ]
        import random
        return [random.choice(stories) for _ in range(n)]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convertir a bytes
        bytes_data = list(text.encode('utf-8', errors='ignore'))[:self.seq_len]
        
        # Padding
        if len(bytes_data) < self.seq_len:
            bytes_data = bytes_data + [0] * (self.seq_len - len(bytes_data))
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        return tokens[:-1], tokens[1:]  # input, target


class TrainerV3:
    """Trainer para LLARRI v3 Multiescala."""
    
    def __init__(
        self,
        model: LLARRIv3,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        batch_size: int = 16,
        lr: float = 3e-4,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        grad_accum: int = 2
    ):
        self.device, device_info = get_device_info()
        print(f"\n🖥️  Dispositivo: {device_info}")
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum = grad_accum
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Optimizador
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * 10
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Stats
        self.best_loss = float('inf')
        
        # Info
        params = sum(p.numel() for p in model.parameters())
        print(f"\n╔{'═'*60}╗")
        print(f"║{'LLARRI v3 MULTIESCALA TRAINER':^60}║")
        print(f"╠{'═'*60}╣")
        print(f"║  Parámetros: {params:>15,} ({params/1e6:.2f}M){'':>17}║")
        print(f"║  Batch Size: {batch_size:>15} (×{grad_accum} = {batch_size*grad_accum}){'':>14}║")
        print(f"║  AMP: {'Sí' if self.use_amp else 'No':>21}{'':>32}║")
        print(f"║  Niveles: {str(model.config.niveles):>18}{'':>26}║")
        print(f"╚{'═'*60}╝")
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward con AMP
            if self.use_amp:
                with autocast():
                    output = self.model(inputs, labels=targets)
                    loss = output['loss'] / self.grad_accum
                
                self.scaler.scale(loss).backward()
            else:
                output = self.model(inputs, labels=targets)
                loss = output['loss'] / self.grad_accum
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum == 0:
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
            
            # Log cada 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1:5d}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            # Limpiar memoria cada 500 batches
            if (batch_idx + 1) % 500 == 0:
                clear_memory()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Valida el modelo."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(inputs, labels=targets)
            else:
                output = self.model(inputs, labels=targets)
            
            total_loss += output['loss'].item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch: int, val_loss: float, path: str = None):
        """Guarda checkpoint."""
        path = path or self.checkpoint_dir / "llarri_v3_best.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.model.config
        }, path)
        print(f"  💾 Guardado: {path}")
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time") -> str:
        """Genera texto de muestra."""
        self.model.eval()
        return self.model.generate(prompt, max_new_tokens=60, temperatura=0.8)
    
    def train(self, epochs: int = 10, save_every: int = 2):
        """Entrenamiento completo."""
        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO ENTRENAMIENTO - {epochs} épocas")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n📘 Época {epoch}/{epochs}")
            print("-" * 40)
            
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n  ✓ Train Loss: {train_loss:.4f}")
            print(f"  ✓ Val Loss:   {val_loss:.4f}")
            print(f"  ✓ Perplexity: {perplexity:.2f}")
            print(f"  ⏱️  Tiempo: {epoch_time:.1f}s")
            
            # Guardar mejor modelo
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # Checkpoint periódico
            if epoch % save_every == 0:
                path = self.checkpoint_dir / f"llarri_v3_epoch_{epoch}.pt"
                self.save_checkpoint(epoch, val_loss, path)
            
            clear_memory()
        
        total_time = (time.time() - start_time) / 60
        
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Tiempo total: {total_time:.1f} minutos")
        print(f"   Mejor loss: {self.best_loss:.4f}")
        print(f"{'='*60}")
        
        # Muestra de generación
        print("\n📝 Generando texto de muestra...")
        sample = self.generate_sample("Once upon a time")
        print(f"   {sample[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Entrenar LLARRI v3 Multiescala")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠 LLARRI v3 MULTIESCALA - Entrenamiento                      ║
║                                                                  ║
║   Procesamiento simultáneo en múltiples resoluciones            ║
║   con embeddings compartidos para máxima eficiencia             ║
║                                                                  ║
║   Author: Lucas Ricardo Mella Chillemi                           ║
║   Organization: Segunda Cabeza                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Config
    config = LLARRIv3Config(
        embed_dim=args.embed_dim,
        niveles=[2, 4, 8, 16],
        max_length=args.seq_len + 1,
        fusion_type="concat_project"
    )
    
    print(f"📊 Configuración:")
    print(f"   embed_dim: {config.embed_dim}")
    print(f"   niveles: {config.niveles}")
    print(f"   max_length: {config.max_length}")
    
    # Modelo
    print("\n📦 Creando modelo...")
    model = LLARRIv3(config)
    
    # Cargar checkpoint si existe
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n📂 Cargando checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
    
    # Datasets
    print(f"\n📚 Cargando dataset TinyStories...")
    train_dataset = TinyStoriesDataset(
        split='train',
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    
    val_dataset = TinyStoriesDataset(
        split='validation',
        seq_len=args.seq_len,
        max_samples=args.max_samples // 10
    )
    
    # Trainer
    trainer = TrainerV3(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum
    )
    
    # Entrenar
    trainer.train(epochs=args.epochs, save_every=2)


if __name__ == "__main__":
    main()
