# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Entrenamiento del LLARRI Language Model.

Diseñado para entrenar el modelo de lenguaje fractal con:
- Entrenamiento por cuadrantes (niveles fractales)
- Dataset de texto (TinyStories, WikiText, o custom)
- Optimizado para GPU limitada (4GB VRAM)

Usage:
    python scripts/train_language_model.py --dataset tinystories --epochs 10
    python scripts/train_language_model.py --dataset wikitext --batch_size 4
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1 import LLARRILanguageModel, LLARRIConfig


def get_device_info():
    """Obtiene información del dispositivo."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, f"{gpu_name} ({vram:.1f} GB)"
    return torch.device('cpu'), "CPU"


def clear_memory():
    """Limpia memoria GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# DATASETS
# ============================================================================

class TextDataset(Dataset):
    """Dataset genérico de texto para entrenamiento."""
    
    def __init__(self, texts: list, seq_len: int = 128):
        self.texts = texts
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convertir a bytes y truncar/pad
        bytes_data = list(text.encode('utf-8', errors='ignore'))[:self.seq_len]
        
        # Pad si es necesario
        if len(bytes_data) < self.seq_len:
            bytes_data = bytes_data + [0] * (self.seq_len - len(bytes_data))
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        # Labels = siguiente token
        return tokens[:-1], tokens[1:]


class TinyStoriesDataset(Dataset):
    """Dataset TinyStories de HuggingFace."""
    
    def __init__(self, split='train', seq_len=128, max_samples=50000):
        try:
            from datasets import load_dataset
            print(f"Cargando TinyStories ({split})...")
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            
            self.texts = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                self.texts.append(item['text'])
                if (i + 1) % 10000 == 0:
                    print(f"  Cargados {i+1} textos...")
            
            print(f"  Total: {len(self.texts)} textos")
        except ImportError:
            print("ERROR: Necesitas instalar 'datasets': pip install datasets")
            raise
        except Exception as e:
            print(f"ERROR cargando TinyStories: {e}")
            # Fallback a datos sintéticos
            print("Usando datos sintéticos de prueba...")
            self.texts = self._generate_synthetic_data(max_samples)
        
        self.seq_len = seq_len
    
    def _generate_synthetic_data(self, n=10000):
        """Genera datos sintéticos si no se puede cargar el dataset."""
        stories = [
            "Once upon a time there was a little girl named Lucy.",
            "The cat sat on the mat and looked at the bird.",
            "Tom went to the park to play with his friends.",
            "The sun was shining bright in the blue sky.",
            "A dog ran across the field chasing a ball.",
            "Mom made cookies and they smelled delicious.",
            "The little boy found a magic stone in the garden.",
            "Birds were singing in the trees early morning.",
            "She opened the book and started reading.",
            "The train went through the long dark tunnel.",
        ]
        import random
        return [random.choice(stories) for _ in range(n)]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        bytes_data = list(text.encode('utf-8', errors='ignore'))[:self.seq_len]
        
        if len(bytes_data) < self.seq_len:
            bytes_data = bytes_data + [0] * (self.seq_len - len(bytes_data))
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        return tokens[:-1], tokens[1:]


class WikiTextDataset(Dataset):
    """Dataset WikiText-2."""
    
    def __init__(self, split='train', seq_len=128, max_samples=30000):
        try:
            from datasets import load_dataset
            print(f"Cargando WikiText-2 ({split})...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            
            self.texts = []
            for item in ds:
                text = item['text'].strip()
                if len(text) > 20:  # Ignorar líneas muy cortas
                    self.texts.append(text)
                if len(self.texts) >= max_samples:
                    break
            
            print(f"  Total: {len(self.texts)} textos")
        except ImportError:
            print("ERROR: Necesitas instalar 'datasets': pip install datasets")
            raise
        
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        bytes_data = list(text.encode('utf-8', errors='ignore'))[:self.seq_len]
        
        if len(bytes_data) < self.seq_len:
            bytes_data = bytes_data + [0] * (self.seq_len - len(bytes_data))
        
        tokens = torch.tensor(bytes_data, dtype=torch.long)
        return tokens[:-1], tokens[1:]


# ============================================================================
# TRAINER POR CUADRANTES
# ============================================================================

class LanguageModelTrainer:
    """
    Entrenador para LLARRI Language Model.
    
    Soporta entrenamiento por cuadrantes (niveles fractales):
    - Cuadrante 1: niveles [2, 4] - tokens básicos
    - Cuadrante 2: niveles [8, 16] - tokens medios  
    - Cuadrante 3: niveles [32, 64] - tokens largos
    - Cuadrante 4: niveles [128, 256] - tokens muy largos
    
    Los parámetros se COMPARTEN entre todos los niveles,
    por lo que entrenar un cuadrante actualiza el modelo completo.
    """
    
    def __init__(
        self,
        model: LLARRILanguageModel,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        batch_size: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        gradient_accumulation: int = 4,
    ):
        self.device, device_info = get_device_info()
        print(f"\n🖥️  Dispositivo: {device_info}")
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation = gradient_accumulation
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Evitar problemas de memoria
            pin_memory=True if torch.cuda.is_available() else False
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
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * 10  # ~10 epochs
        warmup_steps = total_steps // 10
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # AMP
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': [],
            'lr': [],
        }
        
        # Info
        self._print_config()
    
    def _print_config(self):
        """Imprime configuración."""
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           LLARRI LANGUAGE MODEL TRAINER                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Parámetros:     {n_params:>12,} ({n_params/1e6:.2f}M)                    ║
║  Entrenables:    {n_trainable:>12,}                                 ║
║  Batch Size:     {self.batch_size:>12} (×{self.gradient_accumulation} acum = {self.batch_size * self.gradient_accumulation})             ║
║  AMP:            {'Sí':>12}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Niveles Fractales: {self.model.config.niveles}          ║
║  Arquitectura: 6 Cajas (Mezcla → Procesa ×3 → Evalúa → Output)   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    def train_step(self, batch):
        """Un paso de entrenamiento."""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        if self.use_amp:
            with autocast('cuda'):
                output = self.model(input_ids=input_ids, labels=labels)
                loss = output['loss'] / self.gradient_accumulation
            self.scaler.scale(loss).backward()
        else:
            output = self.model(input_ids=input_ids, labels=labels)
            loss = output['loss'] / self.gradient_accumulation
            loss.backward()
        
        return loss.item() * self.gradient_accumulation
    
    def train_epoch(self, epoch: int):
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
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
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  Batch {batch_idx+1:4d}/{len(self.train_loader)} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e} | Mem: {mem:.1f}GB")
            
            # Limpiar memoria periódicamente
            if batch_idx % 100 == 0:
                clear_memory()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self):
        """Evalúa en validation set."""
        if not self.val_loader:
            return None, None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    output = self.model(input_ids=input_ids, labels=labels)
            else:
                output = self.model(input_ids=input_ids, labels=labels)
            
            total_loss += output['loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch: int, loss: float, filename: str = None):
        """Guarda checkpoint."""
        filename = filename or f"lm_checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.model.config,
            'history': self.history,
        }, path)
        print(f"  💾 Checkpoint: {path}")
    
    def train(self, epochs: int = 10, save_every: int = 2):
        """Entrena el modelo."""
        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO ENTRENAMIENTO - {epochs} épocas")
        print(f"{'='*60}\n")
        
        start_total = time.time()
        
        for epoch in range(1, epochs + 1):
            start_epoch = time.time()
            
            print(f"\n📘 Época {epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, perplexity = self.evaluate()
            
            elapsed = time.time() - start_epoch
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(self.scheduler.get_last_lr()[0])
            
            if val_loss:
                self.history['val_loss'].append(val_loss)
                self.history['val_perplexity'].append(perplexity)
                print(f"\n  ✓ Train Loss: {train_loss:.4f}")
                print(f"  ✓ Val Loss:   {val_loss:.4f}")
                print(f"  ✓ Perplexity: {perplexity:.2f}")
                
                # Save best
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, "best_lm_model.pt")
            else:
                print(f"\n  ✓ Train Loss: {train_loss:.4f}")
                
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint(epoch, train_loss, "best_lm_model.pt")
            
            print(f"  ⏱️  Tiempo: {elapsed:.1f}s")
            
            # Checkpoint periódico
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # Limpiar memoria
            clear_memory()
        
        total_time = time.time() - start_total
        
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Tiempo total: {total_time/60:.1f} minutos")
        print(f"   Mejor loss: {self.best_loss:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time", max_tokens: int = 50):
        """Genera texto de muestra."""
        self.model.eval()
        output = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperatura=0.8,
            top_k=40
        )
        return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenar LLARRI Language Model")
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "wikitext", "synthetic"],
                        help="Dataset a usar")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Longitud de secuencia")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=30000, help="Max muestras")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--checkpoint", type=str, default=None, help="Cargar checkpoint")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠 LLARRI-O1 Language Model Training                          ║
║                                                                  ║
║   "Mezcla primero, procesa con vecinos — de pequeño a grande"   ║
║                                                                  ║
║   Author: Lucas Ricardo Mella Chillemi                           ║
║   Organization: Segunda Cabeza                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Configuración del modelo (optimizada para GTX 1650 4GB)
    config = LLARRIConfig(
        embed_dim=args.embed_dim,
        base_dim=args.embed_dim,
        max_dim=args.embed_dim * 2,
        niveles=[2, 4, 8, 16],  # 4 niveles para GPU limitada
        num_heads=4,
        ffn_expansion=2.0,
        num_vecinos=3,
        umbral_confianza=0.7,
        dropout=0.1,
        max_length=args.seq_len + 1
    )
    
    print(f"📊 Configuración del modelo:")
    print(f"   embed_dim: {config.embed_dim}")
    print(f"   niveles: {config.niveles}")
    print(f"   max_length: {config.max_length}")
    
    # Crear modelo
    print("\n📦 Creando modelo...")
    model = LLARRILanguageModel(config)
    
    # Cargar checkpoint si existe
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"📂 Cargando checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # Dataset
    print(f"\n📚 Cargando dataset: {args.dataset}")
    if args.dataset == "tinystories":
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
    elif args.dataset == "wikitext":
        train_dataset = WikiTextDataset(
            split='train',
            seq_len=args.seq_len,
            max_samples=args.max_samples
        )
        val_dataset = WikiTextDataset(
            split='validation',
            seq_len=args.seq_len,
            max_samples=args.max_samples // 10
        )
    else:
        # Synthetic para testing
        from torch.utils.data import random_split
        full_dataset = TinyStoriesDataset(
            split='train',
            seq_len=args.seq_len,
            max_samples=1000
        )
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Trainer
    trainer = LanguageModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        gradient_accumulation=args.grad_accum,
        use_amp=True
    )
    
    # Entrenar
    history = trainer.train(epochs=args.epochs, save_every=2)
    
    # Generar muestra
    print("\n📝 Generando texto de muestra...")
    sample = trainer.generate_sample("Once upon a time", max_tokens=100)
    print(f"\n{sample}")
    
    print("\n✅ ¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
