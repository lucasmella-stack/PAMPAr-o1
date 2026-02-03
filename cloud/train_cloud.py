#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-o1 - Entrenamiento en Google Cloud Platform
==================================================

Script optimizado para entrenar en GPUs de la nube (T4, V100, A100).
Descarga datasets automáticamente y sube checkpoints a GCS.

Uso:
    python cloud/train_cloud.py --gpu t4 --hours 100
    python cloud/train_cloud.py --gpu v100 --hours 50 --resume
    python cloud/train_cloud.py --gpu a100 --scale large
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.config import ConfigPampaR, SERVER_8GB, SERVER_24GB, SERVER_80GB
from pampar.cerebro.model import PampaR

# ============================================================================
# CONFIGURACIONES POR GPU
# ============================================================================

GPU_CONFIGS = {
    "t4": {
        "config": ConfigPampaR(
            vocab_size=32000,
            dim=256,
            n_heads=8,
            n_capas=6,
            dropout=0.1,
            max_seq_len=512,
            peso_llaves=0.6,
            usar_axiomas=True,
            usar_memoria=True,
            capacidad_memoria=500,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            batch_size=32,
            learning_rate=2e-4,
            max_epochs=50,
        ),
        "effective_batch": 128,
        "tokens_per_hour": 15_000_000,
        "cost_per_hour": 0.35,
    },
    "v100": {
        "config": ConfigPampaR(
            vocab_size=32000,
            dim=384,
            n_heads=8,
            n_capas=8,
            dropout=0.1,
            max_seq_len=512,
            peso_llaves=0.5,
            usar_axiomas=True,
            usar_memoria=True,
            capacidad_memoria=1000,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            batch_size=48,
            learning_rate=1.5e-4,
            max_epochs=50,
        ),
        "effective_batch": 192,
        "tokens_per_hour": 40_000_000,
        "cost_per_hour": 2.48,
    },
    "l4": {
        "config": ConfigPampaR(
            vocab_size=32000,
            dim=384,
            n_heads=8,
            n_capas=8,
            dropout=0.1,
            max_seq_len=512,
            peso_llaves=0.5,
            usar_axiomas=True,
            usar_memoria=True,
            capacidad_memoria=1000,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            batch_size=48,
            learning_rate=1.5e-4,
            max_epochs=50,
        ),
        "effective_batch": 192,
        "tokens_per_hour": 50_000_000,
        "cost_per_hour": 0.81,
    },
    "a100": {
        "config": ConfigPampaR(
            vocab_size=50000,
            dim=512,
            n_heads=8,
            n_capas=10,
            dropout=0.1,
            max_seq_len=1024,
            peso_llaves=0.4,
            usar_axiomas=True,
            usar_memoria=True,
            capacidad_memoria=2000,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            batch_size=64,
            learning_rate=1e-4,
            max_epochs=100,
        ),
        "effective_batch": 256,
        "tokens_per_hour": 100_000_000,
        "cost_per_hour": 3.67,
    },
}

# ============================================================================
# DATASET CON MÚLTIPLES FUENTES
# ============================================================================

class MultiSourceDataset(Dataset):
    """Dataset que combina múltiples fuentes de texto."""
    
    def __init__(self, tokenizer_path: str, seq_len: int = 512, 
                 max_tokens: int = None, include_spanish: bool = True):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        
        self.tokens = []
        
        # 1. WikiText-103 (inglés)
        wiki_path = Path("data/corpus/wikitext-103-raw/wiki.train.raw")
        if wiki_path.exists():
            print(f"📖 Cargando WikiText-103...")
            self._load_file(wiki_path, max_tokens)
        
        # 2. OpenWebText (si está disponible)
        owt_path = Path("data/corpus/openwebtext")
        if owt_path.exists():
            print(f"📖 Cargando OpenWebText...")
            for f in sorted(owt_path.glob("*.txt"))[:100]:  # Primeros 100 archivos
                if max_tokens and len(self.tokens) >= max_tokens:
                    break
                self._load_file(f, max_tokens)
        
        # 3. Spanish corpus (si está habilitado)
        if include_spanish:
            spanish_path = Path("data/corpus/spanish")
            if spanish_path.exists():
                print(f"📖 Cargando corpus español...")
                for f in spanish_path.glob("*.txt"):
                    if max_tokens and len(self.tokens) >= max_tokens:
                        break
                    self._load_file(f, max_tokens)
        
        # Crear secuencias
        if max_tokens:
            self.tokens = self.tokens[:max_tokens]
        
        n_seqs = len(self.tokens) // seq_len
        self.tokens = self.tokens[:n_seqs * seq_len]
        print(f"✅ Total: {len(self.tokens):,} tokens, {n_seqs:,} secuencias")
    
    def _load_file(self, path: Path, max_tokens: int = None):
        """Cargar un archivo de texto."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 30:
                        ids = self.sp.Encode(line)
                        self.tokens.extend(ids)
                        
                        if max_tokens and len(self.tokens) >= max_tokens:
                            return
        except Exception as e:
            print(f"⚠️ Error cargando {path}: {e}")
    
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = self.tokens[start:start + self.seq_len]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ============================================================================
# DESCARGA DE DATOS
# ============================================================================

def download_datasets(include_spanish: bool = True):
    """Descargar datasets necesarios."""
    import urllib.request
    import zipfile
    import tarfile
    
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. WikiText-103
    wiki_dir = corpus_dir / "wikitext-103-raw"
    if not wiki_dir.exists():
        print("📥 Descargando WikiText-103...")
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        zip_path = corpus_dir / "wikitext.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(corpus_dir)
        zip_path.unlink()
        print("✅ WikiText-103 descargado")
    
    # 2. Spanish corpus (textos de dominio público)
    if include_spanish:
        spanish_dir = corpus_dir / "spanish"
        if not spanish_dir.exists():
            spanish_dir.mkdir(parents=True)
            print("📥 Descargando corpus español...")
            
            spanish_texts = [
                ("https://www.gutenberg.org/cache/epub/2000/pg2000.txt", "quijote.txt"),
                ("https://www.gutenberg.org/cache/epub/17073/pg17073.txt", "regenta.txt"),
                ("https://www.gutenberg.org/cache/epub/17656/pg17656.txt", "fortunata.txt"),
                ("https://www.gutenberg.org/cache/epub/15530/pg15530.txt", "pepita.txt"),
                ("https://www.gutenberg.org/cache/epub/15781/pg15781.txt", "sombrero.txt"),
                ("https://www.gutenberg.org/cache/epub/16212/pg16212.txt", "alhambra.txt"),
            ]
            
            for url, filename in spanish_texts:
                try:
                    dest = spanish_dir / filename
                    urllib.request.urlretrieve(url, dest)
                    print(f"  ✅ {filename}")
                except Exception as e:
                    print(f"  ⚠️ Error descargando {filename}: {e}")
            
            print("✅ Corpus español descargado")
    
    # 3. Intentar descargar OpenWebText subset (HuggingFace)
    try:
        from datasets import load_dataset
        owt_dir = corpus_dir / "openwebtext"
        if not owt_dir.exists():
            print("📥 Descargando OpenWebText subset...")
            owt_dir.mkdir(parents=True)
            
            # Cargar subset pequeño
            ds = load_dataset("openwebtext", split="train", streaming=True)
            
            with open(owt_dir / "owt_subset.txt", 'w', encoding='utf-8') as f:
                for i, example in enumerate(ds):
                    if i >= 500000:  # ~500K documentos
                        break
                    f.write(example['text'] + '\n')
                    if (i + 1) % 50000 == 0:
                        print(f"  {i+1:,} documentos...")
            
            print("✅ OpenWebText descargado")
    except Exception as e:
        print(f"⚠️ OpenWebText no disponible: {e}")


# ============================================================================
# GOOGLE CLOUD STORAGE
# ============================================================================

def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str):
    """Subir archivo a Google Cloud Storage."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"☁️ Subido: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"⚠️ Error subiendo a GCS: {e}")


def download_from_gcs(bucket_name: str, blob_name: str, local_path: str):
    """Descargar archivo de Google Cloud Storage."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"☁️ Descargado: gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        print(f"⚠️ Error descargando de GCS: {e}")
        return False


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

class CloudTrainer:
    """Entrenador optimizado para la nube."""
    
    def __init__(self, gpu_type: str, budget_hours: float, 
                 gcs_bucket: str = None, resume: bool = False,
                 include_spanish: bool = True):
        
        self.gpu_type = gpu_type
        self.budget_hours = budget_hours
        self.gcs_bucket = gcs_bucket
        self.include_spanish = include_spanish
        
        # Obtener configuración para la GPU
        gpu_cfg = GPU_CONFIGS[gpu_type]
        self.config = gpu_cfg["config"]
        self.effective_batch = gpu_cfg["effective_batch"]
        self.tokens_per_hour = gpu_cfg["tokens_per_hour"]
        self.cost_per_hour = gpu_cfg["cost_per_hour"]
        
        # Calcular tokens objetivo
        self.target_tokens = int(budget_hours * self.tokens_per_hour)
        self.estimated_cost = budget_hours * self.cost_per_hour
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Directorios
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🧠 PAMPAr-o1 - Cloud Training")
        print(f"{'='*60}")
        print(f"🖥️  GPU: {gpu_type.upper()}")
        print(f"⏰ Horas presupuestadas: {budget_hours}")
        print(f"💰 Costo estimado: ${self.estimated_cost:.2f}")
        print(f"🎯 Tokens objetivo: {self.target_tokens:,}")
        print(f"📊 Configuración: dim={self.config.dim}, capas={self.config.n_capas}")
        print(f"🌍 Español incluido: {include_spanish}")
        
        # Descargar datos
        download_datasets(include_spanish)
        
        # Cargar tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load("data/tokenizer/llarri_bpe.model")
        self.config.vocab_size = self.tokenizer.GetPieceSize()
        
        # Crear modelo
        print(f"\n🔧 Inicializando modelo...")
        self.model = PampaR(self.config).to(self.device)
        self.model.registrar_tokenizer(self.tokenizer)
        
        # Cargar checkpoint si existe
        if resume:
            self._load_checkpoint()
        
        # Stats
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Parámetros: {n_params:,}")
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        self.use_amp = self.config.use_mixed_precision and self.device.type == 'cuda'
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
        
        # Progress tracking
        self.tokens_seen = 0
        self.best_loss = float('inf')
        self.start_time = None
    
    def _load_checkpoint(self):
        """Cargar checkpoint local o de GCS."""
        ckpt_path = self.checkpoint_dir / "pampar_cloud_best.pt"
        
        # Intentar descargar de GCS
        if self.gcs_bucket and not ckpt_path.exists():
            download_from_gcs(self.gcs_bucket, "checkpoints/pampar_cloud_best.pt", str(ckpt_path))
        
        if ckpt_path.exists():
            print(f"📂 Cargando checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.tokens_seen = checkpoint.get('tokens_seen', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"✅ Checkpoint cargado. Tokens vistos: {self.tokens_seen:,}")
    
    def _save_checkpoint(self, loss: float, is_best: bool = False):
        """Guardar checkpoint local y a GCS."""
        checkpoint = {
            'model': self.model.state_dict(),
            'config': self.config.__dict__,
            'tokens_seen': self.tokens_seen,
            'best_loss': self.best_loss,
            'gpu_type': self.gpu_type,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Guardar localmente
        ckpt_path = self.checkpoint_dir / "pampar_cloud_latest.pt"
        torch.save(checkpoint, ckpt_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "pampar_cloud_best.pt"
            torch.save(checkpoint, best_path)
            
            # Subir a GCS
            if self.gcs_bucket:
                upload_to_gcs(str(best_path), self.gcs_bucket, "checkpoints/pampar_cloud_best.pt")
    
    def train(self):
        """Ejecutar entrenamiento."""
        print(f"\n{'='*60}")
        print("🚀 Iniciando entrenamiento...")
        print(f"{'='*60}\n")
        
        # Crear dataset
        dataset = MultiSourceDataset(
            tokenizer_path="data/tokenizer/llarri_bpe.model",
            seq_len=self.config.max_seq_len,
            max_tokens=self.target_tokens,
            include_spanish=self.include_spanish
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Gradient accumulation
        accum_steps = self.effective_batch // self.config.batch_size
        
        self.start_time = time.time()
        epoch = 0
        
        while self.tokens_seen < self.target_tokens:
            epoch += 1
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            
            self.optimizer.zero_grad()
            
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(input_ids)
                        logits = outputs['logits'].view(-1, self.config.vocab_size)
                        loss = self.criterion(logits, targets.view(-1))
                        loss = loss / accum_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(input_ids)
                    logits = outputs['logits'].view(-1, self.config.vocab_size)
                    loss = self.criterion(logits, targets.view(-1))
                    loss = loss / accum_steps
                    loss.backward()
                
                epoch_loss += loss.item() * accum_steps
                batch_tokens = input_ids.numel()
                epoch_tokens += batch_tokens
                self.tokens_seen += batch_tokens
                
                # Optimizer step
                if (batch_idx + 1) % accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Logging
                    if (batch_idx + 1) % (accum_steps * 10) == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        ppl = torch.exp(torch.tensor(avg_loss)).item()
                        elapsed = time.time() - self.start_time
                        tokens_per_sec = self.tokens_seen / elapsed
                        eta_hours = (self.target_tokens - self.tokens_seen) / tokens_per_sec / 3600
                        
                        print(f"[Epoch {epoch}] "
                              f"Loss: {avg_loss:.4f} | "
                              f"PPL: {ppl:.1f} | "
                              f"Tokens: {self.tokens_seen:,}/{self.target_tokens:,} | "
                              f"ETA: {eta_hours:.1f}h")
                
                # Checkpoint cada 10M tokens
                if self.tokens_seen % 10_000_000 < batch_tokens:
                    avg_loss = epoch_loss / max(batch_idx, 1)
                    is_best = avg_loss < self.best_loss
                    if is_best:
                        self.best_loss = avg_loss
                    self._save_checkpoint(avg_loss, is_best)
                
                # Verificar si alcanzamos el objetivo
                if self.tokens_seen >= self.target_tokens:
                    break
            
            # Fin de epoch
            avg_loss = epoch_loss / len(dataloader)
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f"\n📊 Epoch {epoch} completado: Loss={avg_loss:.4f}, PPL={ppl:.1f}")
            
            # Guardar checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            self._save_checkpoint(avg_loss, is_best)
        
        # Entrenamiento completado
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"✅ Entrenamiento completado!")
        print(f"{'='*60}")
        print(f"⏰ Tiempo total: {total_time/3600:.2f} horas")
        print(f"📊 Tokens procesados: {self.tokens_seen:,}")
        print(f"🏆 Mejor loss: {self.best_loss:.4f}")
        print(f"💰 Costo estimado: ${total_time/3600 * self.cost_per_hour:.2f}")
        
        # Guardar checkpoint final
        self._save_checkpoint(self.best_loss, True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PAMPAr-o1 Cloud Training")
    parser.add_argument("--gpu", type=str, default="t4", 
                        choices=["t4", "v100", "l4", "a100"],
                        help="Tipo de GPU")
    parser.add_argument("--hours", type=float, default=100,
                        help="Horas de entrenamiento presupuestadas")
    parser.add_argument("--bucket", type=str, default=None,
                        help="Bucket de GCS para checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Continuar desde checkpoint")
    parser.add_argument("--no-spanish", action="store_true",
                        help="No incluir corpus español")
    
    args = parser.parse_args()
    
    trainer = CloudTrainer(
        gpu_type=args.gpu,
        budget_hours=args.hours,
        gcs_bucket=args.bucket,
        resume=args.resume,
        include_spanish=not args.no_spanish
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
