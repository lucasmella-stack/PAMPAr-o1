# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Script de entrenamiento para LLARRI v8

Características:
- Usa la nueva arquitectura cerebral modular
- Entrena con WikiText-103 (o subset)
- Monitorea activación de módulos
- Guarda checkpoints y estadísticas
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sentencepiece as spm

# Añadir el path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.cerebro import LLARRIv8, ConfigLLARRI


class WikiTextDataset(Dataset):
    """Dataset para WikiText tokenizado."""
    
    def __init__(self, data_path: str, tokenizer_path: str, seq_len: int = 256):
        self.seq_len = seq_len
        
        # Cargar tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Cargar y tokenizar texto
        print(f"Cargando datos de {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Limpiar
        lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('=')]
        text_clean = ' '.join(lines)
        
        print("Tokenizando...")
        self.tokens = self.tokenizer.Encode(text_clean)
        print(f"Total tokens: {len(self.tokens):,}")
        
        # Calcular número de ejemplos
        self.n_ejemplos = (len(self.tokens) - 1) // seq_len
    
    def __len__(self):
        return self.n_ejemplos
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        chunk = self.tokens[start:end]
        
        # Pad si es necesario
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class Entrenador:
    """Entrenador para LLARRI v8."""
    
    def __init__(
        self,
        model: LLARRIv8,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        epochs: int = 10,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_interval = log_interval
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1,
        )
        
        # Estadísticas
        self.stats = {
            'train_loss': [],
            'val_loss': [],
            'val_ppl': [],
            'modulos': [],
            'lr': [],
        }
        
        self.best_val_loss = float('inf')
        self.step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(x, labels=y, return_info=(batch_idx % self.log_interval == 0))
            loss = outputs['loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.step += 1
            
            # Log
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                lr = self.scheduler.get_last_lr()[0]
                
                # Stats de módulos
                stats_modulos = self.model.obtener_estadisticas_modulos()
                
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"LR: {lr:.2e}")
                
                # Mostrar activación de módulos
                modulos_str = " | ".join([f"{k[:3]}:{v*100:.1f}%" for k, v in stats_modulos.items()])
                print(f"    Módulos: {modulos_str}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        
        print(f"\n  Época {epoch} completada en {epoch_time:.1f}s | Loss promedio: {avg_loss:.4f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Valida el modelo."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            outputs = self.model(x, labels=y)
            total_loss += outputs['loss'].item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        ppl = math.exp(min(avg_loss, 20))  # Clamp para evitar overflow
        
        return {'loss': avg_loss, 'perplexity': ppl}
    
    @torch.no_grad()
    def generar_muestra(self, tokenizer, prompt: str = "El", max_tokens: int = 50):
        """Genera texto de muestra."""
        self.model.eval()
        
        # Tokenizar prompt
        input_ids = tokenizer.Encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Generar
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        
        # Decodificar
        texto = tokenizer.Decode(output_ids[0].tolist())
        return texto
    
    def guardar_checkpoint(self, nombre: str, extra_info: Dict = None):
        """Guarda checkpoint."""
        path = self.checkpoint_dir / f"{nombre}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'stats': self.stats,
            'config': self.model.config.__dict__,
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint guardado: {path}")
    
    def entrenar(self, tokenizer=None):
        """Bucle principal de entrenamiento."""
        print("\n" + "="*60)
        print("ENTRENAMIENTO LLARRI v8")
        print("="*60)
        
        # Info del modelo
        params = self.model.contar_parametros()
        print(f"\nParámetros totales: {params['total']:,}")
        for k, v in params.items():
            if k != 'total':
                print(f"  {k}: {v:,}")
        
        print(f"\nEpochs: {self.epochs}")
        print(f"Batches por epoch: {len(self.train_loader)}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        for epoch in range(1, self.epochs + 1):
            print(f"\n{'='*20} ÉPOCA {epoch}/{self.epochs} {'='*20}")
            
            # Entrenar
            train_loss = self.train_epoch(epoch)
            self.stats['train_loss'].append(train_loss)
            
            # Validar
            val_metrics = self.validate()
            if val_metrics:
                self.stats['val_loss'].append(val_metrics['loss'])
                self.stats['val_ppl'].append(val_metrics['perplexity'])
                print(f"\n  Validación | Loss: {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")
                
                # Guardar mejor modelo
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.guardar_checkpoint('llarri_v8_best', {'val_loss': val_metrics['loss']})
            
            # Generar muestra
            if tokenizer:
                print("\n  Muestra de generación:")
                for prompt in ["El", "La ciencia", "En el año"]:
                    texto = self.generar_muestra(tokenizer, prompt)
                    print(f"    '{prompt}' → {texto[:100]}...")
            
            # Stats de módulos
            stats_modulos = self.model.obtener_estadisticas_modulos()
            self.stats['modulos'].append(stats_modulos)
            
            # Checkpoint periódico
            if epoch % 3 == 0:
                self.guardar_checkpoint(f'llarri_v8_epoch_{epoch}')
            
            # Reset stats de módulos para siguiente época
            self.model.reset_estadisticas()
        
        # Guardar modelo final
        self.guardar_checkpoint('llarri_v8_final')
        
        # Guardar estadísticas
        stats_path = self.checkpoint_dir / 'training_stats_v8.json'
        with open(stats_path, 'w') as f:
            # Convertir a serializable
            stats_ser = {k: v if not isinstance(v, list) or not v or not isinstance(v[0], dict) else v 
                        for k, v in self.stats.items()}
            json.dump(stats_ser, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*60)


def main():
    # Configuración
    TOKENIZER_PATH = "data/tokenizer/llarri_bpe.model"
    TRAIN_DATA = "data/wikitext-103/wikitext-103-raw/wiki.train.raw"
    VAL_DATA = "data/wikitext-103/wikitext-103-raw/wiki.valid.raw"
    
    # Verificar archivos
    for path in [TOKENIZER_PATH, TRAIN_DATA]:
        if not Path(path).exists():
            print(f"ERROR: No se encuentra {path}")
            return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Cargar tokenizer para vocab_size
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_size = sp.GetPieceSize()
    print(f"Vocab size: {vocab_size}")
    
    # Configuración del modelo (LIVIANA para 4GB VRAM GTX 1650)
    config = ConfigLLARRI(
        vocab_size=vocab_size,
        dim=128,          # Reducido para VRAM
        n_heads=4,
        n_capas=3,        # Reducido para VRAM
        dropout=0.1,
        peso_llaves=0.7,
        usar_axiomas=True,
        usar_memoria=True,
        capacidad_memoria=200,
        max_seq_len=256,
        repetition_penalty=1.2,
    )
    
    # Crear modelo
    model = LLARRIv8(config)
    model.registrar_tokenizer(sp)
    
    # Datasets
    seq_len = 256
    batch_size = 16  # Conservador para VRAM
    
    train_dataset = WikiTextDataset(TRAIN_DATA, TOKENIZER_PATH, seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = None
    if Path(VAL_DATA).exists():
        val_dataset = WikiTextDataset(VAL_DATA, TOKENIZER_PATH, seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    
    # Entrenador
    entrenador = Entrenador(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        epochs=10,
        device=device,
        checkpoint_dir='checkpoints',
        log_interval=100,
    )
    
    # Entrenar
    entrenador.entrenar(tokenizer=sp)


if __name__ == '__main__':
    main()
