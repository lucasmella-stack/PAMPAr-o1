#!/usr/bin/env python3
"""
LLARRI v8 - Entrenamiento COMPLETO
==================================
Entrena con TODO el corpus WikiText-103 (~103M tokens)
Optimizado para 4GB VRAM

Uso:
    python scripts/train_full.py --epocas 5
    python scripts/train_full.py --epocas 3 --desde_mejor
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import sentencepiece as spm

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.config import ConfiguracionEntrenamiento, PRESETS
from llarri_o1.cerebro.model import LlarriO1

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA PARA 4GB VRAM
# ============================================================================

class ConfigFull:
    """Configuración para entrenamiento completo"""
    def __init__(self):
        # Modelo
        self.preset = "LOCAL_4GB"
        self.checkpoint_mejor = "checkpoints/llarri_v8_max_best.pt"
        self.checkpoint_base = "checkpoints/llarri_v8_best.pt"
        
        # Entrenamiento - Optimizado para 4GB VRAM
        self.batch_size = 16  # Más grande para eficiencia
        self.gradient_accumulation = 4  # Batch efectivo = 64
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_grad_norm = 1.0
        
        # Mixed precision - CRÍTICO para 4GB
        self.use_mixed_precision = True
        
        # Corpus
        self.corpus_path = "data/wikitext-103/wikitext-103-raw/wiki.train.raw"
        self.tokenizer_path = "data/tokenizer/llarri_bpe.model"
        
        # Checkpoints
        self.checkpoint_dir = "checkpoints"
        self.save_every_steps = 5000
        self.eval_every_steps = 2000
        
        # Validación
        self.val_corpus_path = "data/wikitext-103/wikitext-103-raw/wiki.valid.raw"
        self.val_samples = 1000

# ============================================================================
# DATASET EFICIENTE - Streaming desde disco
# ============================================================================

class StreamingDataset(Dataset):
    """Dataset que procesa línea por línea sin cargar todo en memoria"""
    
    def __init__(self, corpus_path: str, tokenizer_path: str, seq_len: int, 
                 max_tokens: int = None, skip_tokens: int = 0):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        
        print(f"📖 Procesando corpus: {corpus_path}")
        
        # Tokenizar línea por línea
        self.tokens = []
        tokens_count = 0
        lines_count = 0
        skipped = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('='):  # Skip headers
                    continue
                
                line_tokens = self.sp.EncodeAsIds(line)
                
                # Skip inicial (para resumir entrenamiento)
                if skipped < skip_tokens:
                    skipped += len(line_tokens)
                    continue
                
                self.tokens.extend(line_tokens)
                tokens_count += len(line_tokens)
                lines_count += 1
                
                # Límite de tokens
                if max_tokens and tokens_count >= max_tokens:
                    break
                
                # Progreso cada 100K líneas
                if lines_count % 100000 == 0:
                    print(f"    Líneas: {lines_count:,} | Tokens: {tokens_count:,}")
        
        # Crear secuencias
        self.num_sequences = max(0, len(self.tokens) - seq_len)
        
        print(f"  ✅ Total: {len(self.tokens):,} tokens")
        print(f"  ✅ Secuencias: {self.num_sequences:,}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ============================================================================
# ENTRENADOR COMPLETO
# ============================================================================

class EntrenadorCompleto:
    def __init__(self, config: ConfigFull, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stats
        self.stats = {
            'inicio': datetime.now().isoformat(),
            'mejor_loss': float('inf'),
            'mejor_val_loss': float('inf'),
            'historial': []
        }
        
        self._setup()
    
    def _setup(self):
        """Configuración inicial"""
        print("\n" + "="*70)
        print("🚀 LLARRI v8 - ENTRENAMIENTO COMPLETO")
        print("="*70)
        
        # Info VRAM
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n📱 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM Total: {vram_total:.1f} GB")
        
        # Cargar modelo
        self._cargar_modelo()
        
        # Cargar tokenizer config
        preset = PRESETS[self.config.preset]
        self.seq_len = preset['seq_len']
        self.vocab_size = preset['vocab_size']
        
    def _cargar_modelo(self):
        """Carga el modelo desde el mejor checkpoint"""
        print("\n🧠 Cargando modelo...")
        
        # Determinar qué checkpoint usar
        if self.args.desde_mejor and os.path.exists(self.config.checkpoint_mejor):
            checkpoint_path = self.config.checkpoint_mejor
            print(f"   Usando mejor modelo MAX: {checkpoint_path}")
        elif os.path.exists(self.config.checkpoint_base):
            checkpoint_path = self.config.checkpoint_base
            print(f"   Usando modelo base: {checkpoint_path}")
        else:
            raise FileNotFoundError("No se encontró ningún checkpoint")
        
        # Cargar
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Crear modelo
        preset = PRESETS[self.config.preset]
        config_modelo = ConfiguracionEntrenamiento(**preset)
        self.model = LlarriO1(config_modelo)
        
        # Cargar pesos
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        
        # Info
        params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parámetros: {params:,}")
        
        # VRAM usada
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated() / 1e6
            print(f"   VRAM modelo: {vram:.0f} MB")
    
    def _limpiar_memoria(self):
        """Limpia memoria GPU"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _vram_usada(self):
        """Retorna VRAM usada en MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
        return 0
    
    def entrenar(self, epocas: int):
        """Entrenamiento principal"""
        print(f"\n🎯 Plan: {epocas} épocas sobre corpus completo")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"   Batch efectivo: {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"   Learning rate: {self.config.learning_rate}")
        
        # Cargar dataset completo
        print("\n" + "="*70)
        print("📚 CARGANDO CORPUS COMPLETO")
        print("="*70)
        
        dataset = StreamingDataset(
            self.config.corpus_path,
            self.config.tokenizer_path,
            self.seq_len,
            max_tokens=None  # TODO el corpus
        )
        
        total_tokens = len(dataset.tokens)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Evitar memoria extra
            pin_memory=True,
            drop_last=True
        )
        
        # Cargar validación
        print("\n📊 Cargando validación...")
        val_dataset = StreamingDataset(
            self.config.val_corpus_path,
            self.config.tokenizer_path,
            self.seq_len,
            max_tokens=500000  # 500K tokens para validación
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        # Optimizer y scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        total_steps = len(dataloader) * epocas // self.config.gradient_accumulation
        
        # Warmup + Cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
            return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler('cuda', enabled=self.config.use_mixed_precision)
        
        # Training loop
        print("\n" + "="*70)
        print("🏋️ INICIANDO ENTRENAMIENTO")
        print("="*70)
        
        print(f"\n📊 Estadísticas:")
        print(f"   Tokens totales: {total_tokens:,}")
        print(f"   Batches por época: {len(dataloader):,}")
        print(f"   Steps totales: {total_steps:,}")
        print(f"   Tiempo estimado: ~{total_steps * 0.15 / 60:.0f} minutos")
        
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        inicio_total = time.time()
        
        for epoca in range(1, epocas + 1):
            print(f"\n{'='*70}")
            print(f"📖 ÉPOCA {epoca}/{epocas}")
            print(f"{'='*70}")
            
            epoca_loss = 0
            epoca_tokens = 0
            batch_losses = []
            optimizer.zero_grad()
            
            inicio_epoca = time.time()
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward con mixed precision
                with autocast('cuda', enabled=self.config.use_mixed_precision):
                    output = self.model(x)
                    logits = output['logits'] if isinstance(output, dict) else output
                    
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                    loss = loss / self.config.gradient_accumulation
                
                # Backward
                scaler.scale(loss).backward()
                
                batch_losses.append(loss.item() * self.config.gradient_accumulation)
                epoca_tokens += x.numel()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Progreso cada 100 steps
                    if global_step % 100 == 0:
                        avg_loss = sum(batch_losses[-100:]) / min(100, len(batch_losses))
                        elapsed = time.time() - inicio_epoca
                        tokens_per_sec = epoca_tokens / elapsed
                        lr = scheduler.get_last_lr()[0]
                        vram = self._vram_usada()
                        
                        # Estimar tiempo restante
                        batches_restantes = len(dataloader) - batch_idx - 1
                        batches_restantes += len(dataloader) * (epocas - epoca)
                        eta_seconds = batches_restantes / (batch_idx + 1) * elapsed
                        eta = str(timedelta(seconds=int(eta_seconds)))
                        
                        print(f"  Step {global_step:,} | Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | {tokens_per_sec/1000:.1f}K tok/s | "
                              f"VRAM: {vram:.0f}MB | ETA: {eta}")
                    
                    # Evaluación periódica
                    if global_step % self.config.eval_every_steps == 0:
                        val_loss = self._evaluar(val_dataloader)
                        print(f"  📊 Validación: Loss={val_loss:.4f} | PPL={torch.exp(torch.tensor(val_loss)):.1f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._guardar_checkpoint(f"llarri_v8_full_best.pt", 
                                                    epoca, global_step, val_loss)
                            print(f"  ✅ Nuevo mejor modelo!")
                        
                        self.model.train()
                    
                    # Guardar checkpoint periódico
                    if global_step % self.config.save_every_steps == 0:
                        self._guardar_checkpoint(f"llarri_v8_full_step_{global_step}.pt",
                                                epoca, global_step, sum(batch_losses[-100:])/100)
                
                # Limpiar memoria cada 500 batches
                if batch_idx % 500 == 0:
                    self._limpiar_memoria()
            
            # Fin de época
            epoca_loss = sum(batch_losses) / len(batch_losses)
            tiempo_epoca = time.time() - inicio_epoca
            
            print(f"\n  📈 Época {epoca} completada:")
            print(f"     Loss promedio: {epoca_loss:.4f}")
            print(f"     PPL: {torch.exp(torch.tensor(epoca_loss)):.1f}")
            print(f"     Tiempo: {tiempo_epoca/60:.1f} min")
            print(f"     Tokens procesados: {epoca_tokens:,}")
            
            # Validación fin de época
            val_loss = self._evaluar(val_dataloader)
            val_ppl = torch.exp(torch.tensor(val_loss))
            print(f"     Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
            
            # Guardar estadísticas
            self.stats['historial'].append({
                'epoca': epoca,
                'train_loss': epoca_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl.item(),
                'tiempo_min': tiempo_epoca / 60,
                'tokens': epoca_tokens
            })
            
            # Guardar checkpoint de época
            self._guardar_checkpoint(f"llarri_v8_full_epoch_{epoca}.pt",
                                    epoca, global_step, val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._guardar_checkpoint("llarri_v8_full_best.pt", epoca, global_step, val_loss)
                print(f"     ✅ Nuevo mejor modelo!")
        
        # Finalizar
        tiempo_total = time.time() - inicio_total
        print(f"\n{'='*70}")
        print(f"🎉 ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"\n📊 Resumen:")
        print(f"   Tiempo total: {tiempo_total/60:.1f} minutos")
        print(f"   Mejor Val Loss: {best_val_loss:.4f}")
        print(f"   Mejor Val PPL: {torch.exp(torch.tensor(best_val_loss)):.1f}")
        print(f"   Steps totales: {global_step:,}")
        
        # Guardar estadísticas
        self.stats['fin'] = datetime.now().isoformat()
        self.stats['mejor_val_loss'] = best_val_loss
        self.stats['tiempo_total_min'] = tiempo_total / 60
        
        with open("checkpoints/train_full_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Guardar modelo final
        self._guardar_checkpoint("llarri_v8_full_final.pt", epocas, global_step, val_loss)
        
        return best_val_loss
    
    def _evaluar(self, val_dataloader) -> float:
        """Evalúa en validación"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                with autocast('cuda', enabled=self.config.use_mixed_precision):
                    output = self.model(x)
                    logits = output['logits'] if isinstance(output, dict) else output
                    
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 100:  # Limitar evaluación
                    break
        
        return total_loss / num_batches
    
    def _guardar_checkpoint(self, nombre: str, epoca: int, step: int, loss: float):
        """Guarda checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, nombre)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoca': epoca,
            'step': step,
            'loss': loss,
            'config': self.config.preset
        }, path)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLARRI v8 - Entrenamiento Completo')
    parser.add_argument('--epocas', type=int, default=3, help='Número de épocas (default: 3)')
    parser.add_argument('--desde_mejor', action='store_true', help='Partir del mejor modelo MAX')
    args = parser.parse_args()
    
    config = ConfigFull()
    entrenador = EntrenadorCompleto(config, args)
    entrenador.entrenar(args.epocas)

if __name__ == "__main__":
    main()
