# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Entrenamiento fragmentado de PampaR v9.
Diseñado para entrenar en sesiones cortas con guardado frecuente.
Puede resumirse en cualquier momento con --resume.

Uso:
    python scripts/train_fragmentado.py --fragmento 1    # Primera sesión (10M tokens)
    python scripts/train_fragmentado.py --fragmento 2    # Segunda sesión (20M tokens)
    python scripts/train_fragmentado.py --resume         # Continuar donde quedó
    python scripts/train_fragmentado.py --max            # Entrenar hasta el máximo
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.config import ConfigPampaR, LOCAL_4GB_MAX
from pampar.cerebro.model import PampaR
from pampar.utils.data import cargar_corpus, crear_dataloader
from pampar.utils.device import get_optimal_device

# ============================================================================
# CONFIGURACIÓN DE FRAGMENTOS
# ============================================================================

FRAGMENTOS = {
    1: {"tokens": 10_000_000, "epochs": 3, "descripcion": "Fase inicial - 10M tokens"},
    2: {"tokens": 20_000_000, "epochs": 3, "descripcion": "Expansión - 20M tokens"},
    3: {"tokens": 35_000_000, "epochs": 3, "descripcion": "Profundización - 35M tokens"},
    4: {"tokens": 50_000_000, "epochs": 3, "descripcion": "Máximo - 50M tokens"},
    5: {"tokens": 75_000_000, "epochs": 2, "descripcion": "Extended - 75M tokens"},
    6: {"tokens": 100_000_000, "epochs": 2, "descripcion": "Full corpus - 100M tokens"},
}

CHECKPOINT_DIR = Path("checkpoints")
PROGRESS_FILE = CHECKPOINT_DIR / "training_progress.json"

# ============================================================================
# GESTIÓN DE PROGRESO
# ============================================================================

def cargar_progreso():
    """Cargar progreso de entrenamiento."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "fragmento_actual": 0,
        "tokens_totales_vistos": 0,
        "mejor_loss": float('inf'),
        "epochs_completados": 0,
        "historial": []
    }

def guardar_progreso(progreso):
    """Guardar progreso de entrenamiento."""
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progreso, f, indent=2)

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_fragmento(fragmento_num: int, config: ConfigPampaR, resume: bool = False):
    """Entrenar un fragmento específico."""
    
    device = get_optimal_device()
    print(f"\n{'='*60}")
    print(f"🧠 PampaR v9 - Entrenamiento Fragmentado")
    print(f"{'='*60}")
    print(f"📊 Fragmento: {fragmento_num}")
    print(f"🖥️  Device: {device}")
    
    # Cargar progreso
    progreso = cargar_progreso()
    
    # Configurar fragmento
    frag_config = FRAGMENTOS[fragmento_num]
    max_tokens = frag_config["tokens"]
    epochs = frag_config["epochs"]
    
    print(f"📝 {frag_config['descripcion']}")
    print(f"🎯 Tokens objetivo: {max_tokens:,}")
    print(f"🔄 Epochs: {epochs}")
    
    # Cargar tokenizer
    import sentencepiece as sp
    tokenizer = sp.SentencePieceProcessor()
    tokenizer_path = Path("data/tokenizer/llarri_bpe.model")
    tokenizer.Load(str(tokenizer_path))
    
    # Actualizar config con vocab size
    config.vocab_size = tokenizer.GetPieceSize()
    print(f"📚 Vocab size: {config.vocab_size:,}")
    
    # Cargar o crear modelo
    model = PampaR(config).to(device)
    model.registrar_tokenizer(tokenizer)
    
    # Cargar checkpoint si existe
    checkpoint_path = CHECKPOINT_DIR / "pampar_fragmentado_best.pt"
    start_epoch = 0
    
    if resume or checkpoint_path.exists():
        if checkpoint_path.exists():
            print(f"\n📂 Cargando checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint.get('epoch', 0)
            progreso = checkpoint.get('progreso', progreso)
            print(f"✅ Modelo cargado desde epoch {start_epoch}")
            print(f"📊 Mejor loss anterior: {progreso['mejor_loss']:.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢 Parámetros totales: {total_params:,}")
    print(f"🎓 Parámetros entrenables: {trainable_params:,}")
    
    # Cargar datos
    print(f"\n📖 Cargando corpus (máx {max_tokens:,} tokens)...")
    corpus_path = Path("data/wikitext-103/wikitext-103-raw/wiki.train.raw")
    tokens = cargar_corpus(corpus_path, tokenizer, max_tokens=max_tokens)
    print(f"✅ Tokens cargados: {len(tokens):,}")
    
    # Crear dataloader
    batch_size = 4
    seq_length = config.max_seq_len
    dataloader = crear_dataloader(tokens, batch_size, seq_length, shuffle=True)
    print(f"📦 Batches por epoch: {len(dataloader):,}")
    
    # Configurar optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Gradient scaler para mixed precision
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = config.use_mixed_precision and device.type == 'cuda'
    
    # Gradient accumulation
    accum_steps = 8
    effective_batch = batch_size * accum_steps
    print(f"🔄 Gradient accumulation: {accum_steps} (effective batch: {effective_batch})")
    
    # Criterio
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    
    # ========================================================================
    # LOOP DE ENTRENAMIENTO
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("🚀 Iniciando entrenamiento...")
    print(f"{'='*60}\n")
    
    mejor_loss = progreso['mejor_loss']
    tokens_vistos = progreso['tokens_totales_vistos']
    
    # Guardar cada N steps
    save_every = 500
    log_every = 50
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        step = 0
        
        optimizer.zero_grad()
        epoch_start = time.time()
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass con mixed precision
            if use_amp:
                with autocast('cuda'):
                    logits = model(input_ids)
                    logits = logits.view(-1, config.vocab_size)
                    targets_flat = targets.view(-1)
                    loss = criterion(logits, targets_flat)
                    loss = loss / accum_steps
                
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids)
                logits = logits.view(-1, config.vocab_size)
                targets_flat = targets.view(-1)
                loss = criterion(logits, targets_flat)
                loss = loss / accum_steps
                loss.backward()
            
            epoch_loss += loss.item() * accum_steps
            epoch_tokens += input_ids.numel()
            tokens_vistos += input_ids.numel()
            
            # Gradient step
            if (batch_idx + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                step += 1
                
                # Logging
                if step % log_every == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    elapsed = time.time() - epoch_start
                    tokens_per_sec = epoch_tokens / elapsed
                    
                    print(f"[Epoch {epoch+1}/{start_epoch+epochs}] "
                          f"Step {step:,} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"PPL: {torch.exp(torch.tensor(avg_loss)):.1f} | "
                          f"Tok/s: {tokens_per_sec:.0f} | "
                          f"Total: {tokens_vistos:,}")
                
                # Guardar checkpoint periódicamente
                if step % save_every == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    progreso['tokens_totales_vistos'] = tokens_vistos
                    
                    # Guardar checkpoint intermedio
                    checkpoint = {
                        'model': model.state_dict(),
                        'config': config.__dict__,
                        'epoch': epoch,
                        'step': step,
                        'loss': avg_loss,
                        'progreso': progreso
                    }
                    
                    temp_path = CHECKPOINT_DIR / f"pampar_frag{fragmento_num}_step_{step}.pt"
                    torch.save(checkpoint, temp_path)
                    print(f"💾 Checkpoint guardado: {temp_path.name}")
                    
                    # Si es mejor, guardar como best
                    if avg_loss < mejor_loss:
                        mejor_loss = avg_loss
                        progreso['mejor_loss'] = mejor_loss
                        torch.save(checkpoint, CHECKPOINT_DIR / "pampar_fragmentado_best.pt")
                        print(f"⭐ Nuevo mejor modelo! Loss: {mejor_loss:.4f}")
        
        # Fin de epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch+1} completado!")
        print(f"   Loss promedio: {avg_epoch_loss:.4f}")
        print(f"   Perplexity: {torch.exp(torch.tensor(avg_epoch_loss)):.2f}")
        print(f"   Tiempo: {epoch_time/60:.1f} min")
        print(f"   Tokens vistos total: {tokens_vistos:,}")
        print(f"{'='*60}\n")
        
        # Guardar al final de cada epoch
        progreso['epochs_completados'] = epoch + 1
        progreso['tokens_totales_vistos'] = tokens_vistos
        progreso['historial'].append({
            'fragmento': fragmento_num,
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'tokens': tokens_vistos,
            'timestamp': datetime.now().isoformat()
        })
        
        checkpoint = {
            'model': model.state_dict(),
            'config': config.__dict__,
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'progreso': progreso
        }
        
        epoch_path = CHECKPOINT_DIR / f"pampar_frag{fragmento_num}_epoch_{epoch+1}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"💾 Epoch guardado: {epoch_path.name}")
        
        if avg_epoch_loss < mejor_loss:
            mejor_loss = avg_epoch_loss
            progreso['mejor_loss'] = mejor_loss
            torch.save(checkpoint, CHECKPOINT_DIR / "pampar_fragmentado_best.pt")
            torch.save(checkpoint, CHECKPOINT_DIR / "pampar_best.pt")
            print(f"⭐ Mejor modelo actualizado! Loss: {mejor_loss:.4f}")
        
        guardar_progreso(progreso)
    
    # Fragmento completado
    progreso['fragmento_actual'] = fragmento_num
    guardar_progreso(progreso)
    
    print(f"\n{'='*60}")
    print(f"✅ Fragmento {fragmento_num} completado!")
    print(f"   Mejor loss: {mejor_loss:.4f}")
    print(f"   Tokens totales: {tokens_vistos:,}")
    print(f"{'='*60}")
    print(f"\n🎯 Para continuar con el siguiente fragmento:")
    print(f"   python scripts/train_fragmentado.py --fragmento {fragmento_num + 1}")
    
    return mejor_loss

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento fragmentado de PampaR v9")
    parser.add_argument('--fragmento', type=int, choices=list(FRAGMENTOS.keys()),
                        help='Número de fragmento a entrenar (1-6)')
    parser.add_argument('--resume', action='store_true',
                        help='Continuar desde el último checkpoint')
    parser.add_argument('--max', action='store_true',
                        help='Entrenar todos los fragmentos hasta el máximo')
    parser.add_argument('--status', action='store_true',
                        help='Mostrar estado del entrenamiento')
    
    args = parser.parse_args()
    
    # Mostrar estado
    if args.status:
        progreso = cargar_progreso()
        print("\n📊 Estado del entrenamiento:")
        print(f"   Fragmento actual: {progreso['fragmento_actual']}")
        print(f"   Tokens vistos: {progreso['tokens_totales_vistos']:,}")
        print(f"   Mejor loss: {progreso['mejor_loss']:.4f}")
        print(f"   Epochs completados: {progreso['epochs_completados']}")
        
        if progreso['historial']:
            print("\n📜 Historial:")
            for h in progreso['historial'][-5:]:
                print(f"   Frag {h['fragmento']} Ep {h['epoch']}: "
                      f"loss={h['loss']:.4f}, tokens={h['tokens']:,}")
        return
    
    # Configuración optimizada para 4GB VRAM
    config = LOCAL_4GB_MAX
    
    if args.max:
        # Entrenar todos los fragmentos
        progreso = cargar_progreso()
        start_frag = progreso['fragmento_actual'] + 1 if progreso['fragmento_actual'] > 0 else 1
        
        for frag_num in range(start_frag, len(FRAGMENTOS) + 1):
            print(f"\n🚀 Iniciando fragmento {frag_num}/{len(FRAGMENTOS)}")
            entrenar_fragmento(frag_num, config, resume=True)
    
    elif args.fragmento:
        entrenar_fragmento(args.fragmento, config, resume=args.resume)
    
    elif args.resume:
        progreso = cargar_progreso()
        frag_num = max(1, progreso['fragmento_actual'])
        entrenar_fragmento(frag_num, config, resume=True)
    
    else:
        # Mostrar ayuda
        print("\n🧠 PampaR v9 - Entrenamiento Fragmentado")
        print("="*50)
        print("\nFragmentos disponibles:")
        for num, info in FRAGMENTOS.items():
            print(f"  {num}. {info['descripcion']} ({info['epochs']} epochs)")
        
        print("\nUso:")
        print("  python scripts/train_fragmentado.py --fragmento 1  # Entrenar fragmento 1")
        print("  python scripts/train_fragmentado.py --resume       # Continuar")
        print("  python scripts/train_fragmentado.py --max          # Entrenar todo")
        print("  python scripts/train_fragmentado.py --status       # Ver estado")

if __name__ == "__main__":
    main()
