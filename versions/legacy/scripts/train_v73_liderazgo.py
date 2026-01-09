# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Entrenamiento de LLARRI v7.3 con Liderazgo Dinámico

Mejoras de entrenamiento:
1. Loss adicional por consenso (premiar cuando módulos acuerdan)
2. Loss por liderazgo claro (penalizar liderazgo difuso)
3. Visualización del comportamiento de liderazgo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from pathlib import Path
import sys
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v73 import LLARRIv73Liderazgo


class WikiText103Dataset(Dataset):
    """Dataset simple para WikiText-103."""
    
    def __init__(
        self, 
        tokenizer, 
        seq_len: int = 256,
        split: str = 'train',
        max_samples: int = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Cargar datos
        data_dir = Path('data/wikitext-103/wikitext-103-raw')
        file_map = {
            'train': 'wiki.train.raw',
            'valid': 'wiki.valid.raw',
            'test': 'wiki.valid.raw',  # Usar valid como test si no existe
        }
        
        path = data_dir / file_map[split]
        
        print(f'   Cargando {path}...')
        with open(path, 'r', encoding='utf-8') as f:
            # Leer solo una porción del archivo para evitar problemas de memoria
            if split == 'train':
                text = f.read(50_000_000)  # ~50MB para train
            else:
                text = f.read(5_000_000)   # ~5MB para valid
        
        # Limpiar texto
        lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('=')]
        text = ' '.join(lines)
        
        # Tokenizar por chunks para evitar crash
        print(f'   Tokenizando ({len(text)/1e6:.1f}MB)...')
        chunk_size = 100_000
        self.tokens = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            self.tokens.extend(tokenizer.encode(chunk))
        
        # Calcular número de samples
        n_samples = (len(self.tokens) - 1) // seq_len
        if max_samples:
            n_samples = min(n_samples, max_samples)
        self.n_samples = n_samples
        
        print(f'   {len(self.tokens)} tokens -> {n_samples} samples')
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        chunk = self.tokens[start:end]
        
        # Padding si es necesario
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        
        return input_ids, targets


def train_v73():
    print('=' * 70)
    print('   🧠 ENTRENAMIENTO LLARRI v7.3 - LIDERAZGO DINÁMICO')
    print('=' * 70)
    
    # === CONFIGURACIÓN ===
    config = {
        'vocab_size': 8000,
        'd_model': 128,
        'n_heads': 4,
        'dropout': 0.15,
        'actividad_basal': 0.15,
        'temperatura_liderazgo': 0.5,
        'usar_hipocampo': True,
        'capacidad_memoria': 2000,
        'max_iteraciones': 2,
        
        # Training
        'batch_size': 16,
        'seq_len': 256,
        'epochs': 10,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,
        
        # Loss adicionales
        'peso_consenso': 0.1,       # Premiar consenso
        'peso_liderazgo_claro': 0.05,  # Premiar liderazgo definido
        
        # Dispositivo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print(f'\n📋 Configuración:')
    for k, v in config.items():
        print(f'   {k}: {v}')
    
    device = torch.device(config['device'])
    
    # === CARGAR TOKENIZER ===
    print('\n📚 Cargando tokenizer BPE...')
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer/llarri_bpe.model')
    print(f'   Vocabulario: {tokenizer.get_piece_size()} tokens')
    
    # === CREAR DATASET ===
    print('\n📊 Preparando datasets...')
    
    train_dataset = WikiText103Dataset(
        tokenizer=tokenizer,
        seq_len=config['seq_len'],
        split='train',
        max_samples=50000,
    )
    
    val_dataset = WikiText103Dataset(
        tokenizer=tokenizer,
        seq_len=config['seq_len'],
        split='valid',
        max_samples=5000,
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
    
    print(f'   Train: {len(train_dataset)} samples')
    print(f'   Val: {len(val_dataset)} samples')
    
    # === CREAR MODELO ===
    print('\n🧠 Creando modelo v7.3...')
    model = LLARRIv73Liderazgo(
        vocab_size=config['vocab_size'],
        dim=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        actividad_basal=config['actividad_basal'],
        temperatura_liderazgo=config['temperatura_liderazgo'],
        usar_hipocampo=config['usar_hipocampo'],
        capacidad_memoria=config['capacidad_memoria'],
        max_iteraciones=config['max_iteraciones'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'   Parámetros: {n_params:,} ({n_params/1e6:.2f}M)')
    
    # === OPTIMIZADOR ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # === LOSS ===
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=config['label_smoothing'],
    )
    
    # === ENTRENAMIENTO ===
    best_val_loss = float('inf')
    historia_lideres = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config["epochs"]}')
        print(f'{"="*60}')
        
        # Training
        model.train()
        train_loss = 0
        train_loss_lm = 0
        train_loss_consenso = 0
        train_loss_liderazgo = 0
        n_batches = 0
        
        conteo_lideres = {
            'lenguaje': 0, 'logica': 0, 'matematicas': 0,
            'patrones': 0, 'contexto': 0, 'creatividad': 0
        }
        
        start_time = time.time()
        
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            result = model(input_ids, targets)
            logits = result['logits']
            stats = result['stats']
            
            # Loss principal (Language Modeling)
            loss_lm = criterion(
                logits.view(-1, config['vocab_size']),
                targets.view(-1)
            )
            
            # Loss de consenso (premiar cuando los módulos acuerdan)
            consenso = stats.get('consenso_ponderado_mean', 0.5)
            loss_consenso = -torch.tensor(consenso, device=device) * config['peso_consenso']
            
            # Loss de liderazgo claro (premiar cuando hay un líder definido)
            confianza = stats.get('lider_confianza', 0.5)
            loss_liderazgo = -torch.tensor(confianza, device=device) * config['peso_liderazgo_claro']
            
            # Loss total
            loss = loss_lm + loss_consenso + loss_liderazgo
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            # Tracking
            train_loss += loss.item()
            train_loss_lm += loss_lm.item()
            train_loss_consenso += loss_consenso.item()
            train_loss_liderazgo += loss_liderazgo.item()
            n_batches += 1
            
            # Contar líder
            lider = stats.get('lider_nombre', 'desconocido')
            if lider in conteo_lideres:
                conteo_lideres[lider] += 1
            
            # Log cada 100 batches
            if batch_idx % 100 == 0:
                avg_loss = train_loss / n_batches
                print(f'   Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss={avg_loss:.4f}, '
                      f'Líder={lider}, '
                      f'Consenso={consenso:.3f}')
        
        # Fin epoch training
        train_loss /= n_batches
        elapsed = time.time() - start_time
        
        print(f'\n📊 Training Epoch {epoch}:')
        print(f'   Loss Total: {train_loss:.4f}')
        print(f'   Loss LM: {train_loss_lm/n_batches:.4f}')
        print(f'   Loss Consenso: {train_loss_consenso/n_batches:.4f}')
        print(f'   Loss Liderazgo: {train_loss_liderazgo/n_batches:.4f}')
        print(f'   Tiempo: {elapsed:.1f}s')
        
        # Mostrar distribución de líderes
        print(f'\n   🎯 Distribución de líderes:')
        total_lideres = sum(conteo_lideres.values())
        for nombre, conteo in conteo_lideres.items():
            pct = conteo / total_lideres * 100 if total_lideres > 0 else 0
            barra = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f'      {nombre:12} {barra} {pct:.1f}%')
        
        historia_lideres.append(conteo_lideres.copy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_consenso = 0
        n_val = 0
        
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                result = model(input_ids, targets)
                logits = result['logits']
                stats = result['stats']
                
                loss = criterion(
                    logits.view(-1, config['vocab_size']),
                    targets.view(-1)
                )
                
                val_loss += loss.item()
                val_consenso += stats.get('consenso_ponderado_mean', 0)
                n_val += 1
        
        val_loss /= n_val
        val_consenso /= n_val
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        print(f'\n📊 Validation:')
        print(f'   Loss: {val_loss:.4f}')
        print(f'   PPL: {val_ppl:.2f}')
        print(f'   Consenso: {val_consenso:.3f}')
        
        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'config': config,
            'historia_lideres': historia_lideres,
        }
        
        # Guardar cada 2 epochs
        if epoch % 2 == 0:
            path = f'checkpoints/llarri_v7.3_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
            print(f'   💾 Guardado: {path}')
        
        # Guardar mejor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = 'checkpoints/llarri_v7.3_best.pt'
            torch.save(checkpoint, path)
            print(f'   ⭐ Nuevo mejor modelo guardado!')
        
        # Step scheduler
        scheduler.step()
    
    print('\n' + '=' * 70)
    print('   ✅ ENTRENAMIENTO COMPLETADO')
    print('=' * 70)
    print(f'   Mejor Val Loss: {best_val_loss:.4f}')
    print(f'   Mejor PPL: {torch.exp(torch.tensor(best_val_loss)).item():.2f}')
    
    # Test de generación rápido
    print('\n🧪 Test de generación:')
    model.eval()
    
    prompts = [
        "The scientist",
        "Once upon a",
        "Machine learning",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=30,
                temperature=0.9,
                top_k=50,
                repetition_penalty=1.3,
            )
        
        generated = tokenizer.decode(output[0].tolist())
        print(f'\n   📝 "{prompt}"')
        print(f'      → {generated}')


if __name__ == '__main__':
    train_v73()
