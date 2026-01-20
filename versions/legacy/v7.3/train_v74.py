# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
#
# ⚠️ LEGACY CODE - For historical reference only
# Imports reference modules that no longer exist in this structure
#
"""
Entrenamiento LLARRI v7.4 con Tálamo Orquestador
OPTIMIZADO PARA 4GB VRAM - GTX 1650

El Tálamo ahora tiene REGLAS CLARAS:
- Detectores explícitos para cada tipo de contenido
- Llaves que activan cada módulo
- Liderazgo definido ANTES de procesar
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import math
import gc
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.models.language_model_v74 import LLARRIv74Orquestado

# ============================================================================
# CONFIGURACIÓN ULTRA-CONSERVADORA PARA 4GB VRAM
# ============================================================================
BATCH_SIZE = 2           
SEQ_LEN = 128            
GRAD_ACCUM = 8           
DIM = 128                
N_HEADS = 4              
VOCAB_SIZE = 8000
EPOCHS = 10
LR = 3e-4
MAX_TOKENS = 500_000     


class WikiTextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = self.tokens[:self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def cargar_datos_limitado(tokenizer, max_tokens=MAX_TOKENS):
    print(f"📂 Cargando datos (máx {max_tokens:,} tokens)...")
    
    train_path = Path("data/wikitext-103/wikitext-103-raw/wiki.train.raw")
    val_path = Path("data/wikitext-103/wikitext-103-raw/wiki.valid.raw")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = ""
        for line in f:
            train_text += line
            if len(train_text) > max_tokens * 4:
                break
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read(max_tokens)
    
    print(f"   Train text: {len(train_text):,} chars")
    print(f"   Val text: {len(val_text):,} chars")
    
    print("   Tokenizando...")
    train_tokens = tokenizer.encode(train_text)[:max_tokens]
    val_tokens = tokenizer.encode(val_text)[:max_tokens // 5]
    
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")
    
    del train_text, val_text
    gc.collect()
    
    return train_tokens, val_tokens


def entrenar():
    print("=" * 70)
    print("   🧠 LLARRI v7.4 - TÁLAMO ORQUESTADOR CON REGLAS")
    print("   ⚠️  Optimizado para 4GB VRAM")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM Total: {mem:.1f} GB")
    
    # Tokenizer
    print("\n📚 Cargando tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer/llarri_bpe.model')
    print(f"   Vocab: {tokenizer.get_piece_size()} tokens")
    
    # Datos
    train_tokens, val_tokens = cargar_datos_limitado(tokenizer)
    
    train_dataset = WikiTextDataset(train_tokens, SEQ_LEN)
    val_dataset = WikiTextDataset(val_tokens, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    print(f"\n📊 Dataset:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Modelo
    print("\n🏗️ Creando modelo v7.4 con Tálamo Orquestador...")
    model = LLARRIv74Orquestado(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_heads=N_HEADS,
        dropout=0.15,
        usar_hipocampo=True,
        capacidad_memoria=2000,
        actividad_basal=0.15,
    )
    
    # ¡IMPORTANTE! Inicializar el Tálamo con el tokenizer
    print("\n🔑 Inicializando llaves del Tálamo...")
    model.inicializar_talamo(tokenizer)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {n_params:,} ({n_params/1e6:.2f}M)")
    
    model = model.to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM usado (modelo): {mem_used:.2f} GB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Entrenamiento
    print("\n" + "=" * 70)
    print("   🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    best_val_loss = float('inf')
    nombres_modulos = model.ORDEN_MODULOS
    
    for epoch in range(1, EPOCHS + 1):
        # === TRAIN ===
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        lider_counts = {n: 0 for n in nombres_modulos}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            output = model(x, targets=y)
            loss = output['loss'] / GRAD_ACCUM
            loss.backward()
            
            total_loss += output['loss'].item()
            
            # Trackear líder
            stats = output.get('stats', {})
            lider_nombre = stats.get('lider', 'unknown')
            if lider_nombre in lider_counts:
                lider_counts[lider_nombre] += 1
            
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lider_actual = stats.get('lider', '?')
                print(f"   Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Líder: {lider_actual}")
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Mostrar distribución de líderes
        print(f"\n   📊 Distribución de líderes en epoch {epoch}:")
        total_lider = sum(lider_counts.values())
        if total_lider > 0:
            for nombre in nombres_modulos:
                count = lider_counts.get(nombre, 0)
                pct = count / total_lider * 100
                barra = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"      {nombre:12} {barra} {pct:.0f}%")
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x, targets=y)
                val_loss += output['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        ppl = math.exp(min(avg_val_loss, 20))
        
        print(f"\n   ✅ Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | PPL={ppl:.2f}")
        
        scheduler.step()
        
        # Guardar
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': {
                    'vocab_size': VOCAB_SIZE,
                    'd_model': DIM,
                    'n_heads': N_HEADS,
                    'dropout': 0.15,
                    'version': '7.4',
                    'arquitectura': 'talamo_orquestador',
                    'usar_hipocampo': True,
                    'capacidad_memoria': 2000,
                }
            }
            torch.save(checkpoint, 'checkpoints/llarri_v7.4_best.pt')
            print(f"   💾 Mejor modelo guardado!")
        
        if epoch % 3 == 0:
            torch.save(checkpoint, f'checkpoints/llarri_v7.4_epoch_{epoch}.pt')
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    # === TEST ===
    print("\n" + "=" * 70)
    print("   🔤 TEST DE GENERACIÓN Y LIDERAZGO")
    print("=" * 70)
    
    model.eval()
    test_prompts = [
        ("The scientist", "lenguaje"),
        ("2 + 2 =", "matematicas"),
        ("If it rains then", "logica"),
        ("Once upon a time", "contexto"),
    ]
    
    for prompt, esperado in test_prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            result = model(input_tensor)
            lider = result['stats'].get('lider', '?')
            
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=20,
                temperature=0.9,
                top_k=50,
                repetition_penalty=1.3,
            )
        
        generated = tokenizer.decode(output_ids[0].tolist())
        match = "✅" if lider == esperado else "❌"
        print(f"\n   {match} '{prompt}' → Líder: {lider} (esperado: {esperado})")
        print(f"      Generado: {generated[:80]}...")
    
    print("\n" + "=" * 70)
    print("   ✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    entrenar()
