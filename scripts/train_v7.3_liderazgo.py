# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Entrenamiento LLARRI v7.3 con Sistema de Liderazgo Dinámico
OPTIMIZADO PARA 4GB VRAM - GTX 1650
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import math
import gc
from pathlib import Path

# Configuración ULTRA-CONSERVADORA para 4GB VRAM
BATCH_SIZE = 2           # Muy pequeño
SEQ_LEN = 128            # Secuencias cortas
GRAD_ACCUM = 8           # Simular batch de 16
DIM = 128                # Dimensión del modelo
N_HEADS = 4              
VOCAB_SIZE = 8000
EPOCHS = 10
LR = 3e-4
MAX_TOKENS = 500_000     # Limitar dataset (500K tokens max)


class ModuloCerebral(nn.Module):
    """Módulo especializado que puede ser LÍDER o SEGUIDOR"""
    
    def __init__(self, dim, n_heads, dropout=0.15, nombre=""):
        super().__init__()
        self.nombre = nombre
        self.dim = dim
        
        # Atención propia
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Reducido de 4x a 2x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # Capa de acoplamiento (recibe señal del líder)
        self.acoplamiento = nn.Linear(dim, dim)
        self.gate_acople = nn.Linear(dim * 2, 1)  # Cuánto seguir al líder
        
    def forward(self, x, mask=None, señal_lider=None, es_lider=False):
        # Atención
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Si hay señal del líder y NO soy el líder, acoplarme
        if señal_lider is not None and not es_lider:
            señal_procesada = self.acoplamiento(señal_lider)
            
            # Gate: cuánto seguir al líder (0-1)
            gate_input = torch.cat([x, señal_procesada], dim=-1)
            gate = torch.sigmoid(self.gate_acople(gate_input))
            
            # Mezclar mi salida con la señal del líder
            x = x * (1 - gate * 0.5) + señal_procesada * (gate * 0.5)
        
        return x


class TalamoConLiderazgo(nn.Module):
    """Tálamo que selecciona un LÍDER y coordina los módulos"""
    
    def __init__(self, dim, n_modulos=6):
        super().__init__()
        self.n_modulos = n_modulos
        
        # Detector de tipo de tarea
        self.task_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_modulos),  # Score por módulo
        )
        
        # Nombres de módulos (para logging)
        self.nombres = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
        
    def forward(self, x):
        # Usar el promedio de la secuencia para detectar tarea
        x_mean = x.mean(dim=1)  # [batch, dim]
        
        # Scores de liderazgo por módulo
        scores = self.task_detector(x_mean)  # [batch, n_modulos]
        
        # Softmax para elegir líder (uno dominante)
        liderazgo = F.softmax(scores * 2.0, dim=-1)  # Temperatura baja = más decisivo
        
        # También calcular pesos de participación (todos participan algo)
        participacion = F.softmax(scores, dim=-1) * 0.5 + 0.5 / self.n_modulos
        
        return {
            'liderazgo': liderazgo,      # Quién es el líder
            'participacion': participacion,  # Cuánto participa cada uno
            'lider_idx': liderazgo.argmax(dim=-1),  # Índice del líder
        }


class LLARRIv73Liderazgo(nn.Module):
    """LLARRI v7.3 con Sistema de Liderazgo Dinámico entre Módulos"""
    
    def __init__(self, vocab_size, dim, n_heads, dropout=0.15, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Tálamo con sistema de liderazgo
        self.talamo = TalamoConLiderazgo(dim, n_modulos=6)
        
        # 6 Módulos cerebrales especializados
        self.modulos = nn.ModuleDict({
            'lenguaje': ModuloCerebral(dim, n_heads, dropout, 'lenguaje'),
            'logica': ModuloCerebral(dim, n_heads, dropout, 'logica'),
            'matematicas': ModuloCerebral(dim, n_heads, dropout, 'matematicas'),
            'patrones': ModuloCerebral(dim, n_heads, dropout, 'patrones'),
            'contexto': ModuloCerebral(dim, n_heads, dropout, 'contexto'),
            'creatividad': ModuloCerebral(dim, n_heads, dropout, 'creatividad'),
        })
        
        # Integrador final
        self.integrador = nn.Linear(dim, dim)
        self.norm_final = nn.LayerNorm(dim)
        
        # Output
        self.output = nn.Linear(dim, vocab_size)
        
        # Para estadísticas
        self.ultimo_lider = None
        self.ultimos_pesos = None
        
    def forward(self, x, targets=None):
        batch, seq_len = x.shape
        device = x.device
        
        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h = self.dropout(h)
        
        # Máscara causal
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Tálamo decide quién lidera
        talamo_out = self.talamo(h)
        liderazgo = talamo_out['liderazgo']  # [batch, 6]
        participacion = talamo_out['participacion']
        lider_idx = talamo_out['lider_idx']  # [batch]
        
        # Guardar para estadísticas
        self.ultimo_lider = lider_idx
        self.ultimos_pesos = liderazgo
        
        # Procesar módulos
        nombres = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
        salidas = {}
        
        # Primero: el líder procesa (sin señal externa)
        # Usamos el líder más común del batch para simplicidad
        lider_comun = lider_idx.mode().values.item()
        nombre_lider = nombres[lider_comun]
        
        señal_lider = self.modulos[nombre_lider](h, mask=mask, señal_lider=None, es_lider=True)
        salidas[nombre_lider] = señal_lider
        
        # Luego: los seguidores procesan (con señal del líder)
        for i, nombre in enumerate(nombres):
            if nombre != nombre_lider:
                salida = self.modulos[nombre](h, mask=mask, señal_lider=señal_lider, es_lider=False)
                salidas[nombre] = salida
        
        # Integrar con pesos de participación
        h_integrado = torch.zeros_like(h)
        for i, nombre in enumerate(nombres):
            peso = participacion[:, i].view(-1, 1, 1)  # [batch, 1, 1]
            h_integrado = h_integrado + salidas[nombre] * peso
        
        # Normalizar y proyectar
        h_final = self.norm_final(self.integrador(h_integrado))
        logits = self.output(h_final)
        
        # Calcular loss si hay targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'lider': nombre_lider,
            'liderazgo': liderazgo,
            'participacion': participacion,
        }
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.9, top_k=50, 
                 repetition_penalty=1.2, top_p=0.92):
        """Generación con sampling"""
        self.eval()
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = generated[:, -256:]
            
            with torch.no_grad():
                output = self(context)
                logits = output['logits'][:, -1, :] / temperature
                
                # Repetition penalty
                for i in range(generated.shape[0]):
                    for token_id in generated[i].unique():
                        logits[i, token_id] /= repetition_penalty
                
                # Top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class WikiTextDataset(Dataset):
    """Dataset minimalista para WikiText"""
    
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
    """Cargar datos limitados para no crashear"""
    print(f"📂 Cargando datos (máx {max_tokens:,} tokens)...")
    
    train_path = Path("data/wikitext-103/wiki.train.tokens")
    val_path = Path("data/wikitext-103/wiki.valid.tokens")
    
    # Leer solo parte del archivo
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = ""
        for line in f:
            train_text += line
            if len(train_text) > max_tokens * 4:  # Aproximado
                break
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read(max_tokens)  # Solo 500K chars para validación
    
    print(f"   Train text: {len(train_text):,} chars")
    print(f"   Val text: {len(val_text):,} chars")
    
    # Tokenizar en chunks para no saturar memoria
    print("   Tokenizando...")
    train_tokens = tokenizer.encode(train_text)[:max_tokens]
    val_tokens = tokenizer.encode(val_text)[:max_tokens // 5]
    
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")
    
    # Limpiar
    del train_text, val_text
    gc.collect()
    
    return train_tokens, val_tokens


def entrenar():
    print("=" * 70)
    print("   🧠 LLARRI v7.3 - SISTEMA DE LIDERAZGO DINÁMICO")
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
    print(f"   Batch size: {BATCH_SIZE} (efectivo: {BATCH_SIZE * GRAD_ACCUM})")
    
    # Modelo
    print("\n🏗️ Creando modelo v7.3...")
    model = LLARRIv73Liderazgo(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_heads=N_HEADS,
        dropout=0.15,
        max_seq_len=512
    )
    
    # Contar parámetros
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
    
    for epoch in range(1, EPOCHS + 1):
        # === TRAIN ===
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        lider_counts = {n: 0 for n in ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            output = model(x, targets=y)
            loss = output['loss'] / GRAD_ACCUM
            loss.backward()
            
            total_loss += output['loss'].item()
            lider_counts[output['lider']] += 1
            
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Progress cada 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"   Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
                
                # Limpiar memoria
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Mostrar distribución de líderes
        print(f"\n   📊 Distribución de líderes en epoch {epoch}:")
        total_lider = sum(lider_counts.values())
        for nombre, count in sorted(lider_counts.items(), key=lambda x: -x[1]):
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
        
        # Guardar checkpoint
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
                    'version': '7.3',
                    'arquitectura': 'liderazgo_dinamico',
                }
            }
            torch.save(checkpoint, 'checkpoints/llarri_v7.3_liderazgo_best.pt')
            print(f"   💾 Mejor modelo guardado!")
        
        # Limpiar memoria entre epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    # === TEST DE GENERACIÓN ===
    print("\n" + "=" * 70)
    print("   🔤 TEST DE GENERACIÓN")
    print("=" * 70)
    
    model.eval()
    test_prompts = ["The scientist", "Once upon a", "In the year"]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=30,
                temperature=0.9,
                top_k=50,
                repetition_penalty=1.3,
            )
        
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"\n   📝 '{prompt}' →")
        print(f"      {generated}")
    
    print("\n" + "=" * 70)
    print("   ✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    entrenar()
