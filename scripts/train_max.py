#!/usr/bin/env python3
"""
LLARRI v8 - ENTRENAMIENTO MÁXIMO
================================

Sistema avanzado para llevar LLARRI al límite en 4GB VRAM:

1. DESTILACIÓN OFFLINE: Teacher genera → descarga → Student aprende
2. SELF-PLAY: El modelo se desafía a sí mismo
3. CURRICULUM LEARNING: Dificultad progresiva
4. CONTRASTIVE LEARNING: Aprende qué NO hacer
5. REINFORCEMENT: Reward por coherencia

Autor: LLARRI Team
Límite: 4GB VRAM (GTX 1650)
"""

import os
import sys
import json
import time
import random
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.cerebro.model import LLARRIv8
from llarri_o1.config import ConfigLLARRI, LOCAL_4GB


# =============================================================================
# UTILIDADES DE MEMORIA
# =============================================================================

def limpiar_memoria():
    """Libera memoria GPU agresivamente."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def memoria_usada():
    """Retorna MB de VRAM usada."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0

def memoria_disponible():
    """Retorna MB de VRAM disponible."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        usado = torch.cuda.memory_allocated() / 1e6
        return total - usado
    return 0


# =============================================================================
# MÉTRICAS DE CALIDAD
# =============================================================================

def calcular_coherencia(texto: str) -> float:
    """Calcula score de coherencia (0-1)."""
    tokens = texto.split()
    if len(tokens) < 3:
        return 0.0
    
    # Diversidad
    diversidad = len(set(tokens)) / len(tokens)
    
    # Palabras reales
    palabras = sum(1 for t in tokens if len(t) > 2 and t.isalpha())
    prop_palabras = palabras / len(tokens)
    
    # Penalizar símbolos
    simbolos = sum(1 for t in tokens if '@' in t or t in ["'", '"', "''"])
    pen_simbolos = max(0, 1 - simbolos / len(tokens))
    
    # Penalizar repeticiones
    reps = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    pen_reps = max(0, 1 - reps / len(tokens))
    
    return diversidad * 0.25 + prop_palabras * 0.35 + pen_simbolos * 0.2 + pen_reps * 0.2


def calcular_reward(texto_generado: str, prompt: str) -> float:
    """
    Calcula reward para RLHF simplificado.
    """
    coherencia = calcular_coherencia(texto_generado)
    
    # Bonus por longitud razonable
    palabras = len(texto_generado.split())
    bonus_longitud = min(1.0, palabras / 20) if palabras > 5 else 0.5
    
    # Penalizar si repite el prompt exacto
    if prompt.lower() in texto_generado.lower()[:len(prompt)+10]:
        repeticion_prompt = 0.8
    else:
        repeticion_prompt = 1.0
    
    return coherencia * bonus_longitud * repeticion_prompt


# =============================================================================
# DATASET MEJORADO
# =============================================================================

class DatasetMejorado(Dataset):
    """Dataset con datos limpios y curriculum learning."""
    
    def __init__(
        self, 
        data_path: str,
        tokenizer_path: str,
        seq_len: int = 256,
        max_tokens: int = 500000,
        nivel_curriculum: int = 0,
    ):
        self.seq_len = seq_len
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Rangos de curriculum
        rangos = {
            0: (5, 12),    # Muy simple
            1: (8, 20),    # Simple
            2: (15, 35),   # Medio
            3: (25, 60),   # Complejo
            4: (5, 100),   # Todo
        }
        min_p, max_p = rangos.get(nivel_curriculum, (5, 100))
        
        print(f"  📚 Curriculum nivel {nivel_curriculum}: {min_p}-{max_p} palabras")
        
        self.tokens = []
        lineas_buenas = 0
        lineas_total = 0
        
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = []
            
            for line in f:
                lineas_total += 1
                line = line.strip()
                
                # Saltar títulos y vacías
                if not line or line.startswith('='):
                    continue
                
                # Limpiar tokens especiales de WikiText
                line = line.replace(' @-@ ', '-')
                line = line.replace(' @.@ ', '.')
                line = line.replace(' @,@ ', ',')
                line = line.replace("''", '"')
                line = ' '.join(line.split())
                
                palabras = line.split()
                n_palabras = len(palabras)
                
                # Filtrar por calidad
                if n_palabras < min_p or n_palabras > max_p:
                    continue
                
                # Filtrar por proporción de palabras reales
                reales = sum(1 for p in palabras if p.isalpha() and len(p) > 1)
                if reales / n_palabras < 0.6:
                    continue
                
                buffer.append(line)
                lineas_buenas += 1
                
                # Tokenizar en batches
                if len(buffer) >= 50:
                    text = ' '.join(buffer)
                    self.tokens.extend(self.tokenizer.Encode(text))
                    buffer = []
                    
                    if len(self.tokens) >= max_tokens:
                        break
            
            # Último buffer
            if buffer and len(self.tokens) < max_tokens:
                self.tokens.extend(self.tokenizer.Encode(' '.join(buffer)))
        
        self.tokens = self.tokens[:max_tokens]
        self.n_ejemplos = max(1, (len(self.tokens) - 1) // seq_len)
        
        print(f"  ✅ Líneas: {lineas_buenas:,}/{lineas_total:,} ({100*lineas_buenas/max(lineas_total,1):.1f}%)")
        print(f"  ✅ Tokens: {len(self.tokens):,}")
    
    def __len__(self):
        return self.n_ejemplos
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end]
        
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        
        return (
            torch.tensor(tokens[:-1], dtype=torch.long),
            torch.tensor(tokens[1:], dtype=torch.long)
        )


# =============================================================================
# DESTILACIÓN OFFLINE
# =============================================================================

class DestiladorOffline:
    """
    Destilación sin cargar dos modelos a la vez.
    Teacher genera → guarda → descarga → Student aprende
    """
    
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.cache_dir = Path('checkpoints/destilacion_cache')
        self.cache_dir.mkdir(exist_ok=True)
    
    def generar_conocimiento_teacher(
        self, 
        prompts: List[str],
        n_samples: int = 100,
        max_tokens: int = 50,
    ) -> str:
        """
        Teacher genera ejemplos de alta calidad.
        Guarda en disco y libera memoria.
        """
        print("  🎓 Teacher generando conocimiento...")
        
        self.model.eval()
        conocimiento = []
        
        with torch.no_grad():
            for i in range(n_samples):
                prompt = random.choice(prompts)
                tokens = self.tokenizer.Encode(prompt)
                x = torch.tensor([tokens], device=self.device)
                
                # Generar con temperatura baja (más determinista)
                for _ in range(max_tokens):
                    output = self.model(x)
                    logits = output['logits'][:, -1, :] / 0.6
                    
                    # Top-k sampling
                    top_k = 30
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    x = torch.cat([x, next_token], dim=1)
                
                texto = self.tokenizer.Decode(x[0].tolist())
                coherencia = calcular_coherencia(texto)
                
                # Solo guardar ejemplos de buena calidad
                if coherencia > 0.5:
                    conocimiento.append({
                        'prompt': prompt,
                        'generacion': texto,
                        'coherencia': coherencia,
                        'tokens': x[0].tolist(),
                    })
                
                if (i + 1) % 20 == 0:
                    print(f"    Generados: {i+1}/{n_samples}, buenos: {len(conocimiento)}")
        
        # Guardar en disco
        cache_file = self.cache_dir / f'teacher_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(cache_file, 'w') as f:
            json.dump(conocimiento, f)
        
        print(f"  ✅ Guardado: {cache_file.name} ({len(conocimiento)} ejemplos)")
        
        # Liberar memoria
        limpiar_memoria()
        
        return str(cache_file)
    
    def entrenar_student(
        self,
        cache_file: str,
        epocas: int = 2,
        lr: float = 5e-5,
    ) -> float:
        """
        Student aprende del conocimiento del Teacher.
        Usa KL Divergence para imitar distribuciones.
        """
        print("  📖 Student aprendiendo...")
        
        # Cargar conocimiento
        with open(cache_file, 'r') as f:
            conocimiento = json.load(f)
        
        if not conocimiento:
            print("  ⚠️ Sin datos de teacher")
            return 0.0
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.config.use_mixed_precision)
        
        total_loss = 0
        n_batches = 0
        
        for epoca in range(epocas):
            random.shuffle(conocimiento)
            
            for item in conocimiento:
                tokens = item['tokens']
                if len(tokens) < 5:
                    continue
                
                x = torch.tensor([tokens[:-1]], device=self.device)
                y = torch.tensor([tokens[1:]], device=self.device)
                
                optimizer.zero_grad()
                
                with autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(x, labels=y)
                    loss = output['loss']
                    
                    # Bonus: pérdida ponderada por coherencia
                    loss = loss * (1 + item['coherencia'])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  ✅ Loss destilación: {avg_loss:.4f}")
        
        return avg_loss


# =============================================================================
# SELF-PLAY (AUTO-DESAFÍO)
# =============================================================================

class SelfPlay:
    """
    El modelo se desafía a sí mismo con tareas cada vez más difíciles.
    Similar a AlphaGo pero para lenguaje.
    """
    
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Desafíos por nivel
        self.desafios = {
            0: [  # Muy fácil
                "The",
                "A",
                "In the",
            ],
            1: [  # Fácil
                "The sun is",
                "Water is made of",
                "Birds can",
            ],
            2: [  # Medio
                "The most important thing about",
                "Scientists have discovered that",
                "In the future, people will",
            ],
            3: [  # Difícil
                "The relationship between technology and society is",
                "When considering the implications of artificial intelligence,",
                "The fundamental principles of democracy include",
            ],
            4: [  # Muy difícil
                "Analyzing the complex interplay between economic factors and environmental sustainability reveals",
                "The philosophical implications of consciousness in artificial systems suggest",
                "Contemporary approaches to understanding quantum mechanics indicate",
            ],
        }
    
    def generar_desafio(self, nivel: int) -> Tuple[str, str, float]:
        """Genera un desafío y evalúa la respuesta."""
        prompts = self.desafios.get(min(nivel, 4), self.desafios[2])
        prompt = random.choice(prompts)
        
        self.model.eval()
        tokens = self.tokenizer.Encode(prompt)
        x = torch.tensor([tokens], device=self.device)
        
        with torch.no_grad():
            for _ in range(40):
                output = self.model(x)
                logits = output['logits'][:, -1, :] / 0.7
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                x = torch.cat([x, next_token], dim=1)
        
        generacion = self.tokenizer.Decode(x[0].tolist())
        reward = calcular_reward(generacion, prompt)
        
        return prompt, generacion, reward
    
    def ronda_selfplay(
        self,
        nivel: int,
        n_desafios: int = 20,
        lr: float = 5e-5,
    ) -> Tuple[float, float]:
        """
        Una ronda de self-play:
        1. Genera desafíos
        2. Evalúa respuestas
        3. Refuerza las buenas, penaliza las malas
        """
        print(f"  🎮 Self-Play nivel {nivel}")
        
        # Generar y evaluar desafíos
        resultados = []
        for _ in range(n_desafios):
            prompt, gen, reward = self.generar_desafio(nivel)
            resultados.append({
                'prompt': prompt,
                'generacion': gen,
                'reward': reward,
                'tokens': self.tokenizer.Encode(gen),
            })
        
        rewards = [r['reward'] for r in resultados]
        avg_reward = sum(rewards) / len(rewards)
        
        print(f"    Reward promedio: {avg_reward:.3f}")
        
        # Seleccionar buenos y malos ejemplos
        buenos = [r for r in resultados if r['reward'] > avg_reward]
        malos = [r for r in resultados if r['reward'] <= avg_reward * 0.7]
        
        print(f"    Buenos: {len(buenos)}, Malos: {len(malos)}")
        
        if not buenos:
            return avg_reward, 0.0
        
        # Entrenar: reforzar buenos
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.config.use_mixed_precision)
        
        total_loss = 0
        n_batches = 0
        
        for item in buenos:
            tokens = item['tokens']
            if len(tokens) < 5:
                continue
            
            x = torch.tensor([tokens[:-1]], device=self.device)
            y = torch.tensor([tokens[1:]], device=self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.config.use_mixed_precision):
                output = self.model(x, labels=y)
                # Loss ponderado por reward (más reward = más importancia)
                loss = output['loss'] * (0.5 + item['reward'])
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        return avg_reward, avg_loss


# =============================================================================
# ENTRENAMIENTO MÁXIMO
# =============================================================================

def entrenar_maximo(args):
    """Loop principal de entrenamiento máximo."""
    
    print("\n" + "="*70)
    print("🚀 LLARRI v8 - ENTRENAMIENTO MÁXIMO")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM Total: {vram:.1f} GB")
        print(f"   VRAM Límite: 4 GB (configurado)")
    
    # Paths
    ckpt_path = Path(args.checkpoint)
    data_path = Path(args.data_dir) / 'wikitext-103' / 'wikitext-103-raw' / 'wiki.train.raw'
    tokenizer_path = str(Path(args.data_dir) / 'tokenizer' / 'llarri_bpe.model')
    output_dir = Path('checkpoints')
    
    # Cargar modelo
    print(f"\n🧠 Cargando modelo...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    cfg = ckpt.get('config', LOCAL_4GB)
    if isinstance(cfg, dict):
        config = ConfigLLARRI(**{k: v for k, v in cfg.items() if k in ConfigLLARRI.__dataclass_fields__})
    else:
        config = cfg
    
    model = LLARRIv8(config).to(device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {n_params:,}")
    print(f"   VRAM usada: {memoria_usada():.0f} MB")
    
    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    # Componentes
    destilador = DestiladorOffline(model, tokenizer, device, config)
    selfplay = SelfPlay(model, tokenizer, device, config)
    
    # Historial
    historial = []
    mejor_coherencia = 0
    
    # Prompts para evaluación y teacher
    prompts_eval = [
        "The future of technology is",
        "Science has shown that",
        "In the modern world,",
        "The most important aspect of",
        "When we consider the",
    ]
    
    print(f"\n🎯 Plan de entrenamiento:")
    print(f"   Fases: {args.fases}")
    print(f"   Épocas por fase: {args.epocas_fase}")
    print(f"   Self-play rounds: {args.selfplay_rounds}")
    
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO MÁXIMO")
    print("="*70)
    
    for fase in range(args.fases):
        print(f"\n{'='*25} FASE {fase+1}/{args.fases} {'='*25}")
        print(f"VRAM: {memoria_usada():.0f} MB")
        
        fase_stats = {
            'fase': fase + 1,
            'timestamp': datetime.now().isoformat(),
        }
        
        # =====================
        # 1. CURRICULUM TRAINING
        # =====================
        print(f"\n📚 ETAPA 1: Curriculum Training (nivel {min(fase, 4)})")
        
        nivel_curr = min(fase, 4)
        dataset = DatasetMejorado(
            str(data_path),
            tokenizer_path,
            seq_len=config.max_seq_len,
            max_tokens=args.max_tokens,
            nivel_curriculum=nivel_curr,
        )
        
        loader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr * (0.85 ** fase),  # LR decay
            weight_decay=0.01
        )
        scaler = GradScaler(enabled=config.use_mixed_precision)
        
        model.train()
        for epoca in range(args.epocas_fase):
            total_loss = 0
            n_batches = 0
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                with autocast(enabled=config.use_mixed_precision):
                    output = model(x, labels=y)
                    loss = output['loss']
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Época {epoca+1}/{args.epocas_fase}: Loss={avg_loss:.4f}")
        
        fase_stats['curriculum_loss'] = avg_loss
        
        # Liberar dataset
        del dataset, loader
        limpiar_memoria()
        
        # =====================
        # 2. SELF-PLAY
        # =====================
        print(f"\n🎮 ETAPA 2: Self-Play ({args.selfplay_rounds} rondas)")
        
        total_reward = 0
        for ronda in range(args.selfplay_rounds):
            nivel_sp = min(fase + ronda // 2, 4)
            reward, sp_loss = selfplay.ronda_selfplay(
                nivel=nivel_sp,
                n_desafios=15,
                lr=args.lr * 0.5 * (0.9 ** fase),
            )
            total_reward += reward
            print(f"  Ronda {ronda+1}: Reward={reward:.3f}")
        
        fase_stats['selfplay_reward'] = total_reward / args.selfplay_rounds
        
        limpiar_memoria()
        
        # =====================
        # 3. DESTILACIÓN (cada 2 fases)
        # =====================
        if (fase + 1) % 2 == 0:
            print(f"\n🎓 ETAPA 3: Destilación Offline")
            
            cache_file = destilador.generar_conocimiento_teacher(
                prompts_eval * 4,
                n_samples=50,
                max_tokens=40,
            )
            
            limpiar_memoria()
            
            dist_loss = destilador.entrenar_student(
                cache_file,
                epocas=2,
                lr=args.lr * 0.3,
            )
            
            fase_stats['destilacion_loss'] = dist_loss
            limpiar_memoria()
        
        # =====================
        # 4. EVALUACIÓN
        # =====================
        print(f"\n📊 EVALUACIÓN")
        
        model.eval()
        coherencias = []
        
        for prompt in prompts_eval:
            tokens = tokenizer.Encode(prompt)
            x = torch.tensor([tokens], device=device)
            
            with torch.no_grad():
                for _ in range(30):
                    output = model(x)
                    logits = output['logits'][:, -1, :] / 0.7
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    x = torch.cat([x, next_token], dim=1)
            
            gen = tokenizer.Decode(x[0].tolist())
            coh = calcular_coherencia(gen)
            coherencias.append(coh)
            print(f"  '{prompt[:20]}...' → Coherencia: {coh:.3f}")
        
        avg_coherencia = sum(coherencias) / len(coherencias)
        fase_stats['coherencia'] = avg_coherencia
        
        print(f"\n📈 Coherencia promedio: {avg_coherencia:.3f}")
        
        # Guardar si es mejor
        if avg_coherencia > mejor_coherencia:
            mejor_coherencia = avg_coherencia
            best_path = output_dir / 'llarri_v8_max_best.pt'
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'fase': fase + 1,
                'coherencia': avg_coherencia,
            }, best_path)
            print(f"  ✅ Nuevo mejor modelo! Guardado: {best_path.name}")
        
        historial.append(fase_stats)
        
        # Checkpoint periódico
        if (fase + 1) % 2 == 0:
            ckpt_path = output_dir / f'llarri_v8_max_f{fase+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'historial': historial,
            }, ckpt_path)
        
        print(f"\nVRAM final fase: {memoria_usada():.0f} MB")
    
    # =====================
    # RESUMEN FINAL
    # =====================
    print("\n" + "="*70)
    print("📊 RESUMEN ENTRENAMIENTO MÁXIMO")
    print("="*70)
    
    print("\nProgreso por fase:")
    print("┌───────┬────────────┬─────────────┬────────────┐")
    print("│ Fase  │ Coherencia │ Self-Play R │ Curr Loss  │")
    print("├───────┼────────────┼─────────────┼────────────┤")
    for h in historial:
        sp_r = h.get('selfplay_reward', 0)
        c_loss = h.get('curriculum_loss', 0)
        print(f"│ {h['fase']:5} │ {h['coherencia']:10.3f} │ {sp_r:11.3f} │ {c_loss:10.4f} │")
    print("└───────┴────────────┴─────────────┴────────────┘")
    
    # Mejora total
    if len(historial) >= 2:
        mejora = historial[-1]['coherencia'] - historial[0]['coherencia']
        print(f"\n📈 Mejora coherencia: {mejora:+.3f}")
    
    print(f"\n🏆 Mejor coherencia: {mejor_coherencia:.3f}")
    
    # Guardar modelo final
    final_path = output_dir / 'llarri_v8_max_final.pt'
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'historial': historial,
        'mejor_coherencia': mejor_coherencia,
    }, final_path)
    print(f"✅ Modelo final: {final_path}")
    
    # Guardar historial
    hist_path = output_dir / 'train_max_historial.json'
    with open(hist_path, 'w') as f:
        json.dump(historial, f, indent=2)
    
    print("\n¡Entrenamiento máximo completado! 🎉")


def main():
    parser = argparse.ArgumentParser(description='LLARRI v8 - Entrenamiento Máximo')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/llarri_v8_best.pt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--fases', type=int, default=5,
                       help='Número de fases de entrenamiento')
    parser.add_argument('--epocas_fase', type=int, default=2,
                       help='Épocas de curriculum por fase')
    parser.add_argument('--selfplay_rounds', type=int, default=3,
                       help='Rondas de self-play por fase')
    parser.add_argument('--max_tokens', type=int, default=300000,
                       help='Tokens máximos por fase')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate inicial')
    
    args = parser.parse_args()
    entrenar_maximo(args)


if __name__ == "__main__":
    main()
