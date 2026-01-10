#!/usr/bin/env python3
"""
LLARRI v8 - Sistema AutoTrain
============================

Ciclo de auto-mejora iterativo:
1. EVALUAR: Mide calidad del modelo actual
2. MEJORAR: Filtra/genera datos de mejor calidad
3. ENTRENAR: Nueva ronda de entrenamiento
4. REPETIR: Hasta convergencia

Autor: LLARRI Team
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.cerebro.model import LLARRIv8
from llarri_o1.config import ConfigLLARRI, LOCAL_4GB

# =============================================================================
# FASE 1: EVALUACIÓN
# =============================================================================

class Evaluador:
    """Evalúa la calidad del modelo actual."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def calcular_perplexity(self, textos: List[str]) -> float:
        """Calcula perplexity promedio sobre textos."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for texto in textos:
                tokens = self.tokenizer.Encode(texto)
                if len(tokens) < 5:
                    continue
                    
                x = torch.tensor([tokens[:-1]], device=self.device)
                y = torch.tensor([tokens[1:]], device=self.device)
                
                output = self.model(x)
                logits = output['logits']
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += len(tokens) - 1
        
        avg_loss = total_loss / max(total_tokens, 1)
        return float(torch.exp(torch.tensor(avg_loss)))
    
    def evaluar_coherencia(self, prompt: str, generacion: str) -> float:
        """
        Evalúa coherencia de una generación (0-1).
        Métricas:
        - Diversidad de tokens (no repetitivos)
        - Proporción de palabras vs símbolos
        - Longitud de palabras razonable
        """
        tokens = generacion.split()
        if len(tokens) < 3:
            return 0.0
        
        # 1. Diversidad (tokens únicos / total)
        diversidad = len(set(tokens)) / len(tokens)
        
        # 2. Proporción de palabras reales (letras > 2)
        palabras_reales = sum(1 for t in tokens if len(t) > 2 and t.isalpha())
        prop_palabras = palabras_reales / len(tokens)
        
        # 3. Penalizar símbolos especiales
        simbolos = sum(1 for t in tokens if '@' in t or t in ['', "''", '""', "'"*3])
        penalizacion_simbolos = max(0, 1 - simbolos / len(tokens))
        
        # 4. Penalizar repeticiones consecutivas
        repeticiones = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
        penalizacion_rep = max(0, 1 - repeticiones / len(tokens))
        
        # Score final
        score = (diversidad * 0.3 + 
                 prop_palabras * 0.3 + 
                 penalizacion_simbolos * 0.2 +
                 penalizacion_rep * 0.2)
        
        return score
    
    def evaluar_modelo(self, prompts_test: List[str]) -> Dict:
        """Evaluación completa del modelo."""
        self.model.eval()
        
        resultados = {
            'coherencia_promedio': 0,
            'perplexity_estimada': 0,
            'generaciones': [],
            'timestamp': datetime.now().isoformat(),
        }
        
        coherencias = []
        
        for prompt in prompts_test:
            # Generar
            tokens = self.tokenizer.Encode(prompt)
            x = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                for _ in range(30):
                    output = self.model(x)
                    logits = output['logits'][:, -1, :] / 0.7
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    x = torch.cat([x, next_token], dim=1)
            
            generacion = self.tokenizer.Decode(x[0].tolist())
            coherencia = self.evaluar_coherencia(prompt, generacion)
            coherencias.append(coherencia)
            
            resultados['generaciones'].append({
                'prompt': prompt,
                'generacion': generacion,
                'coherencia': coherencia,
            })
        
        resultados['coherencia_promedio'] = sum(coherencias) / len(coherencias)
        
        return resultados


# =============================================================================
# FASE 2: MEJORA DE DATOS
# =============================================================================

class MejoradorDatos:
    """Mejora y filtra datos de entrenamiento."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def limpiar_linea(self, linea: str) -> str:
        """Limpia una línea de texto."""
        # Remover marcadores de Wikipedia
        linea = linea.strip()
        
        # Saltar títulos y líneas vacías
        if linea.startswith('=') or not linea:
            return ''
        
        # Remover tokens especiales problemáticos
        linea = linea.replace(' @-@ ', '-')
        linea = linea.replace(' @.@ ', '.')
        linea = linea.replace(' @,@ ', ',')
        linea = linea.replace("''", '"')
        
        # Remover múltiples espacios
        linea = ' '.join(linea.split())
        
        return linea
    
    def filtrar_por_calidad(self, lineas: List[str], min_palabras: int = 5) -> List[str]:
        """Filtra líneas de baja calidad."""
        lineas_buenas = []
        
        for linea in lineas:
            linea = self.limpiar_linea(linea)
            if not linea:
                continue
            
            palabras = linea.split()
            
            # Mínimo de palabras
            if len(palabras) < min_palabras:
                continue
            
            # Proporción de palabras reales
            palabras_reales = sum(1 for p in palabras if p.isalpha() and len(p) > 1)
            if palabras_reales / len(palabras) < 0.5:
                continue
            
            # No demasiados números
            numeros = sum(1 for p in palabras if p.isdigit())
            if numeros / len(palabras) > 0.3:
                continue
            
            lineas_buenas.append(linea)
        
        return lineas_buenas
    
    def crear_curriculum(self, lineas: List[str], nivel: int) -> List[str]:
        """
        Curriculum Learning: textos de dificultad creciente.
        
        Niveles:
        0: Oraciones simples (< 15 palabras)
        1: Oraciones medianas (15-30 palabras)
        2: Oraciones largas (30-50 palabras)
        3: Todo
        """
        if nivel >= 3:
            return lineas
        
        rangos = {
            0: (5, 15),
            1: (15, 30),
            2: (30, 50),
        }
        
        min_p, max_p = rangos.get(nivel, (5, 100))
        
        return [l for l in lineas if min_p <= len(l.split()) <= max_p]
    
    def preparar_datos_mejorados(
        self, 
        archivo_entrada: str,
        archivo_salida: str,
        nivel_curriculum: int = 0,
        max_lineas: int = 100000,
    ) -> int:
        """Prepara datos mejorados para entrenamiento."""
        
        print(f"\n📊 Mejorando datos (nivel curriculum: {nivel_curriculum})")
        
        # Leer archivo original
        with open(archivo_entrada, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
        
        print(f"   Líneas originales: {len(lineas):,}")
        
        # Filtrar por calidad
        lineas = self.filtrar_por_calidad(lineas)
        print(f"   Después de filtrar: {len(lineas):,}")
        
        # Aplicar curriculum
        lineas = self.crear_curriculum(lineas, nivel_curriculum)
        print(f"   Después de curriculum: {len(lineas):,}")
        
        # Limitar
        if len(lineas) > max_lineas:
            lineas = random.sample(lineas, max_lineas)
        
        # Guardar
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            for linea in lineas:
                f.write(linea + '\n')
        
        print(f"   ✅ Guardado: {archivo_salida}")
        print(f"   Total líneas: {len(lineas):,}")
        
        return len(lineas)


# =============================================================================
# FASE 3: ENTRENAMIENTO
# =============================================================================

def entrenar_ronda(
    model: LLARRIv8,
    config: ConfigLLARRI,
    data_path: str,
    tokenizer_path: str,
    device: torch.device,
    epocas: int = 5,
    lr: float = 1e-4,
) -> Tuple[float, float]:
    """Entrena una ronda y retorna loss inicial y final."""
    
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    
    # Dataset simple
    class TextDataset(Dataset):
        def __init__(self, path, tokenizer_path, seq_len=256, max_tokens=500000):
            self.seq_len = seq_len
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(tokenizer_path)
            
            self.tokens = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                buffer = []
                for line in f:
                    line = line.strip()
                    if line:
                        buffer.append(line)
                        if len(buffer) >= 100:
                            text = ' '.join(buffer)
                            self.tokens.extend(self.tokenizer.Encode(text))
                            buffer = []
                            if len(self.tokens) >= max_tokens:
                                break
                
                if buffer:
                    self.tokens.extend(self.tokenizer.Encode(' '.join(buffer)))
            
            self.tokens = self.tokens[:max_tokens]
            self.n_ejemplos = max(1, (len(self.tokens) - 1) // seq_len)
        
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
    
    # Crear dataset
    dataset = TextDataset(data_path, tokenizer_path, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler(enabled=config.use_mixed_precision)
    
    model.train()
    loss_inicial = None
    loss_final = None
    
    for epoca in range(epocas):
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
            
            if loss_inicial is None:
                loss_inicial = loss.item()
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"     Época {epoca+1}/{epocas} | Loss: {avg_loss:.4f}")
        loss_final = avg_loss
    
    return loss_inicial, loss_final


# =============================================================================
# CICLO PRINCIPAL AUTOTRAIN
# =============================================================================

def autotrain_ciclo(args):
    """Ejecuta el ciclo completo de AutoTrain."""
    
    print("\n" + "="*70)
    print("🔄 LLARRI v8 - AUTOTRAIN")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path('checkpoints')
    output_dir.mkdir(exist_ok=True)
    
    # Cargar modelo
    print(f"\n🧠 Cargando modelo: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    # Tokenizer
    tokenizer_path = str(data_dir / 'tokenizer' / 'llarri_bpe.model')
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    # Componentes
    evaluador = Evaluador(model, tokenizer, device)
    mejorador = MejoradorDatos(tokenizer)
    
    # Prompts de test
    prompts_test = [
        "The future of",
        "Science has discovered",
        "In the year",
        "The most important",
        "Once upon a time",
    ]
    
    # Archivo de datos original
    data_original = str(data_dir / 'wikitext-103' / 'wikitext-103-raw' / 'wiki.train.raw')
    data_mejorado = str(output_dir / 'train_mejorado.txt')
    
    # Historial de ciclos
    historial = []
    
    print(f"\n🔁 Iniciando {args.ciclos} ciclos de AutoTrain")
    print("="*70)
    
    for ciclo in range(args.ciclos):
        print(f"\n{'='*30} CICLO {ciclo+1}/{args.ciclos} {'='*30}")
        
        # =====================
        # FASE 1: EVALUAR
        # =====================
        print("\n📊 FASE 1: Evaluando modelo actual...")
        eval_resultado = evaluador.evaluar_modelo(prompts_test)
        coherencia = eval_resultado['coherencia_promedio']
        print(f"   Coherencia promedio: {coherencia:.3f}")
        
        for gen in eval_resultado['generaciones'][:2]:
            print(f"   └─ '{gen['prompt']}' → {gen['generacion'][:60]}...")
        
        # =====================
        # FASE 2: MEJORAR DATOS
        # =====================
        print("\n🔧 FASE 2: Mejorando datos de entrenamiento...")
        
        # Curriculum: empezar con textos simples, ir complicando
        nivel_curriculum = min(ciclo, 3)
        
        n_lineas = mejorador.preparar_datos_mejorados(
            data_original,
            data_mejorado,
            nivel_curriculum=nivel_curriculum,
            max_lineas=args.max_lineas,
        )
        
        # =====================
        # FASE 3: ENTRENAR
        # =====================
        print(f"\n🎓 FASE 3: Entrenando {args.epocas_por_ciclo} épocas...")
        
        # Learning rate decreciente
        lr = args.lr * (0.9 ** ciclo)
        
        loss_ini, loss_fin = entrenar_ronda(
            model=model,
            config=config,
            data_path=data_mejorado,
            tokenizer_path=tokenizer_path,
            device=device,
            epocas=args.epocas_por_ciclo,
            lr=lr,
        )
        
        # Guardar progreso
        historial.append({
            'ciclo': ciclo + 1,
            'coherencia': coherencia,
            'loss_inicial': loss_ini,
            'loss_final': loss_fin,
            'nivel_curriculum': nivel_curriculum,
            'lr': lr,
            'lineas_entrenamiento': n_lineas,
        })
        
        print(f"\n📈 Resumen ciclo {ciclo+1}:")
        print(f"   Coherencia: {coherencia:.3f}")
        print(f"   Loss: {loss_ini:.4f} → {loss_fin:.4f} (Δ {loss_fin-loss_ini:+.4f})")
        
        # Guardar checkpoint
        if (ciclo + 1) % args.guardar_cada == 0 or ciclo == args.ciclos - 1:
            ckpt_path = output_dir / f'llarri_v8_autotrain_c{ciclo+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'ciclo': ciclo + 1,
                'historial': historial,
            }, ckpt_path)
            print(f"   ✅ Checkpoint guardado: {ckpt_path}")
    
    # =====================
    # RESUMEN FINAL
    # =====================
    print("\n" + "="*70)
    print("📊 RESUMEN AUTOTRAIN")
    print("="*70)
    
    print("\nProgreso por ciclo:")
    print("┌───────┬────────────┬────────────┬────────────┬───────────┐")
    print("│ Ciclo │ Coherencia │ Loss Ini   │ Loss Fin   │ Curriculum│")
    print("├───────┼────────────┼────────────┼────────────┼───────────┤")
    for h in historial:
        print(f"│ {h['ciclo']:5} │ {h['coherencia']:10.3f} │ {h['loss_inicial']:10.4f} │ {h['loss_final']:10.4f} │ {h['nivel_curriculum']:9} │")
    print("└───────┴────────────┴────────────┴────────────┴───────────┘")
    
    # Mejora total
    if len(historial) >= 2:
        mejora_coherencia = historial[-1]['coherencia'] - historial[0]['coherencia']
        mejora_loss = historial[0]['loss_final'] - historial[-1]['loss_final']
        print(f"\n📈 Mejora total:")
        print(f"   Coherencia: {mejora_coherencia:+.3f}")
        print(f"   Loss: {mejora_loss:+.4f}")
    
    # Guardar historial
    historial_path = output_dir / 'autotrain_historial.json'
    with open(historial_path, 'w') as f:
        json.dump(historial, f, indent=2)
    print(f"\n✅ Historial guardado: {historial_path}")
    
    # Guardar modelo final
    final_path = output_dir / 'llarri_v8_autotrain_final.pt'
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'historial': historial,
    }, final_path)
    print(f"✅ Modelo final guardado: {final_path}")
    
    print("\n¡AutoTrain completado! 🎉")


def main():
    parser = argparse.ArgumentParser(description='LLARRI v8 AutoTrain')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/llarri_v8_best.pt',
                       help='Checkpoint inicial')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directorio de datos')
    parser.add_argument('--ciclos', type=int, default=5,
                       help='Número de ciclos de AutoTrain')
    parser.add_argument('--epocas_por_ciclo', type=int, default=3,
                       help='Épocas de entrenamiento por ciclo')
    parser.add_argument('--max_lineas', type=int, default=50000,
                       help='Máximo de líneas por ciclo')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate inicial')
    parser.add_argument('--guardar_cada', type=int, default=2,
                       help='Guardar checkpoint cada N ciclos')
    
    args = parser.parse_args()
    autotrain_ciclo(args)


if __name__ == "__main__":
    main()
