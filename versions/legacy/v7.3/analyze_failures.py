# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Análisis de fallos del modelo LLARRI v7.2
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


def analizar_fallos():
    print('=' * 70)
    print('   🔍 ANÁLISIS DE FALLOS - LLARRI v7.2')
    print('=' * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tokenizer/llarri_bpe.model')
    print(f'\n📚 Tokenizer: {tokenizer.get_piece_size()} tokens')
    
    # Cargar modelo
    checkpoint = torch.load('checkpoints/llarri_v7.2_bpe_best.pt', map_location=device)
    config = checkpoint['config']
    
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        usar_hipocampo=True,
        capacidad_memoria=2000,
        max_iteraciones=1,
        actividad_basal=0.2,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'📊 Modelo: Epoch {checkpoint["epoch"]}, Val Loss {checkpoint["val_loss"]:.4f}')
    
    # === ANÁLISIS 1: Distribución de probabilidades ===
    print('\n' + '=' * 50)
    print('1. DISTRIBUCIÓN DE PROBABILIDADES')
    print('=' * 50)
    
    test_prompts = [
        "The scientist",
        "Once upon a",
        "In the year",
        "Machine learning",
    ]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            result = model(input_tensor)
            logits = result['logits'][0, -1, :]  # Último token
            probs = F.softmax(logits, dim=-1)
            
            # Top 10 predicciones
            top_probs, top_ids = torch.topk(probs, 10)
            
            print(f'\n📝 Prompt: "{prompt}"')
            print('   Top 10 predicciones:')
            for i, (prob, idx) in enumerate(zip(top_probs, top_ids)):
                token = tokenizer.id_to_piece(idx.item())
                print(f'      {i+1}. "{token}" ({prob.item()*100:.1f}%)')
            
            # Entropía (incertidumbre)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            print(f'   Entropía: {entropy:.2f} (mayor = más incertidumbre)')
    
    # === ANÁLISIS 2: Modulación del Tálamo ===
    print('\n' + '=' * 50)
    print('2. MODULACIÓN DEL TÁLAMO')
    print('=' * 50)
    
    for prompt in test_prompts[:2]:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            result = model(input_tensor)
            stats = result.get('stats', {})
            
            print(f'\n📝 Prompt: "{prompt}"')
            print('   Activación de módulos:')
            
            modulos = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
            for mod in modulos:
                key = f'mod_{mod}'
                if key in stats:
                    val = stats[key]
                    barra = '█' * int(val * 20) + '░' * (20 - int(val * 20))
                    print(f'      {mod:12} {barra} {val*100:.0f}%')
    
    # === ANÁLISIS 3: Tokens problemáticos ===
    print('\n' + '=' * 50)
    print('3. TOKENS MÁS GENERADOS (posibles problemas)')
    print('=' * 50)
    
    # Generar muchos tokens y ver cuáles se repiten
    generated_tokens = []
    
    for prompt in ["The", "A", "In", "Once"]:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=50,
                temperature=0.9,
                top_k=50,
                repetition_penalty=1.0,  # Sin penalty para ver comportamiento natural
            )
        
        new_tokens = output[0, len(input_ids):].tolist()
        generated_tokens.extend(new_tokens)
    
    # Contar frecuencias
    counter = Counter(generated_tokens)
    print('\n   Tokens más frecuentes en generación:')
    for token_id, count in counter.most_common(15):
        token = tokenizer.id_to_piece(token_id)
        pct = count / len(generated_tokens) * 100
        print(f'      "{token}" (ID {token_id}): {count} veces ({pct:.1f}%)')
    
    # === ANÁLISIS 4: Consenso entre módulos ===
    print('\n' + '=' * 50)
    print('4. CONSENSO ENTRE MÓDULOS')
    print('=' * 50)
    
    for prompt in test_prompts[:2]:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            result = model(input_tensor)
            stats = result.get('stats', {})
            
            consenso = stats.get('consenso_mean', 0)
            conflicto = stats.get('conflicto_mean', 0)
            
            print(f'\n📝 Prompt: "{prompt}"')
            print(f'   Consenso:  {consenso:.3f}')
            print(f'   Conflicto: {conflicto:.3f}')
            print(f'   Balance:   {consenso - conflicto:+.3f}')
    
    # === RESUMEN DE PROBLEMAS ===
    print('\n' + '=' * 70)
    print('   📋 RESUMEN DE PROBLEMAS IDENTIFICADOS')
    print('=' * 70)
    
    print('''
    1. ALTA ENTROPÍA: El modelo tiene mucha incertidumbre sobre qué 
       token generar, lo que causa selecciones aleatorias.
    
    2. TOKENS FRECUENTES: Ciertos tokens (números, sufijos) se generan
       desproporcionadamente, indicando sesgo en el entrenamiento.
    
    3. BAJO CONSENSO: Los módulos no "acuerdan" sobre la respuesta,
       lo que indica procesamiento incoherente.
    
    4. MODULACIÓN UNIFORME: El tálamo no diferencia bien entre tipos
       de contenido (todo se procesa similar).
    
    SOLUCIÓN PROPUESTA:
    - Autoentrenamiento que premie alto consenso
    - Penalizar tokens que se generan demasiado frecuentemente
    - Reforzar modulación diferenciada del tálamo
    ''')


if __name__ == '__main__':
    analizar_fallos()
