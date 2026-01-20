# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Test real del modelo PAMPAr-o1 v9 - Pruebas de generación y evaluación
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pampar import ConfigPampaR, PampaR
from pampar.utils.device import get_device
import sentencepiece as spm

def load_model_and_tokenizer():
    """Carga el mejor modelo y tokenizer"""
    device = get_device()
    print(f"🖥️  Dispositivo: {device}")
    
    # Cargar tokenizer
    tokenizer_path = "data/tokenizer/llarri_bpe.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    print(f"📝 Tokenizer cargado: {tokenizer.GetPieceSize()} tokens")
    
    # Buscar mejor checkpoint
    checkpoint_paths = [
        "checkpoints/pampar_fragmentado_best.pt",
        "checkpoints/pampar_best.pt",
        "checkpoints/pampar_frag3_epoch_9.pt",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("❌ No se encontró checkpoint")
        return None, None, None
    
    print(f"📦 Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Crear modelo
    config = ConfigPampaR(**checkpoint['config'])
    model = PampaR(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Registrar tokenizer
    model.registrar_tokenizer(tokenizer)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modelo cargado: {params:,} parámetros")
    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('val_loss', checkpoint.get('best_loss', '?'))
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    print(f"📊 Entrenado hasta epoch {epoch}, loss: {loss_str}")
    
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_tokens=50, temperature=0.8, top_p=0.9):
    """Genera texto a partir de un prompt"""
    # Tokenizar prompt
    input_ids = tokenizer.EncodeAsIds(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Limitar contexto
            context = generated[-model.config.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            # Forward pass - modelo retorna dict
            output = model(x)
            logits = output['logits'] if isinstance(output, dict) else output
            
            # Obtener siguiente token
            next_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Filtrar tokens con prob acumulada > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')
            
            # Samplear
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # Parar en EOS
            if next_token == tokenizer.eos_id():
                break
    
    # Decodificar
    output_text = tokenizer.DecodeIds(generated)
    return output_text


def run_generation_tests(model, tokenizer, device):
    """Ejecuta tests de generación con diferentes prompts"""
    print("\n" + "="*70)
    print("🧪 PRUEBAS DE GENERACIÓN")
    print("="*70)
    
    test_prompts = [
        # Prompts en inglés (WikiText-103 es en inglés)
        ("Continuación narrativa", "The city of", 60),
        ("Histórico", "In the year 1900 ,", 60),
        ("Científico", "The study of", 50),
        ("Descriptivo", "The largest", 40),
        ("Secuencia", "First , second , third ,", 30),
        ("Persona", "He was born in", 50),
        ("Lugar", "Located in the", 50),
        ("Evento", "The war began when", 60),
    ]
    
    for nombre, prompt, max_tokens in test_prompts:
        print(f"\n📝 Test: {nombre}")
        print(f"   Prompt: \"{prompt}\"")
        print("-" * 50)
        
        # Generar con diferentes temperaturas
        for temp in [0.7, 1.0]:
            output = generate_text(
                model, tokenizer, device, 
                prompt, 
                max_tokens=max_tokens,
                temperature=temp
            )
            print(f"   [T={temp}] {output}")
        
        print()


def run_perplexity_test(model, tokenizer, device):
    """Calcula perplejidad en textos de prueba"""
    print("\n" + "="*70)
    print("📊 PRUEBAS DE PERPLEJIDAD")
    print("="*70)
    
    test_texts = [
        "The history of the world is the history of humanity .",
        "In mathematics , a function is a relation between a set of inputs and a set of outputs .",
        "The sun rises in the east and sets in the west .",
        "She walked through the forest , listening to the birds singing .",
        "The experiment was conducted in a controlled environment .",
    ]
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    for text in test_texts:
        ids = tokenizer.EncodeAsIds(text)
        if len(ids) < 2:
            continue
            
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output = model(x)
            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            ppl = torch.exp(loss.mean()).item()
        
        print(f"   PPL: {ppl:6.1f} | \"{text[:60]}...\"" if len(text) > 60 else f"   PPL: {ppl:6.1f} | \"{text}\"")


def run_token_routing_analysis(model, tokenizer, device):
    """Analiza cómo el Tálamo enruta diferentes tipos de tokens"""
    print("\n" + "="*70)
    print("🧠 ANÁLISIS DE ENRUTAMIENTO (LLAVES)")
    print("="*70)
    
    # Verificar si el modelo tiene tálamo con LLAVES
    if not hasattr(model, 'bloques') or len(model.bloques) == 0:
        print("   ⚠️ Modelo sin bloques territoriales")
        return
    
    bloque = model.bloques[0]
    if not hasattr(bloque, 'talamo'):
        print("   ⚠️ Bloque sin tálamo")
        return
    
    talamo = bloque.talamo
    
    # Tokens de prueba por categoría
    test_tokens = {
        "Lenguaje": ["the", "and", "of", "to", "in"],
        "Matemáticas": ["1", "2", "100", "=", "+"],
        "Lógica": ["if", "then", "because", "therefore", "thus"],
        "Contexto": ["he", "she", "they", "it", "this"],
    }
    
    print("\n   Token → Activación LLAVES por módulo:")
    print("-" * 60)
    
    for categoria, tokens in test_tokens.items():
        print(f"\n   [{categoria}]")
        for token_str in tokens:
            token_id = tokenizer.PieceToId(token_str)
            if token_id == tokenizer.unk_id():
                # Intentar con espacio
                token_id = tokenizer.PieceToId("▁" + token_str)
            
            if token_id != tokenizer.unk_id():
                # Crear tensor de un solo token
                x = torch.tensor([[token_id]], dtype=torch.long, device=device)
                
                # Obtener embedding (token_embed en vez de embedding)
                with torch.no_grad():
                    if hasattr(model, 'token_embed'):
                        emb = model.token_embed(x)
                    else:
                        print(f"      '{token_str}' (id={token_id}): [modelo sin token_embed]")
                        continue
                    
                    # Obtener activaciones del tálamo si tiene el método
                    if hasattr(talamo, 'calcular_activaciones'):
                        activaciones = talamo.calcular_activaciones(emb)
                        if activaciones is not None:
                            acts_str = " | ".join([f"{k}: {v:.2f}" for k, v in activaciones.items()])
                            print(f"      '{token_str}' (id={token_id}): {acts_str}")
                        else:
                            print(f"      '{token_str}' (id={token_id}): [sin activaciones]")
                    else:
                        print(f"      '{token_str}' (id={token_id}): [tálamo sin calcular_activaciones]")
            else:
                print(f"      '{token_str}': [token desconocido]")


def run_coherence_test(model, tokenizer, device):
    """Test de coherencia: genera múltiples continuaciones y mide diversidad"""
    print("\n" + "="*70)
    print("🔄 TEST DE COHERENCIA Y DIVERSIDAD")
    print("="*70)
    
    prompt = "The scientist discovered that"
    n_generations = 5
    
    print(f"\n   Prompt: \"{prompt}\"")
    print(f"   Generando {n_generations} continuaciones diferentes:\n")
    
    generations = []
    for i in range(n_generations):
        output = generate_text(
            model, tokenizer, device,
            prompt,
            max_tokens=30,
            temperature=1.0
        )
        generations.append(output)
        print(f"   [{i+1}] {output}")
    
    # Medir diversidad (tokens únicos)
    all_tokens = []
    for gen in generations:
        all_tokens.extend(tokenizer.EncodeAsIds(gen))
    
    unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    print(f"\n   📈 Ratio de diversidad (tokens únicos/total): {unique_ratio:.2%}")


def main():
    print("="*70)
    print("🧪 TEST REAL - PAMPAr-o1 v9 Territorial")
    print("="*70)
    
    # Cargar modelo
    model, tokenizer, device = load_model_and_tokenizer()
    if model is None:
        return
    
    # Ejecutar pruebas
    run_generation_tests(model, tokenizer, device)
    run_perplexity_test(model, tokenizer, device)
    run_token_routing_analysis(model, tokenizer, device)
    run_coherence_test(model, tokenizer, device)
    
    print("\n" + "="*70)
    print("✅ PRUEBAS COMPLETADAS")
    print("="*70)


if __name__ == "__main__":
    main()
