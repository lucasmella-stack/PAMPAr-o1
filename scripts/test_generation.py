#!/usr/bin/env python3
"""
Test de generación de texto con PampaR v8
Compara con estadísticas de otros modelos
"""

import torch
import sentencepiece as spm
import sys
import time
sys.path.insert(0, '.')

from PampaR_o1.cerebro.model import PampaRv8
from PampaR_o1.config import ConfigPampaR

def main():
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("PampaR v8 - Test de Generación")
    print("="*60)
    
    ckpt = torch.load('checkpoints/PampaR_v8_best.pt', map_location=device, weights_only=False)
    
    # Reconstruir config (puede ser dict o ConfigPampaR)
    cfg = ckpt.get('config', {})
    if isinstance(cfg, dict):
        config = ConfigPampaR(**{k: v for k, v in cfg.items() if k in ConfigPampaR.__dataclass_fields__})
    else:
        config = cfg  # Ya es ConfigPampaR
    
    model = PampaRv8(config).to(device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('data/tokenizer/PampaR_bpe.model')
    
    n_params = sum(p.numel() for p in model.parameters())
    val_loss = ckpt.get('val_loss', None)
    perplexity = ckpt.get('perplexity', None)
    
    print(f"\n📊 Modelo: PampaR v8")
    print(f"   Parámetros: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"   Device: {device}")
    print(f"   Val Loss: {val_loss:.4f}" if val_loss else "   Val Loss: N/A")
    print(f"   Perplexity: {perplexity:.2f}" if perplexity else "   Perplexity: ~487 (estimado)")
    
    def generate(prompt, max_tokens=50, temp=0.8, top_k=50):
        tokens = tokenizer.Encode(prompt)
        x = torch.tensor([tokens], device=device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(max_tokens):
                # El modelo devuelve un dict con 'logits'
                output = model(x)
                logits = output['logits'][:, -1, :] / temp
                
                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                x = torch.cat([x, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_id():
                    break
        
        elapsed = time.time() - start_time
        tokens_generated = x.shape[1] - len(tokens)
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
        
        return tokenizer.Decode(x[0].tolist()), tokens_per_sec
    
    print("\n" + "="*60)
    print("GENERACIÓN DE TEXTO")
    print("="*60)
    
    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Science has discovered that",
        "The most important thing in life",
        "Once upon a time there was",
    ]
    
    total_tps = 0
    for p in prompts:
        print(f'\n📝 Prompt: "{p}"')
        result, tps = generate(p, max_tokens=40, temp=0.7, top_k=40)
        total_tps += tps
        print(f'🤖 PampaR: {result}')
        print(f'   ⚡ {tps:.1f} tokens/s')
    
    avg_tps = total_tps / len(prompts)
    
    print("\n" + "="*60)
    print("COMPARACIÓN CON OTROS MODELOS")
    print("="*60)
    
    # Tabla comparativa
    print("""
┌─────────────────────┬────────────┬─────────────┬────────────┐
│ Modelo              │ Parámetros │ Perplexity  │ Arquitectura│
├─────────────────────┼────────────┼─────────────┼────────────┤
│ PampaR v8 (local)   │    6.1M    │   ~487      │ Modular 6  │
├─────────────────────┼────────────┼─────────────┼────────────┤
│ GPT-2 Small         │   124M     │   ~35-40    │ Transformer│
│ GPT-2 Medium        │   355M     │   ~25-30    │ Transformer│
│ DistilGPT2          │    82M     │   ~40-45    │ Transformer│
│ TinyLlama           │   1.1B     │   ~7-10     │ Transformer│
│ Phi-1               │   1.3B     │   ~5-8      │ Transformer│
└─────────────────────┴────────────┴─────────────┴────────────┘
""")
    
    print("📊 ANÁLISIS:")
    print(f"""
    PampaR v8 tiene {n_params/1e6:.1f}M params vs GPT-2 Small con 124M.
    
    - PampaR es ~20x más pequeño que GPT-2 Small
    - Perplexity más alto es esperado por:
      1. Modelo mucho más pequeño
      2. Vocab de 8k tokens (vs 50k de GPT-2)
      3. Solo 30 épocas de entrenamiento
      4. Arquitectura experimental modular
    
    ✅ LOGROS:
    - Entrena en 4GB VRAM (GTX 1650)
    - Arquitectura única con 6 módulos especializados
    - Incluye sistema de axiomas (LLAVES)
    - Escalable a servidores
    
    🎯 PRÓXIMOS PASOS para mejorar:
    - Más épocas de entrenamiento
    - Corpus más grande/limpio
    - Aumentar parámetros en servidor
    - Fine-tuning en tareas específicas
    """)
    
    print(f"\n⚡ Velocidad promedio: {avg_tps:.1f} tokens/segundo")
    
    # Test interactivo opcional
    print("\n" + "="*60)
    print("MODO INTERACTIVO (escribe 'salir' para terminar)")
    print("="*60)
    
    while True:
        try:
            prompt = input("\n📝 Tu prompt: ").strip()
            if prompt.lower() in ['salir', 'exit', 'quit', '']:
                break
            result, tps = generate(prompt, max_tokens=60, temp=0.7, top_k=40)
            print(f'🤖 PampaR: {result}')
            print(f'   ⚡ {tps:.1f} tokens/s')
        except KeyboardInterrupt:
            break
    
    print("\n¡Hasta luego! 👋")

if __name__ == "__main__":
    main()
