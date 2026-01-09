# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v8 - Inferencia Interactiva

Chat interactivo con el modelo LLARRI v8.

Uso:
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/llarri_v8_best.pt
    python scripts/chat.py --temperature 0.7 --top_p 0.95
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import sentencepiece as spm

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from llarri_o1.config import LOCAL_4GB, get_config_for_vram
from llarri_o1.cerebro.model import LLARRIv8


def load_model(checkpoint_path: str, device: torch.device):
    """Carga el modelo desde un checkpoint."""
    print(f"📥 Cargando modelo desde: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Obtener configuración
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        # Config por defecto
        config = LOCAL_4GB
    
    # Crear modelo
    model = LLARRIv8(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Info del checkpoint
    if 'epoch' in ckpt:
        print(f"   Época: {ckpt['epoch']+1}")
    if 'val_loss' in ckpt and ckpt['val_loss']:
        print(f"   Val Loss: {ckpt['val_loss']:.4f}")
    
    return model, config


def generate_text(
    model, 
    tokenizer, 
    prompt: str,
    device: torch.device,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    show_stats: bool = True,
):
    """Genera texto a partir de un prompt."""
    # Tokenizar prompt
    tokens = tokenizer.Encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    # Reset stats
    model.reset_estadisticas()
    
    # Generar
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    
    # Decodificar
    generated_text = tokenizer.Decode(output[0].tolist())
    
    # Stats de módulos
    if show_stats:
        stats = model.obtener_estadisticas_modulos()
        if stats:
            print("\n  📊 Activación de módulos:")
            for nombre, valor in sorted(stats.items(), key=lambda x: -x[1]):
                bar = "█" * int(valor * 50)
                print(f"     {nombre:12}: {bar} {valor*100:.1f}%")
    
    return generated_text


def interactive_chat(
    model,
    tokenizer,
    device: torch.device,
    config,
    args,
):
    """Modo chat interactivo."""
    print("\n" + "=" * 60)
    print("🤖 LLARRI v8 - Chat Interactivo")
    print("=" * 60)
    print("\nComandos especiales:")
    print("  /quit, /exit    - Salir")
    print("  /temp <valor>   - Cambiar temperatura (actual: {})".format(args.temperature))
    print("  /tokens <n>     - Cambiar max tokens (actual: {})".format(args.max_tokens))
    print("  /stats          - Mostrar/ocultar stats de módulos")
    print("  /clear          - Limpiar pantalla")
    print("  /help           - Mostrar ayuda")
    print("-" * 60)
    
    show_stats = True
    
    while True:
        try:
            # Prompt del usuario
            print()
            user_input = input("📝 Tú: ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd in ['/quit', '/exit', '/q']:
                    print("\n👋 ¡Hasta luego!")
                    break
                    
                elif cmd == '/temp':
                    if len(parts) > 1:
                        try:
                            args.temperature = float(parts[1])
                            print(f"   ✅ Temperatura: {args.temperature}")
                        except:
                            print("   ❌ Valor inválido")
                    else:
                        print(f"   Temperatura actual: {args.temperature}")
                    continue
                    
                elif cmd == '/tokens':
                    if len(parts) > 1:
                        try:
                            args.max_tokens = int(parts[1])
                            print(f"   ✅ Max tokens: {args.max_tokens}")
                        except:
                            print("   ❌ Valor inválido")
                    else:
                        print(f"   Max tokens actual: {args.max_tokens}")
                    continue
                    
                elif cmd == '/stats':
                    show_stats = not show_stats
                    print(f"   Stats: {'ON' if show_stats else 'OFF'}")
                    continue
                    
                elif cmd == '/clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                    
                elif cmd == '/help':
                    print("\n  Escribe cualquier texto para que LLARRI continúe.")
                    print("  Ejemplo: 'El futuro de la inteligencia artificial'")
                    continue
                    
                else:
                    print("   ❌ Comando no reconocido. Usa /help")
                    continue
            
            # Generar respuesta
            print("\n🤖 LLARRI: ", end="", flush=True)
            
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                device=device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                show_stats=show_stats,
            )
            
            # Mostrar solo la parte generada (sin el prompt)
            generated_part = response[len(user_input):].strip()
            print(generated_part)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Chat con LLARRI v8')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/llarri_v8_best.pt',
                       help='Path al checkpoint')
    parser.add_argument('--tokenizer', type=str,
                       default='data/tokenizer/llarri_bpe.model',
                       help='Path al tokenizer')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperatura para sampling')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-K filtering')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling (top-p)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                       help='Penalidad por repetición')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Máximo de tokens a generar')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Prompt único (no interactivo)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Verificar archivos
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint no encontrado: {args.checkpoint}")
        print("\nCheckpoints disponibles:")
        ckpt_dir = Path('checkpoints')
        if ckpt_dir.exists():
            for f in ckpt_dir.glob('*.pt'):
                print(f"   {f}")
        return
    
    if not os.path.exists(args.tokenizer):
        print(f"❌ Tokenizer no encontrado: {args.tokenizer}")
        return
    
    # Cargar modelo
    model, config = load_model(args.checkpoint, device)
    
    # Cargar tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer)
    
    # Registrar tokenizer en el modelo
    model.registrar_tokenizer(tokenizer)
    
    # Modo prompt único o interactivo
    if args.prompt:
        # Modo no interactivo
        print(f"\n📝 Prompt: {args.prompt}")
        print("\n🤖 LLARRI: ", end="")
        
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            show_stats=True,
        )
        
        generated_part = response[len(args.prompt):].strip()
        print(generated_part)
    else:
        # Modo interactivo
        interactive_chat(model, tokenizer, device, config, args)


if __name__ == '__main__':
    main()
