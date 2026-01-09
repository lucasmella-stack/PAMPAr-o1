# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Descarga corpus de texto para entrenar LLARRI v8.

Incluye:
- Wikipedia en español (dump simplificado)
- Subset de datos de dominio público

Uso:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --lang es --max_mb 100
"""

import os
import sys
import argparse
import urllib.request
import gzip
import shutil
from pathlib import Path


def download_file(url: str, dest: str, desc: str = ""):
    """Descarga un archivo con barra de progreso."""
    print(f"📥 Descargando: {desc or url}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb_done = count * block_size / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(f"\r   {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest, progress_hook)
        print()  # Nueva línea
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def create_spanish_subset(output_dir: str, max_chars: int = 50_000_000):
    """
    Crea un subset de texto en español de dominio público.
    
    Incluye textos clásicos de Project Gutenberg en español.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs de textos en español de dominio público (Project Gutenberg)
    spanish_texts = [
        # Don Quijote
        ("https://www.gutenberg.org/cache/epub/2000/pg2000.txt", "quijote.txt"),
        # La Regenta
        ("https://www.gutenberg.org/cache/epub/17073/pg17073.txt", "regenta.txt"),
        # Fortunata y Jacinta
        ("https://www.gutenberg.org/cache/epub/17656/pg17656.txt", "fortunata.txt"),
        # Pepita Jiménez
        ("https://www.gutenberg.org/cache/epub/15530/pg15530.txt", "pepita.txt"),
        # El sombrero de tres picos
        ("https://www.gutenberg.org/cache/epub/15781/pg15781.txt", "sombrero.txt"),
        # Cuentos de la Alhambra (Washington Irving, traducido)
        ("https://www.gutenberg.org/cache/epub/16212/pg16212.txt", "alhambra.txt"),
    ]
    
    all_text = []
    total_chars = 0
    
    for url, filename in spanish_texts:
        if total_chars >= max_chars:
            break
            
        dest_path = output_dir / filename
        
        if dest_path.exists():
            print(f"  ✅ Ya existe: {filename}")
            with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            if not download_file(url, str(dest_path), filename):
                continue
            
            with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Limpiar texto de Gutenberg (remover header/footer)
        start_marker = "*** START OF"
        end_marker = "*** END OF"
        
        start_idx = text.find(start_marker)
        if start_idx != -1:
            start_idx = text.find('\n', start_idx) + 1
        else:
            start_idx = 0
            
        end_idx = text.find(end_marker)
        if end_idx == -1:
            end_idx = len(text)
        
        clean_text = text[start_idx:end_idx].strip()
        all_text.append(clean_text)
        total_chars += len(clean_text)
        
        print(f"  📖 {filename}: {len(clean_text):,} caracteres")
    
    # Guardar corpus combinado
    combined_path = output_dir / "corpus_spanish.txt"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    print(f"\n✅ Corpus español guardado: {combined_path}")
    print(f"   Total: {total_chars:,} caracteres (~{total_chars//4:,} tokens)")
    
    return combined_path


def create_simple_texts(output_dir: str):
    """Crea textos simples para testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Texto de ejemplo
    example_text = """
La inteligencia artificial es una rama de la ciencia de la computación que busca crear máquinas capaces de imitar comportamientos inteligentes. Desde los primeros días de la computación, los científicos han soñado con crear máquinas que puedan pensar, aprender y resolver problemas como los humanos.

El aprendizaje automático es una subdisciplina de la inteligencia artificial que permite a las máquinas aprender de los datos sin ser explícitamente programadas. Los algoritmos de aprendizaje automático identifican patrones en grandes conjuntos de datos y utilizan estos patrones para hacer predicciones o tomar decisiones.

Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano. Consisten en capas de nodos interconectados que procesan información de manera similar a como lo hacen las neuronas biológicas. Las redes neuronales profundas tienen múltiples capas ocultas que les permiten aprender representaciones cada vez más abstractas de los datos.

El procesamiento del lenguaje natural es un campo de la inteligencia artificial que se enfoca en la interacción entre las computadoras y el lenguaje humano. Los modelos de lenguaje son sistemas que pueden entender, generar y traducir texto en diferentes idiomas.

LLARRI es un modelo de lenguaje experimental que utiliza una arquitectura modular inspirada en la neurociencia. Su diseño incluye módulos especializados para diferentes tipos de procesamiento: lenguaje, lógica, matemáticas, patrones, contexto y creatividad. El sistema del Tálamo coordina estos módulos utilizando reglas explícitas llamadas LLAVES.

La ciencia avanza mediante la observación, la experimentación y el análisis. Los científicos formulan hipótesis, diseñan experimentos para probarlas y analizan los resultados para llegar a conclusiones. El método científico es fundamental para el progreso del conocimiento humano.

La educación es la base del desarrollo individual y social. A través de la educación, las personas adquieren conocimientos, habilidades y valores que les permiten contribuir a la sociedad. La tecnología está transformando la educación, haciendo el aprendizaje más accesible y personalizado.
"""
    
    # Repetir para tener más datos
    full_text = (example_text.strip() + "\n\n") * 100
    
    path = output_dir / "textos_ejemplo.txt"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"✅ Textos de ejemplo guardados: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Descargar corpus para LLARRI')
    parser.add_argument('--output', type=str, default='data/spanish',
                       help='Directorio de salida')
    parser.add_argument('--max_mb', type=int, default=50,
                       help='Máximo tamaño en MB')
    parser.add_argument('--skip_gutenberg', action='store_true',
                       help='No descargar de Gutenberg')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DESCARGA DE CORPUS PARA LLARRI v8")
    print("=" * 60)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear textos de ejemplo
    print("\n📝 Creando textos de ejemplo...")
    create_simple_texts(output_dir)
    
    # Descargar textos en español
    if not args.skip_gutenberg:
        print("\n📚 Descargando corpus en español...")
        max_chars = args.max_mb * 1_000_000
        create_spanish_subset(output_dir, max_chars)
    
    print("\n" + "=" * 60)
    print("DESCARGA COMPLETADA")
    print("=" * 60)
    print(f"\nArchivos guardados en: {output_dir}")
    print("\nPara entrenar con estos datos:")
    print("  python scripts/train_robust.py")


if __name__ == '__main__':
    main()
