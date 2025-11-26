#!/usr/bin/env python3
"""
prepare_splits.py - Prepara los archivos JSONL de train/val/test

Este script:
1. Lee imágenes y transcripciones de diferentes datasets
2. Unifica el formato a JSONL
3. Divide en train/val/test
4. Genera los archivos en data/splits/

Uso:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --data-dir data/external --output-dir data/splits
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_iam_dataset(iam_path: str) -> List[Dict]:
    """
    Carga dataset IAM.
    
    Estructura esperada:
    iam/
        lines/
            a01/
                a01-000u/
                    a01-000u-00.png
                    ...
        ascii/
            lines.txt
    """
    samples = []
    lines_file = os.path.join(iam_path, "ascii", "lines.txt")
    
    if not os.path.exists(lines_file):
        print(f"⚠️ IAM: archivo lines.txt no encontrado en {lines_file}")
        return samples
    
    with open(lines_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignorar comentarios
            if line.startswith('#') or not line:
                continue
            
            parts = line.split(' ')
            if len(parts) < 9:
                continue
            
            line_id = parts[0]
            # El texto está después del 8vo campo
            text = ' '.join(parts[8:]).replace('|', ' ')
            
            # Construir path de imagen
            # a01-000u-00 -> a01/a01-000u/a01-000u-00.png
            parts_id = line_id.split('-')
            if len(parts_id) >= 3:
                folder1 = parts_id[0]
                folder2 = f"{parts_id[0]}-{parts_id[1]}"
                image_path = f"lines/{folder1}/{folder2}/{line_id}.png"
                
                full_image_path = os.path.join(iam_path, image_path)
                if os.path.exists(full_image_path):
                    samples.append({
                        "id": f"iam_{line_id}",
                        "image_path": image_path,
                        "text": text,
                        "source": "iam",
                        "language": "en"
                    })
    
    print(f"✅ IAM: {len(samples)} muestras cargadas")
    return samples


def load_generic_jsonl(jsonl_path: str, source: str) -> List[Dict]:
    """Carga datos desde un archivo JSONL existente."""
    samples = []
    
    if not os.path.exists(jsonl_path):
        print(f"⚠️ {source}: archivo no encontrado en {jsonl_path}")
        return samples
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                item["source"] = source
                if "id" not in item:
                    item["id"] = f"{source}_{len(samples)}"
                samples.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"✅ {source}: {len(samples)} muestras cargadas")
    return samples


def load_image_text_pairs(directory: str, source: str) -> List[Dict]:
    """
    Carga pares imagen-texto desde un directorio.
    
    Espera:
    - imagen.png y imagen.txt en el mismo directorio
    O
    - images/ con imágenes y labels.txt con transcripciones
    """
    samples = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"⚠️ {source}: directorio no encontrado en {directory}")
        return samples
    
    # Método 1: imagen.png + imagen.txt
    for img_file in directory.rglob("*.png"):
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            rel_path = img_file.relative_to(directory)
            samples.append({
                "id": f"{source}_{img_file.stem}",
                "image_path": str(rel_path),
                "text": text,
                "source": source
            })
    
    # Método 2: labels.txt con formato "imagen.png|texto"
    labels_file = directory / "labels.txt"
    if labels_file.exists():
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    img_name, text = line.split('|', 1)
                    img_path = directory / img_name.strip()
                    if img_path.exists():
                        samples.append({
                            "id": f"{source}_{Path(img_name).stem}",
                            "image_path": img_name.strip(),
                            "text": text.strip(),
                            "source": source
                        })
    
    print(f"✅ {source}: {len(samples)} muestras cargadas")
    return samples


def split_data(
    samples: List[Dict], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Divide datos en train/val/test."""
    random.seed(seed)
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]
    
    return train, val, test


def save_jsonl(samples: List[Dict], output_path: str):
    """Guarda muestras en formato JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"💾 Guardado: {output_path} ({len(samples)} muestras)")


def generate_dummy_data(output_dir: str, num_samples: int = 100):
    """Genera datos dummy para testing."""
    samples = []
    
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "El veloz murciélago hindú comía feliz cardillo y kiwi",
        "La cigüeña tocaba el saxofón detrás del palenque de paja",
        "Jovencillo emponzoñado de whisky: ¡qué figurota exhibe!",
        "Benjamín pidió una bebida de kiwi y fresa",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        "Hello World",
        "Testing 1234567890",
    ]
    
    for i in range(num_samples):
        samples.append({
            "id": f"dummy_{i:04d}",
            "image_path": f"dummy/image_{i:04d}.png",
            "text": random.choice(dummy_texts),
            "source": "dummy"
        })
    
    train, val, test = split_data(samples)
    
    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(output_dir, "test.jsonl"))
    
    print(f"\n✅ Datos dummy generados:")
    print(f"   Train: {len(train)}")
    print(f"   Val: {len(val)}")
    print(f"   Test: {len(test)}")


def prepare_splits(
    data_dir: str = "data/external",
    output_dir: str = "data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Función principal para preparar splits de datos.
    """
    all_samples = []
    
    # Cargar datasets disponibles
    datasets_to_load = [
        ("iam", lambda: load_iam_dataset(os.path.join(data_dir, "iam"))),
        ("rimes", lambda: load_generic_jsonl(os.path.join(data_dir, "rimes", "data.jsonl"), "rimes")),
        ("bentham", lambda: load_generic_jsonl(os.path.join(data_dir, "bentham", "data.jsonl"), "bentham")),
        ("custom_es", lambda: load_image_text_pairs(os.path.join(data_dir, "custom_es"), "custom_es")),
    ]
    
    for name, loader in datasets_to_load:
        try:
            samples = loader()
            all_samples.extend(samples)
        except Exception as e:
            print(f"⚠️ Error cargando {name}: {e}")
    
    if not all_samples:
        print("\n⚠️ No se encontraron datos. Generando datos dummy para testing...")
        generate_dummy_data(output_dir)
        return
    
    # Estadísticas por fuente
    print("\n📊 Estadísticas de datos:")
    sources = defaultdict(int)
    for sample in all_samples:
        sources[sample.get("source", "unknown")] += 1
    
    for source, count in sorted(sources.items()):
        print(f"   {source}: {count} muestras")
    
    print(f"   Total: {len(all_samples)} muestras")
    
    # Dividir datos
    train, val, test = split_data(all_samples, train_ratio, val_ratio, seed)
    
    # Guardar splits
    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(output_dir, "test.jsonl"))
    
    print(f"\n✅ Splits generados:")
    print(f"   Train: {len(train)} ({100*len(train)/len(all_samples):.1f}%)")
    print(f"   Val: {len(val)} ({100*len(val)/len(all_samples):.1f}%)")
    print(f"   Test: {len(test)} ({100*len(test)/len(all_samples):.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara archivos JSONL de train/val/test"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data/external",
        help="Directorio con datos externos"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/splits",
        help="Directorio de salida para splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio de datos para training"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio de datos para validación"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidad"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generar datos dummy para testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.dummy:
        generate_dummy_data(args.output_dir)
    else:
        prepare_splits(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

