"""
demo_ensemble.py - Script de Demostración del Sistema Completo

Este script demuestra el flujo completo del sistema llarri-01:
1. Entrenamiento del modelo base con ViT + TrOCR
2. Fine-tuning de expertos con Adapter/LoRA
3. Entrenamiento del selector de estilo
4. Inferencia con el ensemble completo

Flujo completo:
    Base Model → Expert Fine-tuning → Selector Training → Ensemble Inference
         ↓                ↓                    ↓                  ↓
    ViT+TrOCR     Adapter/LoRA         StyleSelector        Routing

Uso:
    # Demo completo (requiere datos etiquetados)
    python scripts/demo_ensemble.py --mode full
    
    # Solo inferencia (requiere modelos entrenados)
    python scripts/demo_ensemble.py --mode inference --image samples/test.jpg
    
    # Benchmark de velocidad
    python scripts/demo_ensemble.py --mode benchmark
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, List

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
from PIL import Image


def demo_training_pipeline():
    """
    Demostración del pipeline completo de entrenamiento.
    
    Fases:
    1. Entrenar modelo base con IAM/RIMES/custom datasets
    2. Fine-tuning de expertos con datasets específicos
    3. Entrenar selector con datos etiquetados
    """
    print("="*70)
    print("🚀 DEMO: PIPELINE COMPLETO DE ENTRENAMIENTO")
    print("="*70)
    
    # =========================================================================
    # FASE 1: Entrenamiento del Modelo Base
    # =========================================================================
    print("\n" + "="*70)
    print("📘 FASE 1: Entrenamiento del Modelo Base (ViT + TrOCR)")
    print("="*70)
    
    print("""
    El modelo base se entrena en un dataset general de escritura manuscrita:
    - IAM Handwriting Database
    - RIMES (opcional)
    - Bentham (opcional)
    - Datasets custom
    
    Comando de ejemplo:
    """)
    
    cmd_base = """
    python src/llarri/training/train_base.py \\
        --config configs/training.yaml \\
        --model_config configs/base_model.yaml \\
        --data_config configs/data_paths.yaml \\
        --max_epochs 50 \\
        --batch_size 8 \\
        --learning_rate 5e-5 \\
        --output_dir outputs/base_model
    """
    print(cmd_base)
    
    print("""
    Después del entrenamiento:
    - Checkpoint guardado en: outputs/base_model/best.ckpt
    - Logs en: outputs/base_model/logs/
    - Métricas: CER (Character Error Rate), WER (Word Error Rate)
    """)
    
    input("✅ Presiona Enter para continuar con Fase 2...")
    
    # =========================================================================
    # FASE 2: Fine-tuning de Expertos
    # =========================================================================
    print("\n" + "="*70)
    print("🎯 FASE 2: Fine-tuning de Expertos Especializados")
    print("="*70)
    
    print("""
    Creamos expertos especializados para diferentes estilos de escritura:
    
    1. Expert España Mayores (Adapter)
       - Dataset: escritura de adultos mayores españoles
       - Estrategia: Adapter layers (1% parámetros)
       - Especialización: temblor, irregularidad
    
    2. Expert Latam Jóvenes (LoRA)
       - Dataset: escritura de jóvenes latinoamericanos
       - Estrategia: LoRA (0.1% parámetros)
       - Especialización: slang, abreviaturas ("xq", "tmb", "ok")
    """)
    
    # Expert 1: España Mayores
    cmd_expert1 = """
    python src/llarri/training/finetune_expert.py \\
        --base_model outputs/base_model/best.ckpt \\
        --config configs/expert_es_mayores.yaml \\
        --data_path data/splits/es_mayores_train.jsonl \\
        --val_data_path data/splits/es_mayores_val.jsonl \\
        --max_epochs 20 \\
        --batch_size 8 \\
        --learning_rate 1e-4 \\
        --output_dir outputs/experts/es_mayores
    """
    
    # Expert 2: Latam Jóvenes
    cmd_expert2 = """
    python src/llarri/training/finetune_expert.py \\
        --base_model outputs/base_model/best.ckpt \\
        --config configs/expert_latam_jovenes.yaml \\
        --data_path data/splits/latam_jovenes_train.jsonl \\
        --val_data_path data/splits/latam_jovenes_val.jsonl \\
        --max_epochs 20 \\
        --batch_size 8 \\
        --learning_rate 1e-4 \\
        --output_dir outputs/experts/latam_jovenes
    """
    
    print("Comando Expert 1 (España Mayores):")
    print(cmd_expert1)
    print("\nComando Expert 2 (Latam Jóvenes):")
    print(cmd_expert2)
    
    print("""
    Después del fine-tuning:
    - outputs/experts/es_mayores/best.ckpt
    - outputs/experts/latam_jovenes/best.ckpt
    - Métricas específicas por experto
    """)
    
    input("✅ Presiona Enter para continuar con Fase 3...")
    
    # =========================================================================
    # FASE 3: Entrenamiento del Selector de Estilo
    # =========================================================================
    print("\n" + "="*70)
    print("🎨 FASE 3: Entrenamiento del Selector de Estilo")
    print("="*70)
    
    print("""
    El selector aprende a clasificar imágenes por estilo de escritura:
    
    Dataset requerido (JSONL):
    {
        "id": "img_001",
        "image_path": "data/user_samples/sample_001.jpg",
        "text": "Hola mundo",
        "style_label": "es_mayores"  # o 0, 1, 2
    }
    
    Clases:
    - 0: "es_mayores" (España adultos mayores)
    - 1: "latam_jovenes" (Latinoamérica jóvenes)
    - 2: "general" (otros casos)
    """)
    
    cmd_selector = """
    python src/llarri/training/train_selector.py \\
        --data_path data/splits/selector_train.jsonl \\
        --val_data_path data/splits/selector_val.jsonl \\
        --selector_type vit \\
        --num_classes 3 \\
        --vit_model_path outputs/base_model/best.ckpt \\
        --max_epochs 30 \\
        --batch_size 32 \\
        --learning_rate 1e-4 \\
        --output_dir outputs/selector
    """
    
    print("Comando de entrenamiento:")
    print(cmd_selector)
    
    print("""
    Tipos de selector disponibles:
    - cnn: Clasificador CNN ligero (rápido, independiente)
    - vit: Basado en ViT encoder (reutiliza features del base model)
    - multitask: Clasifica múltiples atributos (edad, región, formalidad)
    
    Después del entrenamiento:
    - outputs/selector/best.ckpt
    - Métricas: accuracy, confusion matrix, classification report
    """)
    
    input("✅ Presiona Enter para continuar con Fase 4...")
    
    # =========================================================================
    # FASE 4: Inferencia con Ensemble
    # =========================================================================
    print("\n" + "="*70)
    print("🎯 FASE 4: Inferencia con Sistema Ensemble")
    print("="*70)
    
    print("""
    El sistema ensemble combina todos los componentes:
    
    Flujo de inferencia:
    1. Imagen de entrada → Selector de estilo
    2. Selector predice clase (es_mayores / latam_jovenes / general)
    3. Se selecciona el experto apropiado
    4. Experto genera el texto final
    5. (Opcional) Fallback a modelo base si baja confianza
    """)
    
    print("\nVer función demo_inference() para ejemplo de código completo")
    
    print("\n✅ Demo del pipeline completo finalizado")


def demo_inference():
    """
    Demostración de inferencia con el ensemble completo.
    
    Requiere modelos pre-entrenados.
    """
    print("="*70)
    print("🔮 DEMO: INFERENCIA CON ENSEMBLE")
    print("="*70)
    
    # Verificar que existan los modelos
    required_paths = {
        "selector": "outputs/selector/best.ckpt",
        "base_model": "outputs/base_model/best.ckpt",
        "expert_es_mayores": "outputs/experts/es_mayores/best.ckpt",
        "expert_latam_jovenes": "outputs/experts/latam_jovenes/best.ckpt"
    }
    
    missing = []
    for name, path in required_paths.items():
        if not Path(path).exists():
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("⚠️  Modelos faltantes:")
        for m in missing:
            print(m)
        print("\nPara generar modelos, ejecuta primero: python scripts/demo_ensemble.py --mode full")
        return
    
    print("✅ Todos los modelos encontrados\n")
    
    # Importar ensemble
    from llarri.inference.ensemble import EnsembleInference
    
    # Configuración del ensemble
    ensemble_config = {
        "selector_path": "outputs/selector/best.ckpt",
        "expert_paths": {
            "expert_es_mayores": "outputs/experts/es_mayores/best.ckpt",
            "expert_latam_jovenes": "outputs/experts/latam_jovenes/best.ckpt"
        },
        "base_model_path": "outputs/base_model/best.ckpt",
        "class_to_expert": {
            0: "expert_es_mayores",
            1: "expert_latam_jovenes",
            2: "base"
        },
        "confidence_threshold": 0.7
    }
    
    print("🔧 Inicializando ensemble...")
    ensemble = EnsembleInference(**ensemble_config)
    
    # Ejemplo de inferencia
    print("\n" + "="*70)
    print("📸 EJEMPLO DE INFERENCIA")
    print("="*70)
    
    # Verificar si hay imágenes de ejemplo
    sample_images = list(Path("data/user_samples").glob("*.jpg"))
    if not sample_images:
        print("⚠️  No se encontraron imágenes en data/user_samples/")
        print("   Coloca algunas imágenes de prueba ahí para probar la inferencia")
        return
    
    print(f"\nEncontradas {len(sample_images)} imágenes de ejemplo\n")
    
    # Procesar primera imagen
    test_image = str(sample_images[0])
    print(f"Procesando: {test_image}\n")
    
    result = ensemble.predict(test_image, return_details=True)
    
    print("📊 RESULTADO:")
    print(f"   Texto predicho: '{result['text']}'")
    print(f"   Experto usado: {result['expert_used']}")
    print(f"   Clase detectada: {result['class_name']} (id={result['class_id']})")
    print(f"   Confianza del selector: {result['selector_confidence']:.3f}")
    print(f"   Usó fallback: {result['used_fallback']}")
    
    # Procesar batch si hay más imágenes
    if len(sample_images) > 1:
        print("\n" + "="*70)
        print("📦 PROCESAMIENTO EN BATCH")
        print("="*70)
        
        batch_images = [str(img) for img in sample_images[:5]]  # Max 5
        print(f"\nProcesando batch de {len(batch_images)} imágenes...\n")
        
        batch_results = ensemble.predict_batch(batch_images, batch_size=4)
        
        for i, (img, result) in enumerate(zip(batch_images, batch_results)):
            print(f"{i+1}. {Path(img).name}")
            print(f"   Texto: '{result['text']}'")
            print(f"   Experto: {result['expert_used']} (conf={result['selector_confidence']:.3f})")
    
    print("\n✅ Demo de inferencia completada")


def demo_benchmark():
    """
    Benchmark de velocidad del sistema ensemble.
    
    Compara:
    - Modelo base solo
    - Ensemble con selector
    - Batch processing
    """
    print("="*70)
    print("⏱️  DEMO: BENCHMARK DE VELOCIDAD")
    print("="*70)
    
    print("""
    Este benchmark compara:
    1. Modelo base (sin selector)
    2. Ensemble con routing (selector + expertos)
    3. Batch processing
    
    Métricas:
    - Latencia por imagen (ms)
    - Throughput (imágenes/segundo)
    - Overhead del selector
    """)
    
    print("\n⚠️  Benchmark requiere modelos entrenados y imágenes de prueba")
    print("   Implementación pendiente...")


def demo_quick_start():
    """
    Quick start: entrenar con datos dummy para probar el sistema.
    """
    print("="*70)
    print("🚀 QUICK START: Entrenar con Datos Dummy")
    print("="*70)
    
    print("""
    Este modo genera datos sintéticos para probar el pipeline rápidamente.
    
    Pasos:
    1. Generar datos dummy (texto random + imágenes sintéticas)
    2. Entrenar modelo base (pocas épocas)
    3. Fine-tune un experto
    4. Entrenar selector
    5. Probar ensemble
    
    Tiempo estimado: ~30 min en GPU, ~2h en CPU
    """)
    
    from llarri.training.train_base import train
    from llarri.training.finetune_expert import finetune_expert
    from llarri.training.train_selector import train_selector
    
    # 1. Generar datos dummy
    print("\n" + "-"*70)
    print("1️⃣  Generando datos dummy...")
    print("-"*70)
    
    from scripts.prepare_splits import generate_dummy_data
    
    os.makedirs("data/splits", exist_ok=True)
    generate_dummy_data(
        output_dir="data/splits",
        num_train=100,
        num_val=20,
        num_test=20
    )
    
    print("✅ Datos dummy generados")
    
    # 2. Entrenar modelo base (pocas épocas)
    print("\n" + "-"*70)
    print("2️⃣  Entrenando modelo base (5 épocas)...")
    print("-"*70)
    
    # Simplificado - en realidad llamaría a train() con parámetros
    print("⚠️  Para completar: ejecutar train_base.py con max_epochs=5")
    
    # 3. Fine-tune experto
    print("\n" + "-"*70)
    print("3️⃣  Fine-tuning de experto (3 épocas)...")
    print("-"*70)
    
    print("⚠️  Para completar: ejecutar finetune_expert.py con max_epochs=3")
    
    # 4. Entrenar selector
    print("\n" + "-"*70)
    print("4️⃣  Entrenando selector (10 épocas)...")
    print("-"*70)
    
    print("⚠️  Para completar: ejecutar train_selector.py con max_epochs=10")
    
    # 5. Inferencia
    print("\n" + "-"*70)
    print("5️⃣  Probando ensemble...")
    print("-"*70)
    
    print("⚠️  Para completar: cargar ensemble y hacer inferencia")
    
    print("\n✅ Quick start guide completado")


def main():
    parser = argparse.ArgumentParser(description="Demo del sistema llarri-01")
    
    parser.add_argument(
        '--mode',
        type=str,
        default='training',
        choices=['training', 'inference', 'benchmark', 'quickstart'],
        help='Modo de demostración'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path a imagen para inferencia (modo inference)'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "LLARRI-01 DEMO SYSTEM" + " "*27 + "║")
    print("║" + " "*10 + "OCR Especializado con Selector de Expertos" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")
    
    if args.mode == 'training':
        demo_training_pipeline()
    
    elif args.mode == 'inference':
        if args.image:
            # Inferencia de imagen específica
            print(f"Procesando imagen: {args.image}")
            # TODO: implementar inferencia individual
        else:
            # Demo general de inferencia
            demo_inference()
    
    elif args.mode == 'benchmark':
        demo_benchmark()
    
    elif args.mode == 'quickstart':
        demo_quick_start()
    
    print("\n" + "="*70)
    print("📚 RECURSOS ADICIONALES")
    print("="*70)
    print("""
    Documentación:
    - README.md: Visión general del proyecto
    - configs/: Configuraciones de ejemplo
    - src/llarri/: Código fuente documentado
    
    Comandos útiles:
    - Preparar splits: python scripts/prepare_splits.py
    - Entrenar base: python src/llarri/training/train_base.py
    - Fine-tune expert: python src/llarri/training/finetune_expert.py
    - Entrenar selector: python src/llarri/training/train_selector.py
    
    Contacto:
    - Issues: [URL del repositorio]
    - Docs: [URL de documentación]
    """)
    
    print("\n✨ ¡Gracias por usar llarri-01!\n")


if __name__ == '__main__':
    main()
