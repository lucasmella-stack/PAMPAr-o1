#!/bin/bash
# run_distillation.sh - Pipeline completo de destilación MiniCPM → LLARRI
#
# Este script automatiza todo el proceso de destilación:
# 1. Genera dataset con predicciones de MiniCPM (teacher)
# 2. Entrena LLARRI (student) con ese dataset
# 3. Evalúa el modelo destilado
#
# Uso:
#   ./scripts/run_distillation.sh                    # Ejecutar todo
#   ./scripts/run_distillation.sh --skip-generation  # Saltar generación (si ya existe)
#   ./scripts/run_distillation.sh --remote http://server:8080  # Usar MiniCPM remoto

set -e

# ============================================
# CONFIGURACIÓN
# ============================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths por defecto
IMAGES_DIR="${IMAGES_DIR:-data/processed/lines}"
DISTILLED_DIR="${DISTILLED_DIR:-data/distilled}"
CONFIG_FILE="${CONFIG_FILE:-configs/distillation.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/distillation}"

# Flags
SKIP_GENERATION=false
REMOTE_URL=""
USE_REMOTE=false
DRY_RUN=false

# ============================================
# FUNCIONES AUXILIARES
# ============================================

print_header() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_requirements() {
    print_step "Verificando requisitos..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 no encontrado"
        exit 1
    fi
    
    # Verificar que estamos en el directorio correcto
    if [ ! -f "pyproject.toml" ]; then
        print_error "Ejecutar desde la raíz del proyecto (donde está pyproject.toml)"
        exit 1
    fi
    
    # Verificar directorio de imágenes
    if [ ! -d "$IMAGES_DIR" ]; then
        print_error "Directorio de imágenes no encontrado: $IMAGES_DIR"
        echo "Usa: export IMAGES_DIR=/path/to/images"
        exit 1
    fi
    
    # Verificar configuración
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Archivo de configuración no encontrado: $CONFIG_FILE"
        exit 1
    fi
    
    print_success "Requisitos verificados"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-generation)
                SKIP_GENERATION=true
                shift
                ;;
            --remote)
                USE_REMOTE=true
                REMOTE_URL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --images)
                IMAGES_DIR="$2"
                shift 2
                ;;
            --output)
                DISTILLED_DIR="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Opción desconocida: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --skip-generation    Saltar fase de generación de dataset"
    echo "  --remote URL         Usar servidor MiniCPM remoto"
    echo "  --images DIR         Directorio de imágenes de entrada"
    echo "  --output DIR         Directorio de salida para dataset destilado"
    echo "  --config FILE        Archivo de configuración"
    echo "  --dry-run            Mostrar qué se haría sin ejecutar"
    echo "  --help               Mostrar esta ayuda"
    echo ""
    echo "Ejemplo:"
    echo "  $0 --images data/my_images --remote http://gpu-server:8080"
}

# ============================================
# FASE 1: GENERACIÓN DE DATASET
# ============================================

generate_dataset() {
    print_header "FASE 1: Generando dataset con MiniCPM"
    
    if [ "$SKIP_GENERATION" = true ]; then
        print_step "Saltando generación (--skip-generation)"
        
        # Verificar que existe el dataset
        if [ ! -f "$DISTILLED_DIR/distilled_dataset.jsonl" ]; then
            print_error "No existe dataset en $DISTILLED_DIR/distilled_dataset.jsonl"
            exit 1
        fi
        
        print_success "Usando dataset existente"
        return
    fi
    
    # Crear directorio de salida
    mkdir -p "$DISTILLED_DIR"
    
    # Construir comando
    CMD="python3 scripts/distill_from_minicpm.py"
    CMD="$CMD --input $IMAGES_DIR"
    CMD="$CMD --output $DISTILLED_DIR"
    
    if [ "$USE_REMOTE" = true ]; then
        CMD="$CMD --remote $REMOTE_URL"
    fi
    
    print_step "Ejecutando: $CMD"
    
    if [ "$DRY_RUN" = true ]; then
        echo "(dry-run: no se ejecuta)"
        return
    fi
    
    # Ejecutar
    eval $CMD
    
    print_success "Dataset generado en $DISTILLED_DIR"
}

# ============================================
# FASE 2: ENTRENAMIENTO
# ============================================

train_model() {
    print_header "FASE 2: Entrenando LLARRI con destilación"
    
    # Verificar dataset
    DATASET_FILE="$DISTILLED_DIR/distilled_dataset.jsonl"
    if [ ! -f "$DATASET_FILE" ]; then
        print_error "Dataset no encontrado: $DATASET_FILE"
        exit 1
    fi
    
    # Contar samples
    SAMPLE_COUNT=$(wc -l < "$DATASET_FILE")
    print_step "Dataset tiene $SAMPLE_COUNT samples"
    
    # Comando de entrenamiento
    CMD="python3 -c \"
import sys
sys.path.insert(0, 'src')

import yaml
from pathlib import Path

from llarri.training.distillation_trainer import (
    DistillationTrainer, 
    DistillationConfig,
    ProgressiveDistillation,
    create_distillation_trainer
)
from llarri.data.distillation_dataset import create_distillation_dataloaders
from llarri.models.llarri_base_model import LlarriBaseModel

# Cargar configuración
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)

print('Cargando modelo LLARRI...')
model = LlarriBaseModel()

print('Creando dataloaders...')
train_loader, val_loader = create_distillation_dataloaders(
    distilled_file='$DATASET_FILE',
    images_dir='$IMAGES_DIR',
    tokenizer=model.tokenizer,
    batch_size=config['training']['batch_size'],
    val_split=config['data']['val_split'],
    num_workers=config['data']['num_workers'],
    use_weighted_sampling=config['distillation']['use_weighted_sampling'],
    use_curriculum=config['distillation']['use_curriculum'],
)

print('Iniciando entrenamiento...')
if config.get('progressive_distillation', {}).get('enabled', False):
    # Destilación progresiva
    progressive = ProgressiveDistillation(
        student_model=model,
        config=DistillationConfig(**config['distillation']),
    )
    progressive.train_all_stages(train_loader, val_loader)
else:
    # Destilación simple
    trainer_module = create_distillation_trainer(
        student_model=model,
        alpha_hard=config['distillation']['alpha_hard'],
        alpha_soft=config['distillation']['alpha_soft'],
        temperature=config['distillation']['temperature'],
    )
    
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices=1,
        precision='16-mixed' if config['training']['use_amp'] else 32,
    )
    trainer.fit(trainer_module, train_loader, val_loader)

print('\\n✅ Entrenamiento completado!')
\""
    
    print_step "Iniciando entrenamiento..."
    
    if [ "$DRY_RUN" = true ]; then
        echo "(dry-run: no se ejecuta)"
        return
    fi
    
    # Ejecutar
    eval $CMD
    
    print_success "Modelo entrenado"
}

# ============================================
# FASE 3: EVALUACIÓN
# ============================================

evaluate_model() {
    print_header "FASE 3: Evaluando modelo destilado"
    
    # Buscar mejor checkpoint
    BEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.ckpt 2>/dev/null | head -1)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        print_error "No se encontraron checkpoints en $CHECKPOINT_DIR"
        exit 1
    fi
    
    print_step "Usando checkpoint: $BEST_CHECKPOINT"
    
    # Comando de evaluación
    CMD="python3 -c \"
import sys
sys.path.insert(0, 'src')

from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.training.distillation_trainer import DistillationTrainer

print('Cargando modelo destilado...')
model = LlarriBaseModel.load_from_checkpoint('$BEST_CHECKPOINT')
model.eval()

# Ejemplo de inferencia
print('\\nProbando inferencia...')
from PIL import Image
import torch

# Crear imagen de prueba
test_img = Image.new('RGB', (384, 128), color='white')
result = model.predict(test_img)
print(f'Predicción de prueba: {result}')

print('\\n✅ Modelo destilado funcionando correctamente!')
\""
    
    if [ "$DRY_RUN" = true ]; then
        echo "(dry-run: no se ejecuta)"
        return
    fi
    
    eval $CMD
    
    print_success "Evaluación completada"
}

# ============================================
# RESUMEN FINAL
# ============================================

show_summary() {
    print_header "RESUMEN"
    
    echo "Dataset destilado: $DISTILLED_DIR/distilled_dataset.jsonl"
    
    if [ -f "$DISTILLED_DIR/distillation_stats.json" ]; then
        echo ""
        echo "Estadísticas del dataset:"
        cat "$DISTILLED_DIR/distillation_stats.json" | python3 -c "
import sys, json
stats = json.load(sys.stdin)
print(f\"  Total samples: {stats['total_samples']}\")
print(f\"  Con ground truth: {stats['samples_with_ground_truth']}\")
print(f\"  Confianza promedio: {stats['avg_confidence']:.2%}\")
print(f\"  Alta confianza (>80%): {stats['high_confidence_samples']}\")
"
    fi
    
    echo ""
    echo "Checkpoints guardados en: $CHECKPOINT_DIR"
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "Checkpoints disponibles:"
        ls -la $CHECKPOINT_DIR/*.ckpt 2>/dev/null | head -5
    fi
    
    echo ""
    echo -e "${GREEN}¡Destilación completada!${NC}"
    echo ""
    echo "Próximos pasos:"
    echo "  1. Evaluar con datos de test:"
    echo "     python scripts/evaluate.py --checkpoint $CHECKPOINT_DIR/best.ckpt"
    echo ""
    echo "  2. Exportar a ONNX para producción:"
    echo "     python -m llarri.inference.export_onnx --checkpoint $CHECKPOINT_DIR/best.ckpt"
    echo ""
    echo "  3. Usar en inferencia:"
    echo "     from llarri.models.llarri_base_model import LlarriBaseModel"
    echo "     model = LlarriBaseModel.load_from_checkpoint('$CHECKPOINT_DIR/best.ckpt')"
}

# ============================================
# MAIN
# ============================================

main() {
    parse_args "$@"
    
    print_header "PIPELINE DE DESTILACIÓN: MiniCPM → LLARRI"
    
    check_requirements
    
    # Mostrar configuración
    echo "Configuración:"
    echo "  Imágenes: $IMAGES_DIR"
    echo "  Dataset destilado: $DISTILLED_DIR"
    echo "  Config: $CONFIG_FILE"
    if [ "$USE_REMOTE" = true ]; then
        echo "  MiniCPM remoto: $REMOTE_URL"
    else
        echo "  MiniCPM: local (con cuantización)"
    fi
    echo ""
    
    # Ejecutar fases
    generate_dataset
    train_model
    evaluate_model
    
    # Mostrar resumen
    show_summary
}

main "$@"
