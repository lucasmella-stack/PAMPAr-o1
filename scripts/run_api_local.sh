#!/bin/bash
# =============================================================================
# run_api_local.sh - Ejecutar API llarri-01 localmente
# =============================================================================
#
# Este script inicia la API FastAPI para desarrollo local.
#
# Uso:
#   ./scripts/run_api_local.sh              # Modo desarrollo con reload
#   ./scripts/run_api_local.sh --prod       # Modo producción
#   ./scripts/run_api_local.sh --workers 4  # Múltiples workers
#
# Variables de entorno:
#   LLARRI_BASE_MODEL    - Path al modelo base (default: outputs/base_model/best.ckpt)
#   LLARRI_SELECTOR      - Path al selector (default: outputs/selector/best.ckpt)
#   LLARRI_EXPERTS_DIR   - Dir de expertos (default: outputs/experts)
#   PORT                 - Puerto (default: 8000)
#   HOST                 - Host (default: 0.0.0.0)
#   DEBUG                - Debug mode (default: false)
#
# =============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio del proyecto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          llarri-01 OCR API - Local Development            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Parsear argumentos
# =============================================================================
RELOAD="--reload"
WORKERS="1"
LOG_LEVEL="info"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|--production)
            RELOAD=""
            LOG_LEVEL="warning"
            shift
            ;;
        --workers)
            WORKERS="$2"
            RELOAD=""  # No reload con múltiples workers
            shift 2
            ;;
        --debug)
            LOG_LEVEL="debug"
            export DEBUG=true
            shift
            ;;
        --help|-h)
            echo "Uso: $0 [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --prod          Modo producción (sin reload)"
            echo "  --workers N     Número de workers (default: 1)"
            echo "  --debug         Modo debug con logs detallados"
            echo "  --help          Mostrar esta ayuda"
            echo ""
            echo "Variables de entorno:"
            echo "  PORT=8000       Puerto de la API"
            echo "  HOST=0.0.0.0    Host de la API"
            exit 0
            ;;
        *)
            echo -e "${RED}Opción desconocida: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configurar variables de entorno
# =============================================================================
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export DEBUG="${DEBUG:-false}"

# Paths de modelos (configurables)
export LLARRI_BASE_MODEL="${LLARRI_BASE_MODEL:-outputs/base_model/best.ckpt}"
export LLARRI_SELECTOR="${LLARRI_SELECTOR:-outputs/selector/best.ckpt}"
export LLARRI_EXPERTS_DIR="${LLARRI_EXPERTS_DIR:-outputs/experts}"

echo -e "${YELLOW}📋 Configuración:${NC}"
echo "   Host: $HOST"
echo "   Puerto: $PORT"
echo "   Workers: $WORKERS"
echo "   Reload: $([ -n "$RELOAD" ] && echo "Sí" || echo "No")"
echo "   Debug: $DEBUG"
echo ""

# =============================================================================
# Verificar dependencias
# =============================================================================
echo -e "${YELLOW}🔍 Verificando dependencias...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python no encontrado${NC}"
    exit 1
fi

if ! python -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}❌ uvicorn no instalado. Ejecutar: pip install uvicorn${NC}"
    exit 1
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}❌ FastAPI no instalado. Ejecutar: pip install fastapi${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dependencias OK${NC}"
echo ""

# =============================================================================
# Verificar modelos (advertencias si no existen)
# =============================================================================
echo -e "${YELLOW}📂 Verificando modelos...${NC}"

if [ -f "$LLARRI_BASE_MODEL" ]; then
    echo -e "   ${GREEN}✓${NC} Modelo base: $LLARRI_BASE_MODEL"
else
    echo -e "   ${YELLOW}⚠${NC} Modelo base no encontrado: $LLARRI_BASE_MODEL"
fi

if [ -f "$LLARRI_SELECTOR" ]; then
    echo -e "   ${GREEN}✓${NC} Selector: $LLARRI_SELECTOR"
else
    echo -e "   ${YELLOW}⚠${NC} Selector no encontrado: $LLARRI_SELECTOR"
fi

if [ -d "$LLARRI_EXPERTS_DIR" ]; then
    expert_count=$(find "$LLARRI_EXPERTS_DIR" -name "best.ckpt" 2>/dev/null | wc -l)
    echo -e "   ${GREEN}✓${NC} Expertos: $expert_count encontrados en $LLARRI_EXPERTS_DIR"
else
    echo -e "   ${YELLOW}⚠${NC} Directorio de expertos no encontrado: $LLARRI_EXPERTS_DIR"
fi

echo ""

# =============================================================================
# Verificar GPU
# =============================================================================
echo -e "${YELLOW}🎮 Verificando GPU...${NC}"

if python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())" 2>/dev/null; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "   ${GREEN}✓${NC} GPU detectada: $GPU_NAME"
    else
        echo -e "   ${YELLOW}⚠${NC} GPU no disponible, usando CPU"
    fi
else
    echo -e "   ${YELLOW}⚠${NC} PyTorch no detectado"
fi

echo ""

# =============================================================================
# Iniciar servidor
# =============================================================================
echo -e "${GREEN}🚀 Iniciando servidor...${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "   API disponible en: ${GREEN}http://${HOST}:${PORT}${NC}"
echo -e "   Documentación:     ${GREEN}http://${HOST}:${PORT}/docs${NC}"
echo -e "   Health check:      ${GREEN}http://${HOST}:${PORT}/health${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Presiona Ctrl+C para detener el servidor${NC}"
echo ""

# Agregar src al PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Ejecutar uvicorn
if [ "$WORKERS" == "1" ]; then
    # Single worker (permite reload)
    python -m uvicorn llarri.api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        $RELOAD
else
    # Múltiples workers (sin reload)
    python -m uvicorn llarri.api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --workers "$WORKERS"
fi

