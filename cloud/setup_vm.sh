#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# PAMPAr-o1 - Script de setup para VM de GCP

set -e

echo "=========================================="
echo "🧠 PAMPAr-o1 - Setup de entrenamiento GCP"
echo "=========================================="

# Actualizar sistema
echo "📦 Actualizando sistema..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl htop tmux

# Instalar NVIDIA drivers si no están
if ! command -v nvidia-smi &> /dev/null; then
    echo "🎮 Instalando drivers NVIDIA..."
    sudo apt-get install -y -qq nvidia-driver-535
fi

# Verificar GPU
echo "🖥️ Verificando GPU..."
nvidia-smi

# Instalar Miniconda si no está
if [ ! -d "$HOME/miniconda3" ]; then
    echo "🐍 Instalando Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
fi

# Activar conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Crear entorno
echo "🔧 Creando entorno conda..."
conda create -n pampar python=3.11 -y
conda activate pampar

# Instalar PyTorch con CUDA
echo "🔥 Instalando PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clonar repositorio
if [ ! -d "PAMPAr-o1" ]; then
    echo "📂 Clonando repositorio..."
    git clone https://github.com/lucasmella-stack/PAMPAr-o1.git
fi

cd PAMPAr-o1

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt
pip install google-cloud-storage tensorboard datasets huggingface_hub

# Crear directorios
mkdir -p data/corpus checkpoints logs

# Verificar instalación
echo "✅ Verificando instalación..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "✅ Setup completado!"
echo "=========================================="
echo ""
echo "Para entrenar ejecuta:"
echo "  conda activate pampar"
echo "  cd PAMPAr-o1"
echo "  python cloud/train_cloud.py --gpu t4 --hours 100"
echo ""
echo "Recomendación: usa tmux para sesiones persistentes"
echo "  tmux new -s training"
echo ""
