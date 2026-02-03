# SPDX-License-Identifier: AGPL-3.0-or-later
# PAMPAr-o1 Training Container for GCP
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias adicionales para cloud
RUN pip install --no-cache-dir \
    google-cloud-storage \
    tensorboard \
    wandb \
    datasets \
    huggingface_hub

# Copiar código del proyecto
COPY pampar/ ./pampar/
COPY scripts/ ./scripts/
COPY cloud/ ./cloud/
COPY data/tokenizer/ ./data/tokenizer/

# Crear directorios
RUN mkdir -p /app/data/corpus /app/checkpoints /app/logs

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Script de entrada
ENTRYPOINT ["python", "cloud/train_cloud.py"]
