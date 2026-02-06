#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Script para actualizar el repositorio de HuggingFace con los archivos del modelo.
"""
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, whoami

def update_huggingface_repo():
    """Actualiza el repositorio de HuggingFace con los archivos del modelo."""
    
    # Verificar autenticación
    try:
        user_info = whoami()
        print(f"✓ Autenticado como: {user_info['name']}")
    except Exception as e:
        print(f"✗ Error de autenticación: {e}")
        print("Ejecuta: huggingface-cli login")
        return
    
    # Configuración
    repo_id = "lucas-mella/PAMPAr-o1"
    local_dir = Path("c:/Users/lucas/Documents/Be Web/PampaR-o1")
    
    # Crear API
    api = HfApi()
    
    # Verificar si el repo existe
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✓ Repositorio encontrado: {repo_id}")
    except Exception as e:
        print(f"✗ Repositorio no encontrado. Creándolo...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=False)
            print(f"✓ Repositorio creado: {repo_id}")
        except Exception as create_error:
            print(f"✗ Error al crear repositorio: {create_error}")
            return
    
    print(f"\n📤 Subiendo archivos actualizados...")
    
    # Archivos a subir
    files_to_upload = [
        ("PAMPAR-coder.png", "PAMPAR-coder.png"),
        ("README.md", "README.md"),
        ("docs/huggingface/MODEL_CARD.md", "MODEL_CARD.md"),
        ("docs/huggingface/MODEL_CARD.es.md", "MODEL_CARD.es.md"),
        ("LICENSE", "LICENSE"),
        ("diagrams/PampaR_Architecture.html", "diagrams/PampaR_Architecture.html"),
        ("diagrams/PampaR_Arquitectura.html", "diagrams/PampaR_Arquitectura.html"),
        ("diagrams/PampaR_Benchmark.html", "diagrams/PampaR_Benchmark.html"),
        ("diagrams/PampaR_Benchmark_ES.html", "diagrams/PampaR_Benchmark_ES.html"),
    ]
    
    # Subir archivos
    uploaded = []
    for src, dst in files_to_upload:
        src_path = local_dir / src
        
        if src_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(src_path),
                    path_in_repo=dst,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update {dst} with new branding"
                )
                print(f"  ✓ {src} → {dst}")
                uploaded.append(dst)
            except Exception as e:
                print(f"  ✗ Error subiendo {src}: {e}")
        else:
            print(f"  ⚠ {src} no encontrado")
    
    # Subir checkpoints si existen (archivos grandes con LFS)
    checkpoint_files = [
        ("checkpoints/pampar_v9_best.pt", "pampar_v9_best.pt"),
        ("data/tokenizer/llarri_bpe.model", "llarri_bpe.model"),
        ("data/tokenizer/llarri_bpe.vocab", "llarri_bpe.vocab"),
    ]
    
    for src, dst in checkpoint_files:
        src_path = local_dir / src
        if src_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(src_path),
                    path_in_repo=dst,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {dst}"
                )
                print(f"  ✓ {src} → {dst}")
                uploaded.append(dst)
            except Exception as e:
                print(f"  ⚠ {src}: {e}")
    
    print(f"\n✓ {len(uploaded)} archivos subidos exitosamente")
    print(f"🔗 Ver repositorio: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    update_huggingface_repo()
