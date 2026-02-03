#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-o1 - Script para crear y gestionar VM en GCP
====================================================

Uso:
    python cloud/gcp_launcher.py create --gpu t4     # Crear VM con T4
    python cloud/gcp_launcher.py create --gpu l4     # Crear VM con L4 (mejor precio/rendimiento)
    python cloud/gcp_launcher.py ssh                 # Conectar por SSH
    python cloud/gcp_launcher.py delete              # Eliminar VM
    python cloud/gcp_launcher.py status              # Ver estado y costos
"""

import argparse
import subprocess
import sys
import json
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# CONFIGURACIÓN DE VMs
# ============================================================================

@dataclass
class VMConfig:
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    disk_size_gb: int
    cost_per_hour: float
    vram_gb: int

VM_CONFIGS = {
    "t4": VMConfig(
        machine_type="n1-standard-4",
        accelerator_type="nvidia-tesla-t4",
        accelerator_count=1,
        disk_size_gb=100,
        cost_per_hour=0.35,
        vram_gb=16,
    ),
    "l4": VMConfig(
        machine_type="g2-standard-4",
        accelerator_type="nvidia-l4",
        accelerator_count=1,
        disk_size_gb=100,
        cost_per_hour=0.81,
        vram_gb=24,
    ),
    "v100": VMConfig(
        machine_type="n1-standard-8",
        accelerator_type="nvidia-tesla-v100",
        accelerator_count=1,
        disk_size_gb=200,
        cost_per_hour=2.48,
        vram_gb=16,
    ),
    "a100": VMConfig(
        machine_type="a2-highgpu-1g",
        accelerator_type="nvidia-tesla-a100",
        accelerator_count=1,
        disk_size_gb=200,
        cost_per_hour=3.67,
        vram_gb=40,
    ),
}

PROJECT_ID = "pampar-o1"
ZONE = "europe-west4-b"
VM_NAME = "pampar-training"
BUCKET_NAME = "pampar-checkpoints"

# ============================================================================
# FUNCIONES
# ============================================================================

def run_cmd(cmd: str, capture: bool = False) -> Optional[str]:
    """Ejecutar comando gcloud."""
    print(f"🔧 {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    else:
        subprocess.run(cmd, shell=True)
        return None


def create_bucket():
    """Crear bucket de GCS para checkpoints."""
    print("\n📦 Creando bucket de GCS...")
    run_cmd(f"gcloud storage buckets create gs://{BUCKET_NAME} --location=europe-west4 --project={PROJECT_ID} 2>$null")
    print(f"✅ Bucket: gs://{BUCKET_NAME}")


def create_vm(gpu_type: str, preemptible: bool = True):
    """Crear VM con GPU."""
    
    if gpu_type not in VM_CONFIGS:
        print(f"❌ GPU no soportada: {gpu_type}")
        print(f"   Opciones: {list(VM_CONFIGS.keys())}")
        return
    
    cfg = VM_CONFIGS[gpu_type]
    
    print(f"\n{'='*60}")
    print(f"🚀 Creando VM: {VM_NAME}")
    print(f"{'='*60}")
    print(f"🖥️  GPU: {gpu_type.upper()} ({cfg.vram_gb}GB VRAM)")
    print(f"💻 Máquina: {cfg.machine_type}")
    print(f"💾 Disco: {cfg.disk_size_gb}GB")
    print(f"💰 Costo: ${cfg.cost_per_hour}/hora" + (" (preemptible ~60% menos)" if preemptible else ""))
    print(f"📍 Zona: {ZONE}")
    
    # Crear bucket primero
    create_bucket()
    
    # Construir comando
    cmd = f"""gcloud compute instances create {VM_NAME} \
        --project={PROJECT_ID} \
        --zone={ZONE} \
        --machine-type={cfg.machine_type} \
        --accelerator=type={cfg.accelerator_type},count={cfg.accelerator_count} \
        --boot-disk-size={cfg.disk_size_gb}GB \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --scopes=cloud-platform"""
    
    if preemptible:
        cmd += " --provisioning-model=SPOT"
    
    print(f"\n🔧 Creando instancia...")
    run_cmd(cmd)
    
    print(f"\n✅ VM creada: {VM_NAME}")
    print(f"\n📋 Próximos pasos:")
    print(f"   1. Conectar: python cloud/gcp_launcher.py ssh")
    print(f"   2. En la VM ejecutar:")
    print(f"      curl -sL https://raw.githubusercontent.com/lucasmella-stack/PAMPAr-o1/main/cloud/setup_vm.sh | bash")
    print(f"   3. Entrenar:")
    print(f"      tmux new -s train")
    print(f"      conda activate pampar && cd PAMPAr-o1")
    print(f"      python cloud/train_cloud.py --gpu {gpu_type} --hours 100 --bucket {BUCKET_NAME}")


def ssh_to_vm():
    """Conectar por SSH a la VM."""
    print(f"\n🔗 Conectando a {VM_NAME}...")
    run_cmd(f"gcloud compute ssh {VM_NAME} --zone={ZONE} --project={PROJECT_ID}")


def delete_vm():
    """Eliminar VM."""
    print(f"\n🗑️ Eliminando VM: {VM_NAME}")
    run_cmd(f"gcloud compute instances delete {VM_NAME} --zone={ZONE} --project={PROJECT_ID} --quiet")
    print("✅ VM eliminada")


def stop_vm():
    """Detener VM (no elimina, solo para)."""
    print(f"\n⏸️ Deteniendo VM: {VM_NAME}")
    run_cmd(f"gcloud compute instances stop {VM_NAME} --zone={ZONE} --project={PROJECT_ID}")
    print("✅ VM detenida")


def start_vm():
    """Iniciar VM detenida."""
    print(f"\n▶️ Iniciando VM: {VM_NAME}")
    run_cmd(f"gcloud compute instances start {VM_NAME} --zone={ZONE} --project={PROJECT_ID}")
    print("✅ VM iniciada")


def status():
    """Ver estado de la VM y costos."""
    print(f"\n📊 Estado del proyecto: {PROJECT_ID}")
    print("="*60)
    
    # Estado de la VM
    result = run_cmd(f"gcloud compute instances describe {VM_NAME} --zone={ZONE} --project={PROJECT_ID} --format=json 2>$null", capture=True)
    
    if result and "error" not in result.lower():
        try:
            data = json.loads(result)
            status = data.get("status", "UNKNOWN")
            machine = data.get("machineType", "").split("/")[-1]
            
            print(f"🖥️  VM: {VM_NAME}")
            print(f"   Estado: {status}")
            print(f"   Tipo: {machine}")
            
            # Buscar GPU
            for acc in data.get("guestAccelerators", []):
                gpu = acc.get("acceleratorType", "").split("/")[-1]
                count = acc.get("acceleratorCount", 1)
                print(f"   GPU: {gpu} x{count}")
        except:
            print("❌ No se pudo obtener estado de la VM")
    else:
        print(f"🖥️  VM: {VM_NAME} - NO EXISTE")
    
    # Contenido del bucket
    print(f"\n☁️  Bucket: gs://{BUCKET_NAME}")
    run_cmd(f"gcloud storage ls gs://{BUCKET_NAME}/ 2>$null || echo '   (vacío o no existe)'")
    
    # Estimación de costos
    print(f"\n💰 Estimación de costos ($255 disponibles):")
    for gpu, cfg in VM_CONFIGS.items():
        hours = 255 / cfg.cost_per_hour
        tokens = hours * (15_000_000 if gpu == "t4" else 40_000_000 if gpu == "v100" else 50_000_000 if gpu == "l4" else 100_000_000)
        print(f"   {gpu.upper()}: {hours:.0f} horas, ~{tokens/1e9:.1f}B tokens")


def print_recommendations():
    """Imprimir recomendaciones de uso."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              🧠 PAMPAr-o1 - Recomendaciones GCP                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CON $255 DE CRÉDITO:                                           ║
║                                                                  ║
║  🥇 RECOMENDADO: L4 (mejor precio/rendimiento)                  ║
║     - 315 horas disponibles                                     ║
║     - ~15B tokens procesables                                   ║
║     - 24GB VRAM (modelo más grande)                            ║
║     - Comando: python cloud/gcp_launcher.py create --gpu l4    ║
║                                                                  ║
║  🥈 ALTERNATIVA: T4 (más económico)                             ║
║     - 728 horas disponibles                                     ║
║     - ~11B tokens procesables                                   ║
║     - 16GB VRAM                                                 ║
║     - Comando: python cloud/gcp_launcher.py create --gpu t4    ║
║                                                                  ║
║  ⚡ RÁPIDO: V100 (entrenamiento intensivo)                      ║
║     - 102 horas disponibles                                     ║
║     - ~4B tokens procesables                                    ║
║     - Comando: python cloud/gcp_launcher.py create --gpu v100  ║
║                                                                  ║
║  TIPS:                                                          ║
║  • Usa VMs preemptibles (--preemptible) para 60-70% descuento  ║
║  • Guarda checkpoints en GCS frecuentemente                     ║
║  • Usa tmux para sesiones persistentes                          ║
║  • Detén la VM cuando no entrenes (stop, no delete)            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PAMPAr-o1 GCP Launcher")
    subparsers = parser.add_subparsers(dest="command", help="Comandos")
    
    # Create
    create_parser = subparsers.add_parser("create", help="Crear VM")
    create_parser.add_argument("--gpu", type=str, default="l4",
                               choices=["t4", "l4", "v100", "a100"],
                               help="Tipo de GPU (default: l4)")
    create_parser.add_argument("--no-preemptible", action="store_true",
                               help="No usar VM preemptible")
    
    # SSH
    subparsers.add_parser("ssh", help="Conectar por SSH")
    
    # Delete
    subparsers.add_parser("delete", help="Eliminar VM")
    
    # Stop
    subparsers.add_parser("stop", help="Detener VM")
    
    # Start
    subparsers.add_parser("start", help="Iniciar VM")
    
    # Status
    subparsers.add_parser("status", help="Ver estado")
    
    # Recommend
    subparsers.add_parser("recommend", help="Ver recomendaciones")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_vm(args.gpu, not args.no_preemptible)
    elif args.command == "ssh":
        ssh_to_vm()
    elif args.command == "delete":
        delete_vm()
    elif args.command == "stop":
        stop_vm()
    elif args.command == "start":
        start_vm()
    elif args.command == "status":
        status()
    elif args.command == "recommend":
        print_recommendations()
    else:
        print_recommendations()
        parser.print_help()


if __name__ == "__main__":
    main()
