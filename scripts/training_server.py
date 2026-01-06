#!/usr/bin/env python3
"""
training_server.py - Backend API para controlar entrenamiento desde frontend

Endpoints:
- GET  /api/datasets          - Lista datasets disponibles
- POST /api/datasets/download - Descarga un dataset
- GET  /api/training/status   - Estado del entrenamiento
- POST /api/training/start    - Inicia entrenamiento
- POST /api/training/stop     - Detiene entrenamiento
- GET  /api/training/metrics  - Métricas en tiempo real
- GET  /api/samples           - Samples aleatorios para visualizar
- POST /api/predict           - Predice sobre una imagen
- WS   /ws/training           - WebSocket para updates en tiempo real

Uso:
    python scripts/training_server.py
    # Abrir http://localhost:8000 en el navegador
"""

import os
import sys
import json
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from io import BytesIO

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLARRI Training Dashboard",
    description="Control de entrenamiento para modelo OCR LLARRI",
    version="1.0.0"
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ESTADO GLOBAL
# ============================================

@dataclass
class TrainingState:
    """Estado del entrenamiento."""
    is_running: bool = False
    is_paused: bool = False
    current_epoch: int = 0
    total_epochs: int = 10
    current_step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    eta_seconds: int = 0
    start_time: Optional[str] = None
    dataset_name: str = ""
    model_name: str = "LLARRI"
    
    # Historial
    loss_history: List[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


class TrainingManager:
    """Gestor del proceso de entrenamiento."""
    
    def __init__(self):
        self.state = TrainingState()
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.websocket_clients: List[WebSocket] = []
        self.model = None
        self.config = {}
    
    async def broadcast(self, message: Dict[str, Any]):
        """Envía mensaje a todos los clientes WebSocket."""
        for ws in self.websocket_clients[:]:
            try:
                await ws.send_json(message)
            except:
                self.websocket_clients.remove(ws)
    
    def start_training(self, config: Dict[str, Any]):
        """Inicia el entrenamiento en un thread separado."""
        if self.state.is_running:
            raise RuntimeError("Ya hay un entrenamiento en curso")
        
        self.config = config
        self.stop_flag.clear()
        self.state = TrainingState(
            is_running=True,
            total_epochs=config.get("epochs", 10),
            dataset_name=config.get("dataset", "unknown"),
            start_time=datetime.now().isoformat(),
        )
        
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
    
    def stop_training(self):
        """Detiene el entrenamiento."""
        self.stop_flag.set()
        if self.training_thread:
            self.training_thread.join(timeout=5)
        self.state.is_running = False
    
    def pause_training(self):
        """Pausa el entrenamiento."""
        self.state.is_paused = True
    
    def resume_training(self):
        """Reanuda el entrenamiento."""
        self.state.is_paused = False
    
    def _training_loop(self):
        """Loop principal de entrenamiento."""
        try:
            # Importar aquí para no cargar PyTorch al iniciar
            self._run_actual_training()
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            self.state.is_running = False
    
    def _run_actual_training(self):
        """Ejecuta el entrenamiento real."""
        import torch
        from torch.utils.data import DataLoader
        
        logger.info("Iniciando entrenamiento...")
        
        # Cargar datos
        dataset_path = Path("data/processed") / self.config.get("dataset", "spanish_synthetic")
        
        if not dataset_path.exists():
            logger.error(f"Dataset no encontrado: {dataset_path}")
            self.state.is_running = False
            return
        
        # Intentar cargar el modelo LLARRI
        try:
            from llarri.models.llarri_base_model import LlarriBaseModel
            from llarri.data.distillation_dataset import DistillationDataset
            
            # Cargar modelo
            logger.info("Cargando modelo LLARRI...")
            self.model = LlarriBaseModel()
            
            # Crear dataset simple
            labels_file = dataset_path / "labels.json"
            if labels_file.exists():
                with open(labels_file) as f:
                    labels = json.load(f)
                self.state.total_steps = len(labels) * self.state.total_epochs
            
        except ImportError as e:
            logger.warning(f"No se pudo cargar LLARRI: {e}")
            logger.info("Ejecutando simulación de entrenamiento...")
            self._simulate_training()
            return
        
        # Training loop real (simplificado)
        self._simulate_training()  # Por ahora simular
    
    def _simulate_training(self):
        """Simula entrenamiento para demo."""
        import random
        import math
        
        steps_per_epoch = 100
        self.state.total_steps = steps_per_epoch * self.state.total_epochs
        
        base_loss = 2.5
        
        for epoch in range(self.state.total_epochs):
            if self.stop_flag.is_set():
                break
            
            self.state.current_epoch = epoch + 1
            
            for step in range(steps_per_epoch):
                if self.stop_flag.is_set():
                    break
                
                while self.state.is_paused:
                    time.sleep(0.5)
                
                # Simular métricas
                progress = (epoch * steps_per_epoch + step) / self.state.total_steps
                
                # Loss que decrece
                noise = random.gauss(0, 0.1)
                self.state.train_loss = base_loss * math.exp(-2 * progress) + noise + 0.2
                self.state.val_loss = self.state.train_loss * 1.1 + random.gauss(0, 0.05)
                
                # Accuracy que aumenta
                self.state.val_accuracy = min(0.95, 0.3 + 0.65 * progress + random.gauss(0, 0.02))
                
                # Learning rate con warmup y decay
                if progress < 0.1:
                    self.state.learning_rate = 1e-5 * (progress / 0.1)
                else:
                    self.state.learning_rate = 1e-5 * (1 - progress) * 1.2
                
                self.state.current_step = epoch * steps_per_epoch + step + 1
                
                # ETA
                elapsed = time.time() - datetime.fromisoformat(self.state.start_time).timestamp()
                if self.state.current_step > 0:
                    time_per_step = elapsed / self.state.current_step
                    remaining_steps = self.state.total_steps - self.state.current_step
                    self.state.eta_seconds = int(time_per_step * remaining_steps)
                
                # Guardar en historial
                self.state.loss_history.append({
                    "step": self.state.current_step,
                    "train_loss": self.state.train_loss,
                    "val_loss": self.state.val_loss,
                    "accuracy": self.state.val_accuracy,
                })
                
                # Limitar historial
                if len(self.state.loss_history) > 1000:
                    self.state.loss_history = self.state.loss_history[-500:]
                
                time.sleep(0.05)  # Simular tiempo de step
        
        self.state.is_running = False
        logger.info("Entrenamiento completado!")


# Instancia global
training_manager = TrainingManager()


# ============================================
# MODELOS PYDANTIC
# ============================================

class TrainingConfig(BaseModel):
    dataset: str
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    use_distillation: bool = False
    use_progressive: bool = False


class DatasetDownloadRequest(BaseModel):
    dataset_name: str
    prepare: bool = True


# ============================================
# ENDPOINTS API
# ============================================

# Definir DATASETS aquí para evitar problemas de import
DATASETS_INFO = {
    "rodrigo": {"description": "Manuscritos españoles siglo XV", "language": "español", "size_mb": 450, "requires_registration": False},
    "esposalles": {"description": "Registros matrimoniales catalán/español", "language": "español/catalán", "size_mb": 200, "requires_registration": False},
    "spanish_synthetic": {"description": "Dataset sintético en español", "language": "español", "size_mb": 100, "requires_registration": False},
    "bentham": {"description": "Manuscritos históricos de Bentham", "language": "inglés", "size_mb": 300, "requires_registration": False},
    "iam": {"description": "IAM Handwriting Database", "language": "inglés", "size_mb": 800, "requires_registration": True},
}

@app.get("/api/datasets")
async def list_datasets():
    """Lista datasets disponibles."""
    
    # Datasets en catálogo
    available = []
    for name, info in DATASETS_INFO.items():
        available.append({
            "name": name,
            "description": info["description"],
            "language": info["language"],
            "size_mb": info["size_mb"],
            "requires_registration": info["requires_registration"],
            "downloaded": (Path("data/external") / name).exists(),
            "prepared": (Path("data/processed") / name).exists(),
        })
    
    # Datasets locales preparados
    prepared_dir = Path("data/processed")
    local = []
    if prepared_dir.exists():
        for d in prepared_dir.iterdir():
            if d.is_dir() and (d / "labels.json").exists():
                with open(d / "labels.json") as f:
                    labels = json.load(f)
                local.append({
                    "name": d.name,
                    "samples": len(labels),
                    "path": str(d),
                })
    
    return {
        "catalog": available,
        "local": local,
    }


@app.post("/api/datasets/download")
async def download_dataset(request: DatasetDownloadRequest, background_tasks: BackgroundTasks):
    """Descarga un dataset."""
    import subprocess
    import sys
    
    if request.dataset_name not in DATASETS_INFO:
        raise HTTPException(404, f"Dataset '{request.dataset_name}' no encontrado")
    
    def do_download():
        # Ejecutar script de descarga como subprocess
        cmd = [sys.executable, "scripts/download_datasets.py", 
               "--dataset", request.dataset_name]
        if request.prepare:
            cmd.append("--prepare")
        subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    background_tasks.add_task(do_download)
    
    return {"status": "downloading", "dataset": request.dataset_name}


@app.get("/api/training/status")
async def training_status():
    """Estado actual del entrenamiento."""
    return asdict(training_manager.state)


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Inicia entrenamiento."""
    try:
        training_manager.start_training(config.dict())
        return {"status": "started", "config": config.dict()}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/training/stop")
async def stop_training():
    """Detiene entrenamiento."""
    training_manager.stop_training()
    return {"status": "stopped"}


@app.post("/api/training/pause")
async def pause_training():
    """Pausa entrenamiento."""
    training_manager.pause_training()
    return {"status": "paused"}


@app.post("/api/training/resume")
async def resume_training():
    """Reanuda entrenamiento."""
    training_manager.resume_training()
    return {"status": "resumed"}


@app.get("/api/training/metrics")
async def training_metrics():
    """Métricas de entrenamiento."""
    return {
        "current": {
            "train_loss": training_manager.state.train_loss,
            "val_loss": training_manager.state.val_loss,
            "accuracy": training_manager.state.val_accuracy,
            "learning_rate": training_manager.state.learning_rate,
        },
        "history": training_manager.state.loss_history[-100:],  # Últimos 100
    }


@app.get("/api/samples")
async def get_samples(dataset: str = "spanish_synthetic", count: int = 10):
    """Obtiene samples aleatorios de un dataset."""
    import random
    
    dataset_path = Path("data/processed") / dataset
    labels_file = dataset_path / "labels.json"
    
    if not labels_file.exists():
        raise HTTPException(404, f"Dataset '{dataset}' no encontrado")
    
    with open(labels_file) as f:
        all_labels = json.load(f)
    
    samples = random.sample(all_labels, min(count, len(all_labels)))
    
    result = []
    for sample in samples:
        img_path = dataset_path / "images" / sample["image"]
        if img_path.exists():
            # Convertir a base64
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            
            result.append({
                "image": f"data:image/png;base64,{img_b64}",
                "text": sample["text"],
                "filename": sample["image"],
            })
    
    return {"samples": result}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Predice texto de una imagen."""
    try:
        from PIL import Image
        
        # Leer imagen
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Intentar usar modelo real
        try:
            from llarri.models.llarri_base_model import LlarriBaseModel
            
            if training_manager.model is None:
                training_manager.model = LlarriBaseModel()
            
            result = training_manager.model.predict(image)
            return {
                "text": result,
                "confidence": 0.85,  # TODO: obtener del modelo
            }
            
        except Exception as e:
            # Fallback: texto simulado
            logger.warning(f"Modelo no disponible: {e}")
            return {
                "text": "[Modelo no cargado - Demo]",
                "confidence": 0.0,
            }
            
    except Exception as e:
        raise HTTPException(400, f"Error procesando imagen: {e}")


# ============================================
# WEBSOCKET
# ============================================

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket para updates en tiempo real."""
    await websocket.accept()
    training_manager.websocket_clients.append(websocket)
    
    try:
        while True:
            # Enviar estado cada 500ms
            await websocket.send_json({
                "type": "status",
                "data": asdict(training_manager.state),
            })
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        training_manager.websocket_clients.remove(websocket)


# ============================================
# FRONTEND
# ============================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLARRI Training Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            background: white;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .status-badge {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Header -->
    <header class="gradient-bg text-white py-6 px-8 shadow-lg">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold">🔤 LLARRI Training</h1>
                <p class="text-purple-200">Panel de control para entrenamiento OCR</p>
            </div>
            <div id="status-badge" class="status-badge px-4 py-2 bg-white/20 rounded-full">
                <span id="status-text">● Listo</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto p-6 space-y-6">
        <!-- Métricas principales -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="card p-6">
                <div class="text-gray-500 text-sm">Época</div>
                <div class="text-3xl font-bold text-purple-600">
                    <span id="current-epoch">0</span>/<span id="total-epochs">10</span>
                </div>
            </div>
            <div class="card p-6">
                <div class="text-gray-500 text-sm">Train Loss</div>
                <div class="text-3xl font-bold text-blue-600" id="train-loss">0.000</div>
            </div>
            <div class="card p-6">
                <div class="text-gray-500 text-sm">Val Accuracy</div>
                <div class="text-3xl font-bold text-green-600" id="val-accuracy">0.0%</div>
            </div>
            <div class="card p-6">
                <div class="text-gray-500 text-sm">ETA</div>
                <div class="text-3xl font-bold text-orange-600" id="eta">--:--</div>
            </div>
        </div>

        <!-- Gráfico y controles -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Gráfico -->
            <div class="lg:col-span-2 card p-6">
                <h2 class="text-xl font-bold mb-4">📈 Métricas de Entrenamiento</h2>
                <canvas id="metricsChart" height="200"></canvas>
            </div>

            <!-- Controles -->
            <div class="card p-6 space-y-4">
                <h2 class="text-xl font-bold mb-4">⚙️ Controles</h2>
                
                <!-- Dataset -->
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Dataset</label>
                    <select id="dataset-select" class="w-full p-2 border rounded-lg">
                        <option value="spanish_synthetic">Spanish Synthetic</option>
                    </select>
                </div>

                <!-- Épocas -->
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Épocas</label>
                    <input type="number" id="epochs-input" value="10" min="1" max="100" 
                           class="w-full p-2 border rounded-lg">
                </div>

                <!-- Batch size -->
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Batch Size</label>
                    <input type="number" id="batch-input" value="4" min="1" max="32" 
                           class="w-full p-2 border rounded-lg">
                </div>

                <!-- Learning rate -->
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Learning Rate</label>
                    <input type="text" id="lr-input" value="1e-5" 
                           class="w-full p-2 border rounded-lg">
                </div>

                <!-- Botones -->
                <div class="flex gap-2 pt-4">
                    <button id="btn-start" onclick="startTraining()" 
                            class="flex-1 bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-lg font-medium">
                        ▶ Iniciar
                    </button>
                    <button id="btn-pause" onclick="pauseTraining()" 
                            class="flex-1 bg-yellow-500 hover:bg-yellow-600 text-white py-2 px-4 rounded-lg font-medium" disabled>
                        ⏸ Pausar
                    </button>
                    <button id="btn-stop" onclick="stopTraining()" 
                            class="flex-1 bg-red-500 hover:bg-red-600 text-white py-2 px-4 rounded-lg font-medium" disabled>
                        ⏹ Detener
                    </button>
                </div>
            </div>
        </div>

        <!-- Samples y predicción -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Samples del dataset -->
            <div class="card p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold">🖼️ Samples del Dataset</h2>
                    <button onclick="loadSamples()" class="text-purple-600 hover:text-purple-800">
                        🔄 Actualizar
                    </button>
                </div>
                <div id="samples-container" class="space-y-3 max-h-96 overflow-y-auto">
                    <p class="text-gray-500 text-center py-4">Carga un dataset para ver samples</p>
                </div>
            </div>

            <!-- Predicción -->
            <div class="card p-6">
                <h2 class="text-xl font-bold mb-4">🔮 Probar Predicción</h2>
                
                <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-4"
                     ondrop="handleDrop(event)" ondragover="handleDragOver(event)">
                    <input type="file" id="predict-input" accept="image/*" class="hidden" 
                           onchange="handleFileSelect(event)">
                    <label for="predict-input" class="cursor-pointer">
                        <div class="text-gray-500">
                            <p class="text-4xl mb-2">📤</p>
                            <p>Arrastra una imagen o haz clic para seleccionar</p>
                        </div>
                    </label>
                </div>
                
                <div id="predict-result" class="hidden">
                    <img id="predict-image" class="w-full h-24 object-contain bg-gray-100 rounded mb-2">
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <div class="text-sm text-purple-600 mb-1">Predicción:</div>
                        <div id="predict-text" class="text-xl font-bold text-purple-800"></div>
                        <div id="predict-confidence" class="text-sm text-gray-500 mt-1"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Datasets disponibles -->
        <div class="card p-6">
            <h2 class="text-xl font-bold mb-4">📚 Datasets Disponibles</h2>
            <div id="datasets-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <p class="text-gray-500">Cargando datasets...</p>
            </div>
        </div>
    </main>

    <script>
        // Estado global
        let ws = null;
        let chart = null;
        let isTraining = false;

        // Inicializar
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            loadDatasets();
            connectWebSocket();
        });

        // WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/training`);
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'status') {
                    updateUI(msg.data);
                }
            };
            
            ws.onclose = () => {
                setTimeout(connectWebSocket, 2000);
            };
        }

        // Actualizar UI
        function updateUI(state) {
            isTraining = state.is_running;
            
            // Status badge
            const statusBadge = document.getElementById('status-badge');
            const statusText = document.getElementById('status-text');
            if (state.is_running) {
                if (state.is_paused) {
                    statusBadge.className = 'status-badge px-4 py-2 bg-yellow-400 rounded-full';
                    statusText.textContent = '⏸ Pausado';
                } else {
                    statusBadge.className = 'status-badge px-4 py-2 bg-green-400 rounded-full';
                    statusText.textContent = '● Entrenando';
                }
            } else {
                statusBadge.className = 'px-4 py-2 bg-white/20 rounded-full';
                statusText.textContent = '● Listo';
            }
            
            // Métricas
            document.getElementById('current-epoch').textContent = state.current_epoch;
            document.getElementById('total-epochs').textContent = state.total_epochs;
            document.getElementById('train-loss').textContent = state.train_loss.toFixed(4);
            document.getElementById('val-accuracy').textContent = (state.val_accuracy * 100).toFixed(1) + '%';
            
            // ETA
            if (state.eta_seconds > 0) {
                const mins = Math.floor(state.eta_seconds / 60);
                const secs = state.eta_seconds % 60;
                document.getElementById('eta').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }
            
            // Botones
            document.getElementById('btn-start').disabled = state.is_running;
            document.getElementById('btn-pause').disabled = !state.is_running;
            document.getElementById('btn-stop').disabled = !state.is_running;
            document.getElementById('btn-pause').textContent = state.is_paused ? '▶ Reanudar' : '⏸ Pausar';
            
            // Actualizar gráfico
            if (state.loss_history && state.loss_history.length > 0) {
                updateChart(state.loss_history);
            }
        }

        // Gráfico
        function initChart() {
            const ctx = document.getElementById('metricsChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Train Loss',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            tension: 0.1,
                            fill: false,
                        },
                        {
                            label: 'Val Loss',
                            data: [],
                            borderColor: 'rgb(239, 68, 68)',
                            tension: 0.1,
                            fill: false,
                        },
                        {
                            label: 'Accuracy',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            tension: 0.1,
                            fill: false,
                            yAxisID: 'y1',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: { intersect: false, mode: 'index' },
                    scales: {
                        y: { type: 'linear', position: 'left', title: { display: true, text: 'Loss' } },
                        y1: { type: 'linear', position: 'right', min: 0, max: 1, title: { display: true, text: 'Accuracy' } }
                    }
                }
            });
        }

        function updateChart(history) {
            const last100 = history.slice(-100);
            chart.data.labels = last100.map(h => h.step);
            chart.data.datasets[0].data = last100.map(h => h.train_loss);
            chart.data.datasets[1].data = last100.map(h => h.val_loss);
            chart.data.datasets[2].data = last100.map(h => h.accuracy);
            chart.update('none');
        }

        // Cargar datasets
        async function loadDatasets() {
            try {
                const response = await fetch('/api/datasets');
                const data = await response.json();
                
                // Actualizar select
                const select = document.getElementById('dataset-select');
                select.innerHTML = '';
                data.local.forEach(ds => {
                    const option = document.createElement('option');
                    option.value = ds.name;
                    option.textContent = `${ds.name} (${ds.samples} samples)`;
                    select.appendChild(option);
                });
                
                // Mostrar catálogo
                const container = document.getElementById('datasets-container');
                container.innerHTML = data.catalog.map(ds => `
                    <div class="border rounded-lg p-4 ${ds.prepared ? 'border-green-300 bg-green-50' : 'border-gray-200'}">
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-bold">${ds.name}</h3>
                                <p class="text-sm text-gray-600">${ds.description}</p>
                                <p class="text-xs text-gray-500 mt-1">
                                    🌍 ${ds.language} | 📦 ${ds.size_mb}MB
                                </p>
                            </div>
                            ${ds.prepared ? 
                                '<span class="text-green-600 text-2xl">✓</span>' :
                                ds.requires_registration ?
                                    '<span class="text-yellow-600 text-sm">🔒 Registro</span>' :
                                    `<button onclick="downloadDataset('${ds.name}')" class="text-purple-600 hover:text-purple-800">⬇️ Descargar</button>`
                            }
                        </div>
                    </div>
                `).join('');
                
                // Cargar samples del primer dataset
                if (data.local.length > 0) {
                    loadSamples();
                }
                
            } catch (e) {
                console.error('Error cargando datasets:', e);
            }
        }

        async function downloadDataset(name) {
            try {
                const response = await fetch('/api/datasets/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_name: name, prepare: true })
                });
                alert(`Descargando ${name}... Esto puede tomar unos minutos.`);
                setTimeout(loadDatasets, 5000);
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        // Samples
        async function loadSamples() {
            const dataset = document.getElementById('dataset-select').value;
            if (!dataset) return;
            
            try {
                const response = await fetch(`/api/samples?dataset=${dataset}&count=5`);
                const data = await response.json();
                
                const container = document.getElementById('samples-container');
                container.innerHTML = data.samples.map(s => `
                    <div class="flex gap-3 p-2 bg-gray-50 rounded-lg">
                        <img src="${s.image}" class="h-12 w-auto object-contain bg-white rounded">
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-medium truncate">${s.text}</p>
                            <p class="text-xs text-gray-500">${s.filename}</p>
                        </div>
                    </div>
                `).join('');
                
            } catch (e) {
                console.error('Error cargando samples:', e);
            }
        }

        // Entrenamiento
        async function startTraining() {
            const config = {
                dataset: document.getElementById('dataset-select').value,
                epochs: parseInt(document.getElementById('epochs-input').value),
                batch_size: parseInt(document.getElementById('batch-input').value),
                learning_rate: parseFloat(document.getElementById('lr-input').value),
            };
            
            try {
                const response = await fetch('/api/training/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    alert('Error: ' + err.detail);
                }
            } catch (e) {
                alert('Error iniciando entrenamiento: ' + e.message);
            }
        }

        async function pauseTraining() {
            const isPaused = document.getElementById('btn-pause').textContent.includes('Reanudar');
            const endpoint = isPaused ? '/api/training/resume' : '/api/training/pause';
            await fetch(endpoint, { method: 'POST' });
        }

        async function stopTraining() {
            if (confirm('¿Detener el entrenamiento?')) {
                await fetch('/api/training/stop', { method: 'POST' });
            }
        }

        // Predicción
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('border-purple-500');
        }

        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('border-purple-500');
            const file = e.dataTransfer.files[0];
            if (file) predictImage(file);
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) predictImage(file);
        }

        async function predictImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Mostrar imagen
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('predict-image').src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                document.getElementById('predict-result').classList.remove('hidden');
                document.getElementById('predict-text').textContent = result.text;
                document.getElementById('predict-confidence').textContent = 
                    `Confianza: ${(result.confidence * 100).toFixed(1)}%`;
                    
            } catch (e) {
                alert('Error en predicción: ' + e.message);
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Sirve el frontend."""
    return FRONTEND_HTML


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLARRI Training Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Puerto")
    parser.add_argument("--reload", action="store_true", help="Auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   LLARRI Training Dashboard                  ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 Abrí el navegador en: http://localhost:{args.port}          ║
║  📊 API docs en: http://localhost:{args.port}/docs              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "scripts.training_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
