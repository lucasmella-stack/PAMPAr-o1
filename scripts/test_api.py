"""
test_api.py - Tests y ejemplos de uso de la API

Script para probar la API localmente con ejemplos de requests.

Uso:
    # Probar health check
    python scripts/test_api.py --health
    
    # Probar predicción
    python scripts/test_api.py --predict --image path/to/image.jpg
    
    # Probar batch
    python scripts/test_api.py --batch --images img1.jpg img2.jpg img3.jpg
    
    # Probar estilo
    python scripts/test_api.py --style --image path/to/image.jpg
    
    # Benchmark de latencia
    python scripts/test_api.py --benchmark --image path/to/image.jpg --n 100
"""

import os
import sys
import argparse
import time
import base64
import json
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# URL base de la API
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def encode_image(image_path: str) -> str:
    """Codifica una imagen a base64."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def test_health():
    """Prueba el endpoint de health check."""
    print("🔍 Testing /health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Uptime: {data['uptime_seconds']:.1f}s")
        print(f"   Models loaded: {data['models_loaded']}")
        print(f"   GPU available: {data['gpu_available']}")
        
        if data.get('gpu_memory_used_mb'):
            print(f"   GPU memory: {data['gpu_memory_used_mb']:.1f}MB")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API")
        print(f"   Asegúrate de que la API esté corriendo en {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_models():
    """Lista los modelos disponibles."""
    print("📋 Testing /models endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Modelos disponibles: {data['total_loaded']}")
        print(f"   Default: {data['default_model']}")
        print()
        
        for model in data['models']:
            status_icon = "✓" if model['status'] == 'loaded' else "✗"
            print(f"   {status_icon} {model['name']} ({model['type']})")
            print(f"      OCR: {model['supports_ocr']}, Style: {model['supports_style_classification']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict(image_path: str, model_name: Optional[str] = None):
    """Prueba predicción de una imagen."""
    print(f"🔤 Testing /predict endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Imagen no encontrada: {image_path}")
        return False
    
    try:
        # Codificar imagen
        image_b64 = encode_image(image_path)
        
        # Request
        payload = {
            "image_base64": image_b64,
            "model_name": model_name,
            "max_length": 128,
            "num_beams": 4,
            "return_confidence": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60
        )
        latency = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Predicción exitosa")
        print(f"   Texto: \"{data['text']}\"")
        
        if data.get('confidence'):
            print(f"   Confianza: {data['confidence']:.3f}")
        
        if data.get('model_used'):
            print(f"   Modelo usado: {data['model_used']}")
        
        print(f"   Latencia: {latency:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch(image_paths: list, model_name: Optional[str] = None):
    """Prueba predicción batch."""
    print(f"📦 Testing /predict/batch endpoint with {len(image_paths)} images...")
    
    try:
        # Codificar imágenes
        images_b64 = []
        for path in image_paths:
            if not Path(path).exists():
                print(f"⚠️  Imagen no encontrada: {path}")
                continue
            images_b64.append(encode_image(path))
        
        if not images_b64:
            print("❌ No se encontraron imágenes válidas")
            return False
        
        # Request
        payload = {
            "images": images_b64,
            "model_name": model_name,
            "max_length": 128,
            "num_beams": 4
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            timeout=120
        )
        latency = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Batch completado")
        print(f"   Total: {data['total_images']}")
        print(f"   Exitosas: {data['successful']}")
        print(f"   Fallidas: {data['failed']}")
        print(f"   Tiempo total: {data['total_processing_time_ms']:.1f}ms")
        print(f"   Latencia total: {latency:.1f}ms")
        print()
        
        print("   Resultados:")
        for i, pred in enumerate(data['predictions']):
            print(f"   {i+1}. \"{pred['text'][:50]}...\" (conf: {pred.get('confidence', 'N/A')})")
        
        if data.get('errors'):
            print()
            print("   Errores:")
            for err in data['errors']:
                print(f"   - Index {err['index']}: {err['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_style(image_path: str):
    """Prueba clasificación de estilo."""
    print(f"🎨 Testing /predict/style endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Imagen no encontrada: {image_path}")
        return False
    
    try:
        # Codificar imagen
        image_b64 = encode_image(image_path)
        
        # Request
        payload = {
            "image_base64": image_b64,
            "return_probabilities": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/style",
            json=payload,
            timeout=30
        )
        latency = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Clasificación de estilo exitosa")
        print(f"   Estilo: {data['style_class']} (id: {data['style_id']})")
        print(f"   Confianza: {data['confidence']:.3f}")
        print(f"   Latencia: {latency:.1f}ms")
        
        if data.get('probabilities'):
            print()
            print("   Probabilidades:")
            for style, prob in sorted(data['probabilities'].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"   {style:20s} {bar} {prob:.3f}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            print("⚠️  Selector de estilo no disponible")
        else:
            print(f"❌ Error HTTP: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_benchmark(image_path: str, n_requests: int = 100, model_name: Optional[str] = None):
    """Benchmark de latencia."""
    print(f"⏱️  Running benchmark: {n_requests} requests...")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Imagen no encontrada: {image_path}")
        return False
    
    try:
        # Codificar imagen una vez
        image_b64 = encode_image(image_path)
        
        payload = {
            "image_base64": image_b64,
            "model_name": model_name,
            "max_length": 128,
            "num_beams": 4
        }
        
        latencies = []
        errors = 0
        
        print(f"   Enviando {n_requests} requests...")
        
        for i in range(n_requests):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if (i + 1) % 10 == 0:
                    print(f"   {i+1}/{n_requests} completados...")
                    
            except Exception as e:
                errors += 1
        
        # Calcular estadísticas
        import statistics
        
        print()
        print(f"✅ Benchmark completado")
        print(f"   Total requests: {n_requests}")
        print(f"   Exitosos: {len(latencies)}")
        print(f"   Errores: {errors}")
        print()
        print(f"   Latencia (ms):")
        print(f"      Min:    {min(latencies):.1f}")
        print(f"      Max:    {max(latencies):.1f}")
        print(f"      Mean:   {statistics.mean(latencies):.1f}")
        print(f"      Median: {statistics.median(latencies):.1f}")
        print(f"      Std:    {statistics.stdev(latencies):.1f}")
        print()
        print(f"   Throughput: {len(latencies) / (sum(latencies) / 1000):.2f} req/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_test_image(output_path: str = "test_image.png"):
    """Crea una imagen de prueba simple."""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Dibujar líneas simulando escritura
    for y in [30, 50, 70]:
        points = []
        for x in range(50, 350, 10):
            import random
            noise = random.randint(-5, 5)
            points.append((x, y + noise))
        draw.line(points, fill='black', width=2)
    
    img.save(output_path)
    print(f"✅ Imagen de prueba creada: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test API llarri-01")
    
    parser.add_argument('--api-url', type=str, default=None,
                       help='URL de la API (default: http://localhost:8000)')
    
    # Acciones
    parser.add_argument('--health', action='store_true',
                       help='Test health check')
    parser.add_argument('--models', action='store_true',
                       help='List available models')
    parser.add_argument('--predict', action='store_true',
                       help='Test single prediction')
    parser.add_argument('--batch', action='store_true',
                       help='Test batch prediction')
    parser.add_argument('--style', action='store_true',
                       help='Test style classification')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run latency benchmark')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    
    # Parámetros
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for testing')
    parser.add_argument('--images', nargs='+', default=None,
                       help='Paths to images for batch testing')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to use')
    parser.add_argument('--n', type=int, default=100,
                       help='Number of requests for benchmark')
    parser.add_argument('--create-test-image', action='store_true',
                       help='Create a test image')
    
    args = parser.parse_args()
    
    # Configurar URL
    global API_BASE_URL
    if args.api_url:
        API_BASE_URL = args.api_url
    
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║              llarri-01 API Tester                         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print(f"🌐 API URL: {API_BASE_URL}")
    print()
    
    # Crear imagen de prueba si se solicita
    if args.create_test_image:
        create_test_image()
        return
    
    # Ejecutar tests
    success = True
    
    if args.health or args.all:
        success = test_health() and success
        print()
    
    if args.models or args.all:
        success = test_models() and success
        print()
    
    if args.predict or args.all:
        image = args.image or "test_image.png"
        if not Path(image).exists() and args.all:
            image = create_test_image()
        success = test_predict(image, args.model) and success
        print()
    
    if args.batch:
        images = args.images or [args.image] if args.image else ["test_image.png"]
        success = test_batch(images, args.model) and success
        print()
    
    if args.style:
        image = args.image or "test_image.png"
        success = test_style(image) and success
        print()
    
    if args.benchmark:
        image = args.image or "test_image.png"
        success = run_benchmark(image, args.n, args.model) and success
        print()
    
    # Si no se especificó ninguna acción, mostrar ayuda
    if not any([args.health, args.models, args.predict, args.batch, 
                args.style, args.benchmark, args.all]):
        parser.print_help()
        print()
        print("Ejemplos:")
        print("  python scripts/test_api.py --health")
        print("  python scripts/test_api.py --predict --image samples/test.jpg")
        print("  python scripts/test_api.py --all --create-test-image")
    
    print()
    if success:
        print("✅ Todos los tests pasaron")
    else:
        print("❌ Algunos tests fallaron")
        sys.exit(1)


if __name__ == "__main__":
    main()
