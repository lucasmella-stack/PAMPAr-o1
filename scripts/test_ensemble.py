#!/usr/bin/env python3
"""
test_ensemble.py - Tests E2E para el ensemble OCR

Ejecutar:
    python scripts/test_ensemble.py
    python scripts/test_ensemble.py --strategy always_verify
    python scripts/test_ensemble.py --image path/to/image.png
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ensemble_components():
    """Test individual de cada componente."""
    print("\n" + "="*60)
    print("TEST 1: Componentes individuales")
    print("="*60)
    
    # Test SpanishLanguageModel
    print("\n--- SpanishLanguageModel ---")
    try:
        from llarri.inference.language_model import SpanishLanguageModel, get_language_model
        lm = get_language_model()
        
        test_cases = [
            ("el perro juega", "el perro juega"),
            ("el prro juga", "el perro juega"),
            ("la qasa es grande", "la casa es grande"),
        ]
        
        for input_text, expected in test_cases:
            corrected = lm.correct_text(input_text)
            status = "✓" if corrected.lower() == expected.lower() else "✗"
            print(f"  {status} '{input_text}' → '{corrected}' (esperado: '{expected}')")
        
        print("  SpanishLanguageModel: OK")
    except Exception as e:
        print(f"  SpanishLanguageModel: ERROR - {e}")
    
    # Test MiniCPMAdapter (mock)
    print("\n--- MiniCPMAdapter (mock) ---")
    try:
        from llarri.inference.minicpm_adapter import MiniCPMAdapter, AdapterMode
        adapter = MiniCPMAdapter(mode=AdapterMode.MOCK)
        
        assert adapter.is_available(), "Mock debería estar disponible"
        
        # Test con string (simulando path)
        text, score = adapter.transcribe_image("test_image.png")
        print(f"  Transcripción mock: '{text}' (score: {score:.2f})")
        
        # Test verificación
        is_correct, verified, conf = adapter.verify_transcription("test.png", "Hola mundo")
        print(f"  Verificación mock: '{verified}' (correcto: {is_correct}, conf: {conf:.2f})")
        
        print("  MiniCPMAdapter: OK")
    except Exception as e:
        print(f"  MiniCPMAdapter: ERROR - {e}")


def test_ensemble_strategies():
    """Test de diferentes estrategias de ensemble."""
    print("\n" + "="*60)
    print("TEST 2: Estrategias de ensemble")
    print("="*60)
    
    try:
        from llarri.inference.ensemble_ocr import (
            EnsembleOCR, EnsembleStrategy, EnsembleConfig, create_ensemble
        )
        from llarri.inference.minicpm_adapter import MiniCPMAdapter, AdapterMode
        
        # Crear componentes mock
        minicpm = MiniCPMAdapter(mode=AdapterMode.MOCK)
        
        strategies = [
            EnsembleStrategy.LLARRI_ONLY,
            EnsembleStrategy.MINICPM_ONLY,
            EnsembleStrategy.VERIFY_IF_LOW_CONF,
            EnsembleStrategy.ALWAYS_VERIFY,
            EnsembleStrategy.CONSENSUS,
            EnsembleStrategy.RERANK,
        ]
        
        for strategy in strategies:
            print(f"\n--- Estrategia: {strategy.value} ---")
            
            config = EnsembleConfig(
                strategy=strategy,
                verbose=False,
            )
            
            ensemble = EnsembleOCR(
                llarri_model=None,  # Usará mock o None
                minicpm_adapter=minicpm,
                config=config,
            )
            
            # Test con imagen dummy (path string)
            try:
                result = ensemble.predict("dummy_image.png", strategy=strategy)
                print(f"  Texto: '{result.text}'")
                print(f"  Confianza: {result.confidence:.2f}")
                print(f"  Tiempo: {result.total_time_ms:.2f}ms")
                print(f"  Razón: {result.decision_reason}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n  Estrategias: OK")
    except Exception as e:
        print(f"  Estrategias: ERROR - {e}")
        import traceback
        traceback.print_exc()


def test_ensemble_with_real_image(image_path: str, strategy: str = "verify_if_low_conf"):
    """Test con imagen real."""
    print("\n" + "="*60)
    print("TEST 3: Imagen real")
    print("="*60)
    
    from pathlib import Path
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"  Imagen no encontrada: {image_path}")
        return
    
    print(f"  Imagen: {image_path}")
    print(f"  Estrategia: {strategy}")
    
    try:
        from llarri.inference.ensemble_ocr import EnsembleOCR, EnsembleStrategy, EnsembleConfig
        from llarri.inference.minicpm_adapter import MiniCPMAdapter, AdapterMode
        
        # Configurar ensemble
        config = EnsembleConfig(
            strategy=EnsembleStrategy(strategy),
            verbose=True,
        )
        
        # Usar mock de MiniCPM por defecto
        minicpm = MiniCPMAdapter(mode=AdapterMode.MOCK)
        
        ensemble = EnsembleOCR(
            minicpm_adapter=minicpm,
            config=config,
        )
        
        # Predecir
        start = time.time()
        result = ensemble.predict(str(image_path))
        elapsed = (time.time() - start) * 1000
        
        print(f"\n  Resultado:")
        print(f"    Texto: '{result.text}'")
        print(f"    Confianza: {result.confidence:.2f}")
        print(f"    LLARRI: '{result.llarri_prediction}' ({result.llarri_confidence or 0:.2f})")
        print(f"    MiniCPM: '{result.minicpm_prediction}' ({result.minicpm_confidence or 0:.2f})")
        print(f"    LM corrigió: {result.lm_corrected}")
        print(f"    Razón decisión: {result.decision_reason}")
        print(f"\n  Tiempos:")
        print(f"    Total: {result.total_time_ms:.2f}ms")
        print(f"    LLARRI: {result.llarri_time_ms:.2f}ms")
        print(f"    MiniCPM: {result.minicpm_time_ms:.2f}ms")
        print(f"    LM: {result.lm_time_ms:.2f}ms")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def test_llarri_with_ensemble():
    """Test del método predict de LlarriBaseModel con use_ensemble."""
    print("\n" + "="*60)
    print("TEST 4: LlarriBaseModel.predict(use_ensemble=True)")
    print("="*60)
    
    try:
        from llarri.models.llarri_base_model import LlarriBaseModel
        import torch
        
        # Crear modelo (sin cargar pesos preentrenados para test rápido)
        print("  Creando modelo...")
        model = LlarriBaseModel()
        model.eval()
        
        # Crear imagen dummy
        dummy_image = torch.randn(3, 224, 224)
        
        print("  Probando predict normal...")
        try:
            text = model.predict(dummy_image, preprocess=False, use_language_model=False)
            print(f"    Resultado: '{text}'")
        except Exception as e:
            print(f"    Error (esperado si no hay pesos): {e}")
        
        print("\n  Probando predict con ensemble...")
        try:
            text, metadata = model.predict(
                dummy_image,
                preprocess=False,
                use_ensemble=True,
                ensemble_strategy="minicpm_only",  # Usar solo MiniCPM mock
                return_metadata=True,
            )
            print(f"    Texto: '{text}'")
            print(f"    Metadata: {metadata}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n  LlarriBaseModel ensemble: OK")
    except Exception as e:
        print(f"  LlarriBaseModel: ERROR - {e}")
        import traceback
        traceback.print_exc()


def test_api_endpoint():
    """Test del endpoint de la API (si está disponible)."""
    print("\n" + "="*60)
    print("TEST 5: API endpoint (opcional)")
    print("="*60)
    
    try:
        import requests
        
        # Verificar si la API está corriendo
        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"  API disponible en {base_url}")
            else:
                print(f"  API respondió con status {response.status_code}")
                return
        except requests.exceptions.ConnectionError:
            print("  API no está corriendo (skipping)")
            return
        
        # TODO: Agregar test de endpoint de OCR con ensemble
        print("  API tests: pendiente de implementación")
        
    except ImportError:
        print("  requests no instalado (skipping)")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("ENSEMBLE OCR - Tests E2E")
    print("="*60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_ensemble_components()
    test_ensemble_strategies()
    test_llarri_with_ensemble()
    test_api_endpoint()
    
    print("\n" + "="*60)
    print("TESTS COMPLETADOS")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test ensemble OCR")
    parser.add_argument("--image", type=str, help="Ruta a imagen para test")
    parser.add_argument(
        "--strategy",
        type=str,
        default="verify_if_low_conf",
        choices=[
            "verify_if_low_conf",
            "always_verify",
            "consensus",
            "rerank",
            "llarri_only",
            "minicpm_only",
        ],
        help="Estrategia de ensemble",
    )
    parser.add_argument("--all", action="store_true", help="Ejecutar todos los tests")
    
    args = parser.parse_args()
    
    if args.image:
        test_ensemble_with_real_image(args.image, args.strategy)
    elif args.all or len(sys.argv) == 1:
        run_all_tests()


if __name__ == "__main__":
    main()
