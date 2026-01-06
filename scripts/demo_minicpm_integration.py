#!/usr/bin/env python3
"""
Demo: Integración de MiniCPM-V con LLARRI OCR

Este script muestra cómo usar MiniCPM-V para mejorar la precisión de LLARRI.

Opciones de integración:
1. MiniCPM como verificador (valida/corrige predicciones de LLARRI)
2. MiniCPM como OCR alternativo (para documentos complejos)
3. Ensemble (combina ambos modelos)

Requisitos:
    pip install torch transformers pillow
    # MiniCPM-V int4 requiere ~9GB VRAM
    # MiniCPM-V full requiere ~18GB VRAM

Para GPUs con menos VRAM:
    - Usar Ollama: ollama run minicpm-v
    - Usar vLLM con cuantización
    - Usar llama.cpp (GGUF)
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_gpu_memory():
    """Verifica memoria GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / 1024**3
                print(f"GPU {i}: {props.name}")
                print(f"  Memoria total: {total_gb:.1f} GB")
                
                if total_gb < 9:
                    print(f"  ⚠️  MiniCPM-V int4 requiere 9GB VRAM")
                    print(f"  💡 Considera usar Ollama: ollama run minicpm-v")
                elif total_gb < 18:
                    print(f"  ✅ Suficiente para MiniCPM-V int4")
                else:
                    print(f"  ✅ Suficiente para MiniCPM-V full")
                print()
        else:
            print("⚠️  No hay GPU CUDA disponible")
            print("💡 MiniCPM-V puede correr en CPU pero será lento")
    except ImportError:
        print("❌ PyTorch no instalado")


def demo_ollama_integration():
    """
    Demo usando Ollama (alternativa para GPUs pequeñas).
    
    Ollama permite correr MiniCPM-V con GGUF optimizado:
        ollama run minicpm-v
        
    Es más eficiente en memoria y funciona bien en GPUs de 4-8GB.
    """
    print("\n=== Demo: MiniCPM-V via Ollama ===\n")
    print("Ollama es la mejor opción para GPUs con poca VRAM (< 9GB)")
    print()
    print("Instalación:")
    print("  1. curl -fsSL https://ollama.com/install.sh | sh")
    print("  2. ollama pull minicpm-v")
    print()
    
    # Verificar si Ollama está disponible
    import subprocess
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama instalado: {result.stdout.strip()}")
            
            # Verificar si el modelo está disponible
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if 'minicpm-v' in result.stdout.lower():
                print("✅ Modelo minicpm-v disponible")
            else:
                print("⚠️  Modelo minicpm-v no instalado")
                print("   Ejecuta: ollama pull minicpm-v")
        else:
            print("❌ Ollama no disponible")
    except FileNotFoundError:
        print("❌ Ollama no instalado")
        print("   Visita: https://ollama.com")


class OllamaVerifier:
    """
    Verificador OCR usando Ollama (alternativa a MiniCPMVerifier).
    
    Ventajas:
    - Funciona en GPUs pequeñas (4GB+)
    - Fácil de instalar y usar
    - Modelos optimizados (GGUF)
    
    Uso:
        verifier = OllamaVerifier()
        text = verifier.ocr("imagen.jpg")
    """
    
    def __init__(self, model: str = "minicpm-v"):
        """
        Args:
            model: Modelo de Ollama a usar
        """
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Verifica que Ollama esté disponible."""
        import subprocess
        try:
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if self.model.lower() not in result.stdout.lower():
                print(f"⚠️  Modelo {self.model} no instalado")
                print(f"   Ejecuta: ollama pull {self.model}")
        except Exception as e:
            print(f"❌ Error verificando Ollama: {e}")
    
    def _encode_image(self, image_path: str) -> str:
        """Codifica imagen en base64."""
        import base64
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def chat(self, image_path: str, prompt: str) -> str:
        """
        Envía imagen y prompt a Ollama.
        
        Args:
            image_path: Ruta a la imagen
            prompt: Pregunta o instrucción
            
        Returns:
            Respuesta del modelo
        """
        import requests
        
        # Codificar imagen
        image_b64 = self._encode_image(image_path)
        
        # Llamar API de Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': self.model,
                'prompt': prompt,
                'images': [image_b64],
                'stream': False,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise RuntimeError(f"Ollama error: {response.text}")
    
    def ocr(self, image_path: str) -> str:
        """Realiza OCR usando Ollama."""
        prompt = (
            "Transcribe exactamente el texto manuscrito en esta imagen. "
            "Solo responde con el texto, sin explicaciones."
        )
        return self.chat(image_path, prompt)
    
    def verify(self, text: str, image_path: str) -> tuple:
        """Verifica si el texto OCR es correcto."""
        prompt = (
            f'El texto "{text}" fue reconocido de esta imagen. '
            f'¿Es correcto? Si hay errores, responde "INCORRECTO: [texto correcto]". '
            f'Si es correcto, responde "CORRECTO".'
        )
        response = self.chat(image_path, prompt)
        
        if response.upper().startswith("CORRECTO"):
            return True, None
        elif "INCORRECTO" in response.upper():
            # Extraer corrección
            parts = response.split(":", 1)
            if len(parts) > 1:
                return False, parts[1].strip()
            return False, self.ocr(image_path)
        else:
            return True, None


def demo_ensemble_architecture():
    """Muestra la arquitectura del ensemble."""
    print("\n=== Arquitectura del Ensemble LLARRI + MiniCPM-V ===\n")
    
    architecture = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                     ENSEMBLE OCR PIPELINE                        │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌──────────────┐
    │   Imagen     │
    │  Manuscrita  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ PASO 1: PREPROCESAMIENTO                                      │
    │   • Binarización adaptativa                                   │
    │   • Corrección de inclinación                                 │
    │   • Normalización de contraste                                │
    └──────┬───────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ PASO 2: OCR PRIMARIO (LLARRI - TrOCR)                        │
    │   • ViT Encoder → Token features                              │
    │   • TrOCR Decoder → Texto inicial                             │
    │   • Salida: "Holla mundo" (confianza: 0.75)                  │
    └──────┬───────────────────────────────────────────────────────┘
           │
           ▼ (si confianza < 0.8)
    ┌──────────────────────────────────────────────────────────────┐
    │ PASO 3: VERIFICACIÓN (MiniCPM-V)                             │
    │   • Recibe: imagen + texto LLARRI                            │
    │   • Verifica si es correcto                                   │
    │   • Si incorrecto → propone corrección: "Hola mundo"         │
    └──────┬───────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ PASO 4: RANKING CON LANGUAGE MODEL                           │
    │   • Candidatos: ["Holla mundo", "Hola mundo"]                │
    │   • Markov chains → P("Hola mundo") > P("Holla mundo")       │
    │   • SpellChecker → "Holla" no existe, "Hola" sí              │
    │   • Selección: "Hola mundo" ✓                                │
    └──────┬───────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ RESULTADO FINAL                                               │
    │   • Texto: "Hola mundo"                                       │
    │   • Confianza: 0.95                                           │
    │   • Método: llarri + minicpm_verified + lm_corrected         │
    └──────────────────────────────────────────────────────────────┘
    """
    print(architecture)


def demo_integration_code():
    """Muestra código de integración."""
    print("\n=== Código de Integración ===\n")
    
    code = '''
# Ejemplo completo de uso del Ensemble

from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.inference.minicpm_verifier import MiniCPMVerifier, EnsembleOCR
from llarri.inference.language_model import SpanishLanguageModel

# 1. Cargar modelos
llarri = LlarriBaseModel.load_from_checkpoint("checkpoint.ckpt")
llarri.eval()

# 2. Crear verificador MiniCPM-V
# Opción A: Usando transformers (requiere 9GB+ VRAM)
minicpm = MiniCPMVerifier('minicpm-v-4.5-int4', load_on_init=False)

# Opción B: Usando Ollama (funciona en 4GB VRAM)
# minicpm = OllamaVerifier('minicpm-v')

# 3. Cargar modelo de lenguaje
lm = SpanishLanguageModel()

# 4. Crear ensemble
ensemble = EnsembleOCR(
    llarri_model=llarri,
    minicpm_verifier=minicpm,
    language_model=lm,
    confidence_threshold=0.8,  # Verificar si confianza < 80%
)

# 5. Predecir
result = ensemble.predict("imagen_manuscrita.jpg", return_details=True)

print(f"Texto final: {result['final_text']}")
print(f"Confianza: {result['final_confidence']:.2%}")
print(f"Método: {result['method']}")

# Output ejemplo:
# Texto final: Hola mundo
# Confianza: 95.00%
# Método: llarri + minicpm_verified + lm_corrected
'''
    print(code)


def main():
    print("=" * 60)
    print("DEMO: Integración MiniCPM-V con LLARRI OCR")
    print("=" * 60)
    
    # 1. Verificar memoria GPU
    print("\n1. Verificando hardware:")
    check_gpu_memory()
    
    # 2. Verificar Ollama
    print("\n2. Verificando Ollama (alternativa para GPUs pequeñas):")
    demo_ollama_integration()
    
    # 3. Mostrar arquitectura
    demo_ensemble_architecture()
    
    # 4. Mostrar código de integración
    demo_integration_code()
    
    # 5. Recomendaciones
    print("\n=== Recomendaciones según tu GPU ===\n")
    print("GTX 1650 (4GB VRAM):")
    print("  ❌ MiniCPM-V via transformers (requiere 9GB)")
    print("  ✅ MiniCPM-V via Ollama (GGUF optimizado)")
    print("  ✅ LLARRI + Language Model sin verificador")
    print()
    print("RTX 3060 (12GB VRAM):")
    print("  ✅ MiniCPM-V int4 via transformers")
    print("  ✅ Ensemble completo")
    print()
    print("RTX 3090/4090 (24GB VRAM):")
    print("  ✅ MiniCPM-V full via transformers")
    print("  ✅ Ensemble completo con máxima calidad")
    print()
    
    # 6. Instalación
    print("\n=== Instalación ===\n")
    print("Para GPU pequeña (Ollama):")
    print("  curl -fsSL https://ollama.com/install.sh | sh")
    print("  ollama pull minicpm-v")
    print()
    print("Para GPU grande (transformers):")
    print("  pip install torch transformers pillow")
    print("  # El modelo se descarga automáticamente al usar")


if __name__ == "__main__":
    main()
