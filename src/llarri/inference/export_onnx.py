"""
export_onnx.py - Exportación de modelos a formato ONNX

Permite exportar el modelo base y expertos a ONNX para inferencia
optimizada en producción (CPU, TensorRT, OpenVINO, etc.)

Uso:
    python -m llarri.inference.export_onnx --checkpoint models/base.ckpt --output models/base.onnx
    python -m llarri.inference.export_onnx --checkpoint models/expert.ckpt --output models/expert.onnx --expert
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class EncoderWrapper(nn.Module):
    """Wrapper para exportar solo el encoder a ONNX."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] imagen de entrada
            
        Returns:
            encoder_hidden_states: [B, seq_len, hidden_dim]
        """
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state


class FullModelWrapper(nn.Module):
    """Wrapper para exportar encoder+decoder (genera texto directamente)."""
    
    def __init__(self, model, max_length: int = 128):
        super().__init__()
        self.model = model
        self.max_length = max_length
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Nota: ONNX no soporta bien generate(), así que exportamos
        solo un forward pass del decoder. Para generación completa
        usar el modelo PyTorch o implementar beam search en ONNX Runtime.
        
        Args:
            pixel_values: [B, C, H, W]
            
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        # Encode
        encoder_outputs = self.model.encoder(pixel_values=pixel_values)
        encoder_hidden = encoder_outputs.last_hidden_state
        
        # Decoder forward (un paso, para ONNX)
        # En producción, iterar esto con beam search
        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=pixel_values.device)
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
        )
        
        return decoder_outputs.logits


def export_encoder_to_onnx(
    model,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> str:
    """
    Exporta solo el encoder a ONNX.
    
    Args:
        model: LlarriBaseModel cargado
        output_path: Ruta de salida .onnx
        input_shape: Shape de entrada (B, C, H, W)
        opset_version: Versión de ONNX opset
        dynamic_axes: Si usar ejes dinámicos para batch size variable
        
    Returns:
        Path al archivo ONNX generado
    """
    print(f"📦 Exportando encoder a ONNX...")
    
    # Crear wrapper
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()
    
    # Dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes
    axes = None
    if dynamic_axes:
        axes = {
            'pixel_values': {0: 'batch_size'},
            'encoder_output': {0: 'batch_size'}
        }
    
    # Exportar
    output_path = str(output_path)
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input,),
        output_path,
        input_names=['pixel_values'],
        output_names=['encoder_output'],
        dynamic_axes=axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )
    
    # Verificar tamaño
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Encoder exportado: {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def export_full_model_to_onnx(
    model,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    max_length: int = 128,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> str:
    """
    Exporta modelo completo (encoder + decoder) a ONNX.
    
    Nota: Solo exporta un paso de decoder. Para generación completa,
    implementar beam search en el runtime de ONNX.
    """
    print(f"📦 Exportando modelo completo a ONNX...")
    
    # Crear wrapper
    full_wrapper = FullModelWrapper(model, max_length)
    full_wrapper.eval()
    
    # Dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes
    axes = None
    if dynamic_axes:
        axes = {
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    
    # Exportar
    output_path = str(output_path)
    torch.onnx.export(
        full_wrapper,
        (dummy_input,),
        output_path,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes=axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Modelo completo exportado: {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def verify_onnx_model(onnx_path: str) -> bool:
    """Verifica que el modelo ONNX sea válido."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"✅ Modelo ONNX verificado correctamente")
        return True
    except ImportError:
        print("⚠️  onnx no instalado, saltando verificación")
        return True
    except Exception as e:
        print(f"❌ Error verificando ONNX: {e}")
        return False


def test_onnx_inference(
    onnx_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> Optional[Dict[str, Any]]:
    """Prueba inferencia con ONNX Runtime."""
    try:
        import onnxruntime as ort
        import numpy as np
        import time
        
        print(f"🧪 Probando inferencia ONNX...")
        
        # Crear sesión
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options)
        
        # Input dummy
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        _ = session.run(None, {'pixel_values': dummy_input})
        
        # Benchmark
        n_runs = 10
        start = time.time()
        for _ in range(n_runs):
            outputs = session.run(None, {'pixel_values': dummy_input})
        elapsed = (time.time() - start) / n_runs * 1000
        
        print(f"✅ Inferencia ONNX exitosa")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Latencia promedio: {elapsed:.1f} ms")
        
        return {
            'output_shape': outputs[0].shape,
            'latency_ms': elapsed,
        }
        
    except ImportError:
        print("⚠️  onnxruntime no instalado, saltando test de inferencia")
        return None
    except Exception as e:
        print(f"❌ Error en inferencia ONNX: {e}")
        return None


def export_model(
    checkpoint_path: str,
    output_path: str,
    encoder_only: bool = False,
    input_height: int = 224,
    input_width: int = 224,
    opset_version: int = 14,
    verify: bool = True,
    test_inference: bool = True,
) -> str:
    """
    Función principal de exportación.
    
    Args:
        checkpoint_path: Ruta al checkpoint PyTorch (.ckpt)
        output_path: Ruta de salida ONNX (.onnx)
        encoder_only: Si exportar solo encoder (más rápido, más pequeño)
        input_height: Altura de imagen de entrada
        input_width: Anchura de imagen de entrada
        opset_version: Versión de ONNX opset
        verify: Verificar modelo exportado
        test_inference: Probar inferencia con ONNX Runtime
        
    Returns:
        Path al archivo ONNX generado
    """
    from ..models.llarri_base_model import LlarriBaseModel
    
    print("=" * 60)
    print("EXPORTACIÓN A ONNX - LlarriOCR")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Modo: {'Encoder only' if encoder_only else 'Full model'}")
    print()
    
    # Cargar modelo
    print("📂 Cargando modelo...")
    model = LlarriBaseModel.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    model.eval()
    print(f"✅ Modelo cargado")
    
    # Crear directorio de salida
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Input shape
    input_shape = (1, 3, input_height, input_width)
    
    # Exportar
    with torch.no_grad():
        if encoder_only:
            result_path = export_encoder_to_onnx(
                model, output_path, input_shape, opset_version
            )
        else:
            result_path = export_full_model_to_onnx(
                model, output_path, input_shape, opset_version=opset_version
            )
    
    # Verificar
    if verify:
        verify_onnx_model(result_path)
    
    # Test inferencia
    if test_inference:
        test_onnx_inference(result_path, input_shape)
    
    print()
    print("=" * 60)
    print(f"✅ Exportación completada: {result_path}")
    print("=" * 60)
    
    return result_path


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo a ONNX")
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Ruta al checkpoint PyTorch (.ckpt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Ruta de salida ONNX (.onnx)'
    )
    parser.add_argument(
        '--encoder-only',
        action='store_true',
        help='Exportar solo encoder (más pequeño, requiere decoder separado)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=224,
        help='Altura de imagen de entrada (default: 224)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=224,
        help='Anchura de imagen de entrada (default: 224)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='No verificar modelo ONNX'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='No probar inferencia'
    )
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        encoder_only=args.encoder_only,
        input_height=args.height,
        input_width=args.width,
        opset_version=args.opset,
        verify=not args.no_verify,
        test_inference=not args.no_test,
    )


if __name__ == "__main__":
    main()

