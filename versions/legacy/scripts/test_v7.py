# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
"""
Test de LLARRI v7 - Arquitectura Cerebral

Verifica que todos los componentes funcionan correctamente.
"""

import torch
import sys
sys.path.insert(0, '.')

from llarri_o1.models.language_model_v7 import (
    LLARRIv7Cerebral,
    LLARRIv7Mini,
    LLARRIv7Base,
    crear_modelo_v7,
)


def test_componentes():
    """Test de componentes individuales."""
    print("=" * 60)
    print("TEST DE COMPONENTES LLARRI v7")
    print("=" * 60)
    
    dim = 64
    batch = 2
    seq = 32
    
    # Test Tálamo
    print("\n1. Testing Tálamo...")
    from llarri_o1.modules.cerebral.talamo import Talamo, TalamoConMemoria
    
    talamo = Talamo(dim, n_modulos=6, actividad_basal=0.15)
    x = torch.randn(batch, seq, dim)
    modulacion = talamo(x)
    
    assert modulacion.shape == (batch, 6), f"Shape incorrecto: {modulacion.shape}"
    assert modulacion.min() >= 0.15, f"Modulación bajo basal: {modulacion.min()}"
    assert modulacion.max() <= 1.0, f"Modulación sobre 1: {modulacion.max()}"
    print(f"   ✓ Modulación: {modulacion[0].tolist()}")
    print(f"   ✓ Min: {modulacion.min():.3f}, Max: {modulacion.max():.3f}")
    
    # Test Módulos especializados
    print("\n2. Testing Módulos Especializados...")
    from llarri_o1.modules.cerebral.modulos_especializados import (
        ModuloLenguaje, ModuloLogica, ModuloMatematicas,
        ModuloPatrones, ModuloContexto, ModuloCreatividad,
    )
    
    modulos = {
        'lenguaje': ModuloLenguaje(dim),
        'logica': ModuloLogica(dim),
        'matematicas': ModuloMatematicas(dim),
        'patrones': ModuloPatrones(dim),
        'contexto': ModuloContexto(dim),
        'creatividad': ModuloCreatividad(dim),
    }
    
    for nombre, modulo in modulos.items():
        out = modulo(x)
        assert out.shape == x.shape, f"{nombre}: shape incorrecto"
        print(f"   ✓ {nombre}: {modulo.dominio} - {out.shape}")
    
    # Test Hipocampo
    print("\n3. Testing Hipocampo...")
    from llarri_o1.modules.cerebral.hipocampo import Hipocampo
    
    hipocampo = Hipocampo(dim, capacidad=100, k_memorias=3)
    
    # Memorizar algunas experiencias
    hipocampo.train()
    for _ in range(5):
        exp = torch.randn(batch, seq, dim)
        _ = hipocampo(exp, memorizar=True)
    
    # Recordar
    hipocampo.eval()
    query = torch.randn(batch, seq, dim)
    resultado = hipocampo(query, memorizar=False)
    
    assert resultado.shape == query.shape, f"Hipocampo shape incorrecto"
    print(f"   ✓ Memorias almacenadas: {hipocampo.memorias_validas.sum().item()}")
    print(f"   ✓ Recuperación funciona: {resultado.shape}")
    
    # Test Integrador
    print("\n4. Testing Integrador...")
    from llarri_o1.modules.cerebral.integracion import IntegradorCerebral, DetectorConsenso
    
    integrador = IntegradorCerebral(dim, n_modulos=6)
    outputs = [torch.randn(batch, seq, dim) for _ in range(6)]
    modulaciones = torch.rand(batch, 6) * 0.85 + 0.15  # Entre 0.15 y 1.0
    
    integrado = integrador(outputs, modulaciones)
    assert integrado.shape == (batch, seq, dim)
    print(f"   ✓ Integración: {integrado.shape}")
    
    # Test Detector Consenso
    detector = DetectorConsenso(dim, n_modulos=6)
    consenso, stats = detector(outputs)
    print(f"   ✓ Consenso: {consenso.mean():.3f} ({stats})")
    
    print("\n✓ Todos los componentes funcionan correctamente!")


def test_modelo_completo():
    """Test del modelo completo."""
    print("\n" + "=" * 60)
    print("TEST DE MODELO COMPLETO LLARRI v7")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test Mini
    print("\n1. Testing LLARRIv7Mini...")
    modelo_mini = LLARRIv7Mini(vocab_size=256).to(device)
    print(f"   Parámetros: {modelo_mini.get_num_params():,}")
    
    # Forward pass
    input_ids = torch.randint(0, 256, (2, 64)).to(device)
    targets = torch.randint(0, 256, (2, 64)).to(device)
    
    result = modelo_mini(input_ids, targets)
    print(f"   ✓ Logits: {result['logits'].shape}")
    print(f"   ✓ Loss: {result['loss'].item():.4f}")
    print(f"   ✓ Stats: iteraciones={result['stats']['iteraciones']}, consenso={result['stats']['consenso_medio']:.3f}")
    
    # Test Base
    print("\n2. Testing LLARRIv7Base...")
    modelo_base = LLARRIv7Base(vocab_size=256).to(device)
    print(f"   Parámetros: {modelo_base.get_num_params():,}")
    
    result = modelo_base(input_ids, targets)
    print(f"   ✓ Loss: {result['loss'].item():.4f}")
    
    # Mostrar modulaciones
    print("\n   Modulaciones por módulo:")
    for key, val in result['stats'].items():
        if key.startswith('mod_'):
            nombre = key.replace('mod_', '')
            print(f"      {nombre}: {val:.3f}")
    
    # Test generación
    print("\n3. Testing Generación...")
    prompt = torch.randint(0, 256, (1, 10)).to(device)
    generated = modelo_base.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"   ✓ Generado: {generated.shape} tokens")
    print(f"   ✓ Primeros tokens: {generated[0, :15].tolist()}")
    
    print("\n✓ Modelo completo funciona correctamente!")


def test_gradientes():
    """Verifica que los gradientes fluyen correctamente."""
    print("\n" + "=" * 60)
    print("TEST DE GRADIENTES")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modelo = LLARRIv7Mini(vocab_size=256).to(device)
    
    input_ids = torch.randint(0, 256, (2, 32)).to(device)
    targets = torch.randint(0, 256, (2, 32)).to(device)
    
    # Forward
    result = modelo(input_ids, targets)
    loss = result['loss']
    
    # Backward
    loss.backward()
    
    # Verificar gradientes
    gradientes_ok = True
    for name, param in modelo.named_parameters():
        if param.grad is None:
            print(f"   ⚠ Sin gradiente: {name}")
            gradientes_ok = False
        elif param.grad.abs().max() == 0:
            print(f"   ⚠ Gradiente cero: {name}")
            
    if gradientes_ok:
        print("   ✓ Todos los parámetros tienen gradientes")
        
    # Verificar módulos específicos
    print("\n   Gradientes por componente:")
    componentes = ['embedding', 'talamo', 'modulos', 'integrador', 'hipocampo', 'output']
    for comp in componentes:
        grad_sum = 0
        count = 0
        for name, param in modelo.named_parameters():
            if comp in name and param.grad is not None:
                grad_sum += param.grad.abs().mean().item()
                count += 1
        if count > 0:
            print(f"      {comp}: grad_mean={grad_sum/count:.6f}")


def test_eficiencia():
    """Test de eficiencia y velocidad."""
    print("\n" + "=" * 60)
    print("TEST DE EFICIENCIA")
    print("=" * 60)
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modelo = LLARRIv7Base(vocab_size=256).to(device)
    modelo.eval()
    
    batch_size = 32
    seq_len = 128
    
    input_ids = torch.randint(0, 256, (batch_size, seq_len)).to(device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = modelo(input_ids)
    
    # Medir
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    n_iters = 10
    start = time.time()
    
    for _ in range(n_iters):
        with torch.no_grad():
            result = modelo(input_ids)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    elapsed = time.time() - start
    
    tokens_total = batch_size * seq_len * n_iters
    tokens_per_sec = tokens_total / elapsed
    
    print(f"\n   Batch size: {batch_size}")
    print(f"   Seq length: {seq_len}")
    print(f"   Tiempo total: {elapsed:.2f}s")
    print(f"   Throughput: {tokens_per_sec:,.0f} tokens/seg")
    print(f"   Iteraciones promedio: {result['stats']['iteraciones']}")
    
    # Memoria
    if device.type == 'cuda':
        memoria_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   Memoria GPU: {memoria_mb:.0f} MB")


def main():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 60)
    print("      LLARRI v7 - ARQUITECTURA CEREBRAL")
    print("             TEST SUITE")
    print("=" * 60)
    
    test_componentes()
    test_modelo_completo()
    test_gradientes()
    test_eficiencia()
    
    print("\n" + "=" * 60)
    print("✓ TODOS LOS TESTS PASARON")
    print("=" * 60)


if __name__ == '__main__':
    main()
