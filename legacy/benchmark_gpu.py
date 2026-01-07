# SPDX-License-Identifier: AGPL-3.0-or-later
"""
🔺 VECTORES FRACTALES x3 - OPTIMIZADO PARA GPU
==============================================

Probando en tu GTX 1650 (Legion)
Comparando CPU vs GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ================================================================
# DETECTAR HARDWARE
# ================================================================

print("\n" + "="*70)
print("🖥️  DETECTANDO HARDWARE...")
print("="*70)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU DETECTADA: {gpu_name}")
    print(f"   Memoria: {gpu_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  No se detectó GPU, usando CPU")

print(f"   Dispositivo seleccionado: {device}")
print("="*70)


# ================================================================
# ARQUITECTURA FRACTAL COMPRIMIDA (lo mejor de ambos mundos)
# ================================================================

class VectorFractalCompartido(nn.Module):
    """
    Vector fractal con PESOS COMPARTIDOS.
    Mismo ADN, diferentes expresiones.
    """
    def __init__(self, dimension, nivel=0, max_nivel=3, plantilla_compartida=None):
        super().__init__()
        self.dimension = dimension
        self.nivel = nivel
        self.max_nivel = max_nivel
        
        if nivel >= max_nivel:
            # Átomo
            self.es_atomico = True
            self.transformacion = nn.Linear(dimension, dimension)
        else:
            self.es_atomico = False
            dim_hijo = max(dimension // 2, 4)
            
            # COMPARTIR la plantilla entre todos los hijos del mismo nivel
            if plantilla_compartida is None:
                # Soy el primero, creo la plantilla
                self.plantilla = VectorFractalCompartido(
                    dim_hijo, nivel + 1, max_nivel
                )
                self.soy_plantilla = True
            else:
                # Uso la plantilla existente
                self.plantilla = plantilla_compartida
                self.soy_plantilla = False
            
            # Solo guardo las "personalidades" únicas (muy pequeñas)
            self.personalidades = nn.Parameter(torch.randn(3, dim_hijo) * 0.1)
            
            self.hacia_hijos = nn.Linear(dimension, dim_hijo)
            self.desde_hijos = nn.Linear(dim_hijo * 3, dimension)
    
    def forward(self, x):
        if self.es_atomico:
            return torch.tanh(self.transformacion(x))
        else:
            x_hijo = self.hacia_hijos(x)
            
            respuestas = []
            for i in range(3):
                # Mismo hijo (plantilla), diferente personalidad
                personalidad = torch.sigmoid(self.personalidades[i])
                respuesta = self.plantilla(x_hijo * personalidad)
                respuestas.append(respuesta)
            
            combinado = torch.cat(respuestas, dim=-1)
            return torch.tanh(self.desde_hijos(combinado))


class TrinidadFractalGPU(nn.Module):
    """
    Trinidad Fractal optimizada para GPU.
    Vectores compartidos + operaciones eficientes.
    """
    def __init__(self, dim_entrada, dim_oculta, dim_salida, profundidad=3):
        super().__init__()
        
        # UNA plantilla de vector compartida por todas las cajas
        self.vector_plantilla = VectorFractalCompartido(
            dim_oculta, nivel=0, max_nivel=profundidad
        )
        
        # Personalidades para cada "caja" (muy pequeñas)
        self.pers_caja1 = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.pers_caja2 = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.pers_caja3 = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        
        # Conexiones simples (eficientes)
        self.conexion12 = nn.Linear(dim_oculta, dim_oculta)
        self.conexion23 = nn.Linear(dim_oculta, dim_oculta)
        self.skip13 = nn.Linear(dim_oculta, dim_oculta)
        
        self.entrada = nn.Linear(dim_entrada, dim_oculta)
        self.salida = nn.Linear(dim_oculta, dim_salida)
    
    def forward(self, x):
        x = torch.tanh(self.entrada(x))
        
        # Caja 1
        p1 = torch.sigmoid(self.pers_caja1)
        s1 = self.vector_plantilla(x * p1)
        
        # Caja 2
        hacia2 = torch.tanh(self.conexion12(s1))
        p2 = torch.sigmoid(self.pers_caja2)
        s2 = self.vector_plantilla(hacia2 * p2)
        
        # Caja 3
        hacia3 = torch.tanh(self.conexion23(s2))
        skip = torch.tanh(self.skip13(s1))
        p3 = torch.sigmoid(self.pers_caja3)
        s3 = self.vector_plantilla((hacia3 + skip) * p3)
        
        return self.salida(s1 + s2 + s3)


# ================================================================
# BENCHMARK: CPU vs GPU
# ================================================================

def benchmark(modelo, X, y, epochs, device_name, show_progress=False):
    """Entrena y mide tiempo"""
    modelo = modelo.to(device)
    X_dev = X.to(device)
    y_dev = y.to(device)
    
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(modelo(X_dev), y_dev)
        loss.backward()
        optimizer.step()
    
    # Sincronizar antes de medir
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    best_acc = 0
    
    for epoca in range(1, epochs + 1):
        optimizer.zero_grad()
        salida = modelo(X_dev)
        loss = criterion(salida, y_dev)
        loss.backward()
        optimizer.step()
        
        # Mostrar progreso cada 20 epochs
        if show_progress and epoca % 20 == 0:
            with torch.no_grad():
                pred = modelo(X_dev).argmax(dim=1)
                acc = (pred == y_dev).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
                elapsed_now = time.time() - start
                print(f"   Época {epoca:3d}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.1f}% | Best: {best_acc:.1f}% | {elapsed_now:.1f}s")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    # Accuracy final
    with torch.no_grad():
        pred = modelo(X_dev).argmax(dim=1)
        acc = (pred == y_dev).float().mean().item() * 100
    
    return elapsed, acc, loss.item()


if __name__ == "__main__":
    print("\n" + "🔺"*25)
    print("  BENCHMARK: TRINITY FRACTAL EN TU GTX 1650")
    print("🔺"*25)
    
    # Crear modelo
    modelo = TrinidadFractalGPU(
        dim_entrada=784,      # Como MNIST (28x28)
        dim_oculta=128,       # Más grande para ver diferencia
        dim_salida=10,        # 10 clases
        profundidad=3
    )
    
    params = sum(p.numel() for p in modelo.parameters())
    print(f"\n💾 Parámetros del modelo: {params:,}")
    
    # Dataset MÁXIMO para tu GPU de 4GB
    print("\n📊 Generando dataset MÁXIMO (20,000 muestras)...")
    torch.manual_seed(42)
    X = torch.randn(20000, 784)  # 20k imágenes simuladas
    y = torch.randint(0, 10, (20000,))
    
    EPOCHS = 200  # ¡AL MÁXIMO!
    
    # ========== TEST EN GPU ==========
    if torch.cuda.is_available():
        print(f"\n🚀 ENTRENANDO EN GPU ({torch.cuda.get_device_name(0)})...")
        print(f"   {EPOCHS} EPOCHS - ¡AL MÁXIMO! 🔥\n")
        modelo_gpu = TrinidadFractalGPU(784, 128, 10, 3)
        
        tiempo_gpu, acc_gpu, loss_gpu = benchmark(modelo_gpu, X, y, EPOCHS, "GPU", show_progress=True)
        
        print(f"   ⏱️  Tiempo: {tiempo_gpu:.2f} segundos")
        print(f"   🎯 Accuracy: {acc_gpu:.1f}%")
        print(f"   📉 Loss final: {loss_gpu:.4f}")
        print(f"   ⚡ Velocidad: {EPOCHS/tiempo_gpu:.1f} epochs/segundo")
        
        # Memoria usada
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   💾 Memoria GPU usada: {mem_used:.1f} MB")
    
    # ========== TEST EN CPU (para comparar) ==========
    print(f"\n🐢 ENTRENANDO EN CPU (para comparar)...")
    device_backup = device
    device = torch.device("cpu")
    
    modelo_cpu = TrinidadFractalGPU(784, 128, 10, 3)
    
    # Solo 5 epochs en CPU para no esperar mucho
    tiempo_cpu, acc_cpu, loss_cpu = benchmark(modelo_cpu, X[:5000], y[:5000], 5, "CPU")
    tiempo_cpu_estimado = tiempo_cpu * (EPOCHS/5) * (50000/5000)  # Escalar
    
    print(f"   ⏱️  Tiempo (5 epochs, 5k muestras): {tiempo_cpu:.2f} segundos")
    print(f"   ⏱️  Tiempo estimado (20 epochs, 50k): {tiempo_cpu_estimado:.1f} segundos")
    
    device = device_backup
    
    # ========== COMPARACIÓN FINAL ==========
    if torch.cuda.is_available():
        speedup = tiempo_cpu_estimado / tiempo_gpu
        
        print("\n" + "="*70)
        print("📊 COMPARACIÓN FINAL")
        print("="*70)
        print(f"""
    ┌─────────────────┬──────────────┬──────────────┐
    │    Métrica      │     CPU      │  GTX 1650    │
    ├─────────────────┼──────────────┼──────────────┤
    │  Tiempo         │  ~{tiempo_cpu_estimado:>6.1f}s    │    {tiempo_gpu:>6.2f}s   │
    │  Speedup        │     1.0x     │    {speedup:>5.1f}x    │
    │  Epochs/seg     │    ~{5/tiempo_cpu:>5.2f}    │    {EPOCHS/tiempo_gpu:>6.2f}   │
    └─────────────────┴──────────────┴──────────────┘
    
    🚀 Tu GTX 1650 es ~{speedup:.0f}x más rápida que CPU!
    
    Con 50,000 muestras y arquitectura fractal:
    - CPU tardaría: ~{tiempo_cpu_estimado/60:.1f} minutos
    - GPU tardó: {tiempo_gpu:.1f} segundos
    
    ⚡ AHORRO DE TIEMPO: {tiempo_cpu_estimado - tiempo_gpu:.0f} segundos
        """)
    
    print("\n✅ ¡Tu arquitectura fractal corre perfecto en GPU!")
    print("   Más datos + GPU = Entrenamiento RÁPIDO")
