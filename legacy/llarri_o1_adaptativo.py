# SPDX-License-Identifier: AGPL-3.0-or-later
"""
LLARRI-O1 Trinity Fractal - Modo Adaptativo GPU/Híbrido
=======================================================

Detecta automáticamente:
- Si el modelo cabe en VRAM → Todo en GPU (rápido)
- Si el modelo NO cabe → Modo híbrido CPU+GPU (funciona)

Segunda Cabeza - Lucas Mella
"""

import torch
import torch.nn as nn
import math

# ==============================================================================
# DETECTOR DE RECURSOS
# ==============================================================================

class ResourceDetector:
    """Detecta VRAM disponible y decide el modo de ejecución"""
    
    @staticmethod
    def get_vram_total():
        """VRAM total en bytes"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    @staticmethod
    def get_vram_libre():
        """VRAM libre en bytes"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        return 0
    
    @staticmethod
    def get_ram_libre():
        """RAM libre aproximada en bytes"""
        import os
        try:
            import psutil
            return psutil.virtual_memory().available
        except:
            return 16 * 1024**3  # Asumir 16GB si no se puede detectar
    
    @staticmethod
    def estimar_memoria_modelo(params, dtype=torch.float32):
        """Estima memoria necesaria para modelo + gradientes + optimizer"""
        bytes_por_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
        }
        
        base = params * bytes_por_param.get(dtype, 4)
        
        # Inferencia: solo pesos + activaciones (~1.5x pesos)
        inferencia = base * 1.5
        
        # Entrenamiento: pesos + gradientes + optimizer states (~4x pesos)
        entrenamiento = base * 4
        
        return {
            'inferencia': inferencia,
            'entrenamiento': entrenamiento
        }
    
    @staticmethod
    def decidir_modo(params, modo='inferencia', margen_seguridad=0.7):
        """
        Decide si usar GPU pura o modo híbrido
        
        Returns:
            'gpu': Todo en GPU
            'hibrido': CPU+GPU intercalado
        """
        vram_total = ResourceDetector.get_vram_total()
        memoria_necesaria = ResourceDetector.estimar_memoria_modelo(params)
        
        mem_requerida = memoria_necesaria[modo]
        vram_usable = vram_total * margen_seguridad  # Dejar 30% margen
        
        print(f"   [Debug] Memoria requerida: {mem_requerida/1e9:.2f} GB")
        print(f"   [Debug] VRAM usable (70%): {vram_usable/1e9:.2f} GB")
        
        if mem_requerida < vram_usable:
            return 'gpu'
        else:
            return 'hibrido'


# ==============================================================================
# TRINITY BOX ADAPTATIVA
# ==============================================================================

class TrinityBoxAdaptiva(nn.Module):
    """Caja que puede ejecutarse en GPU o CPU según necesidad"""
    
    def __init__(self, input_dim, hidden_dim, num_worlds=3):
        super().__init__()
        self.worlds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_worlds)
        ])
        self.fusion = nn.Linear(hidden_dim * num_worlds, hidden_dim)
        self.hidden_dim = hidden_dim
        
        # Estado actual
        self._device = None
        
    def to_device(self, device):
        """Mueve la caja a un dispositivo específico"""
        self._device = device
        return self.to(device)
    
    def forward(self, x):
        # Asegurar que input está en el mismo dispositivo
        if self._device and x.device != self._device:
            x = x.to(self._device)
            
        world_outputs = [world(x) for world in self.worlds]
        fused = torch.cat(world_outputs, dim=-1)
        return self.fusion(fused)


class LlaveBidireccional(nn.Module):
    """Llave bidireccional - siempre ligera, puede ir en CPU"""
    
    def __init__(self, dim):
        super().__init__()
        self.forward_key = nn.Linear(dim, dim)
        self.backward_key = nn.Linear(dim, dim)
        
    def forward(self, x1, x2):
        x1_nuevo = x1 + self.backward_key(x2)
        x2_nuevo = x2 + self.forward_key(x1)
        return x1_nuevo, x2_nuevo


# ==============================================================================
# MODELO PRINCIPAL ADAPTATIVO
# ==============================================================================

class LLARRI_O1_Adaptativo(nn.Module):
    """
    LLARRI-O1 Trinity Fractal con modo adaptativo
    
    - Detecta automáticamente VRAM disponible
    - Si cabe: todo en GPU (máxima velocidad)
    - Si no cabe: modo híbrido CPU+GPU (funciona igual)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config.get('hidden_size', 512)
        input_size = config.get('input_size', 784)
        output_size = config.get('output_size', 10)
        num_worlds = config.get('num_worlds', 3)
        
        # Las tres cajas
        self.caja_padre = TrinityBoxAdaptiva(input_size, hidden, num_worlds)
        self.caja_hijo = TrinityBoxAdaptiva(hidden, hidden, num_worlds)
        self.caja_espiritu = TrinityBoxAdaptiva(hidden * 2, hidden, num_worlds)
        
        # Llaves (siempre ligeras)
        self.llave_padre_hijo = LlaveBidireccional(hidden)
        self.llave_padre_espiritu = LlaveBidireccional(hidden)
        
        # Salida
        self.output = nn.Linear(hidden, output_size)
        
        # Modo de ejecución
        self._modo = None
        self._gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cpu_device = torch.device('cpu')
        
        # Detectar modo óptimo
        self._auto_configurar()
        
    def _contar_parametros(self):
        """Cuenta parámetros totales"""
        return sum(p.numel() for p in self.parameters())
    
    def _auto_configurar(self):
        """Configura automáticamente el modo basado en recursos"""
        params = self._contar_parametros()
        self._modo = ResourceDetector.decidir_modo(params)
        
        print(f"\n{'='*60}")
        print(f"LLARRI-O1 ADAPTATIVO - Configuración Automática")
        print(f"{'='*60}")
        print(f"Parámetros: {params:,}")
        print(f"VRAM disponible: {ResourceDetector.get_vram_libre() / 1e9:.2f} GB")
        print(f"Modo seleccionado: {self._modo.upper()}")
        
        if self._modo == 'gpu':
            print("→ Todo en GPU (máxima velocidad)")
            self._configurar_gpu()
        else:
            print("→ Modo híbrido CPU+GPU (ahorra VRAM)")
            self._configurar_hibrido()
        print(f"{'='*60}\n")
    
    def _configurar_gpu(self):
        """Mueve todo a GPU"""
        self.to(self._gpu_device)
        self.caja_padre._device = self._gpu_device
        self.caja_hijo._device = self._gpu_device
        self.caja_espiritu._device = self._gpu_device
        
    def _configurar_hibrido(self):
        """Configura modo híbrido: llaves en CPU, cajas se mueven dinámicamente"""
        # Llaves siempre en CPU (son ligeras)
        self.llave_padre_hijo.to(self._cpu_device)
        self.llave_padre_espiritu.to(self._cpu_device)
        self.output.to(self._gpu_device)
    
    def cambiar_modo(self, modo):
        """Cambia manualmente el modo de ejecución"""
        assert modo in ['gpu', 'hibrido'], "Modo debe ser 'gpu' o 'hibrido'"
        self._modo = modo
        if modo == 'gpu':
            self._configurar_gpu()
        else:
            self._configurar_hibrido()
        print(f"Modo cambiado a: {modo.upper()}")
    
    def forward(self, x):
        """Forward pass - se adapta según el modo"""
        if self._modo == 'gpu':
            return self._forward_gpu(x)
        else:
            return self._forward_hibrido(x)
    
    def _forward_gpu(self, x):
        """Todo en GPU - máxima velocidad"""
        x = x.to(self._gpu_device)
        
        # Caja Padre
        padre = self.caja_padre(x)
        
        # Caja Hijo
        hijo = self.caja_hijo(padre)
        
        # Llaves bidireccionales
        padre, hijo = self.llave_padre_hijo(padre, hijo)
        
        # Caja Espíritu (fusión)
        fusion = torch.cat([padre, hijo], dim=-1)
        espiritu = self.caja_espiritu(fusion)
        
        # Skip connection
        padre, espiritu = self.llave_padre_espiritu(padre, espiritu)
        
        return self.output(espiritu)
    
    def _forward_hibrido(self, x):
        """Híbrido CPU+GPU - ahorra VRAM"""
        
        # ===== CAJA PADRE EN GPU =====
        self.caja_padre.to_device(self._gpu_device)
        x_gpu = x.to(self._gpu_device)
        padre = self.caja_padre(x_gpu)
        
        # Liberar VRAM de caja padre
        padre_cpu = padre.cpu()
        self.caja_padre.to_device(self._cpu_device)
        torch.cuda.empty_cache()
        
        # ===== CAJA HIJO EN GPU =====
        self.caja_hijo.to_device(self._gpu_device)
        hijo = self.caja_hijo(padre_cpu.to(self._gpu_device))
        
        # Liberar VRAM de caja hijo
        hijo_cpu = hijo.cpu()
        self.caja_hijo.to_device(self._cpu_device)
        torch.cuda.empty_cache()
        
        # ===== LLAVES EN CPU (siempre) =====
        padre_cpu, hijo_cpu = self.llave_padre_hijo(padre_cpu, hijo_cpu)
        
        # ===== CAJA ESPÍRITU EN GPU =====
        self.caja_espiritu.to_device(self._gpu_device)
        fusion = torch.cat([padre_cpu, hijo_cpu], dim=-1).to(self._gpu_device)
        espiritu = self.caja_espiritu(fusion)
        
        # Skip connection en CPU
        espiritu_cpu = espiritu.cpu()
        padre_cpu, espiritu_cpu = self.llave_padre_espiritu(padre_cpu, espiritu_cpu)
        
        # ===== SALIDA EN GPU =====
        output = self.output(espiritu_cpu.to(self._gpu_device))
        
        # Limpiar
        self.caja_espiritu.to_device(self._cpu_device)
        torch.cuda.empty_cache()
        
        return output
    
    def info(self):
        """Muestra información del modelo"""
        params = self._contar_parametros()
        mem = ResourceDetector.estimar_memoria_modelo(params)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              LLARRI-O1 ADAPTATIVO - INFO                     ║
╠══════════════════════════════════════════════════════════════╣
║  Parámetros:     {params:>12,}                               ║
║  Modo actual:    {self._modo:>12}                               ║
║  VRAM inferencia: {mem['inferencia']/1e6:>10.1f} MB                          ║
║  VRAM entrenamiento: {mem['entrenamiento']/1e6:>7.1f} MB                          ║
╠══════════════════════════════════════════════════════════════╣
║  GPU disponible: {ResourceDetector.get_vram_libre()/1e9:>10.2f} GB                          ║
╚══════════════════════════════════════════════════════════════╝
        """)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    import time
    
    print("\n" + "=" * 70)
    print("DEMO: LLARRI-O1 Adaptativo")
    print("=" * 70)
    
    # Crear modelo pequeño (debería usar GPU)
    print("\n[1] Modelo PEQUEÑO (debería elegir GPU):")
    config_pequeno = {
        'hidden_size': 256,
        'input_size': 784,
        'output_size': 10,
        'num_worlds': 3
    }
    modelo_pequeno = LLARRI_O1_Adaptativo(config_pequeno)
    modelo_pequeno.info()
    
    # Test de velocidad
    print("Probando velocidad...")
    x = torch.randn(64, 784)
    
    # Warmup
    for _ in range(10):
        _ = modelo_pequeno(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(100):
        _ = modelo_pequeno(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    tiempo_gpu = time.perf_counter() - start
    print(f"Tiempo GPU: {tiempo_gpu:.3f}s (100 batches)")
    
    # Forzar modo híbrido para comparar
    print("\n[2] Mismo modelo en modo HÍBRIDO (forzado):")
    modelo_pequeno.cambiar_modo('hibrido')
    
    # Warmup
    for _ in range(10):
        _ = modelo_pequeno(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(100):
        _ = modelo_pequeno(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    tiempo_hibrido = time.perf_counter() - start
    print(f"Tiempo híbrido: {tiempo_hibrido:.3f}s (100 batches)")
    
    print(f"\n→ GPU es {tiempo_hibrido/tiempo_gpu:.1f}x más rápido para modelo pequeño")
    
    # Crear modelo GRANDE (forzará híbrido pero manejable)
    print("\n" + "=" * 70)
    print("[3] Modelo GRANDE (forzará modo HÍBRIDO):")
    config_grande = {
        'hidden_size': 2048,
        'input_size': 784,
        'output_size': 10,
        'num_worlds': 16  # Suficientes mundos para exceder VRAM
    }
    modelo_grande = LLARRI_O1_Adaptativo(config_grande)
    modelo_grande.info()
    
    # Probar que funciona
    print("Probando inferencia en modo híbrido...")
    try:
        x_test = torch.randn(8, 784)  # Batch pequeño para híbrido
        output = modelo_grande(x_test)
        print(f"✅ Inferencia exitosa! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print("""
✅ El modelo ahora detecta automáticamente:
   - VRAM disponible
   - Tamaño del modelo
   - Modo óptimo (GPU o híbrido)

✅ Si cabe en VRAM → GPU (máxima velocidad)
✅ Si NO cabe → Híbrido (funciona igual, más lento)

✅ Puedes forzar modo con: modelo.cambiar_modo('hibrido')
""")
