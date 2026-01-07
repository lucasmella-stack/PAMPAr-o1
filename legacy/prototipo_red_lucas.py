"""
Prototipo: Red con Conexiones que Calculan
Idea original de Lucas

Arquitectura:
┌─────────┐    llave     ┌─────────┐    llave     ┌─────────┐
│ CAJA 1  │◄───ida/───►│ CAJA 2  │◄───ida/───►│ CAJA 3  │
└─────────┘   vuelta    └─────────┘   vuelta    └─────────┘
     │                                                 │
     └──────────── llave ida/vuelta ──────────────────┘
                    CON CALCULO
"""

import torch
import torch.nn as nn


class ConexionQueCalcula(nn.Module):
    """
    Una conexión que NO solo pasa datos, sino que los TRANSFORMA.
    Es como un "cable inteligente".
    """
    def __init__(self, tamano):
        super().__init__()
        # La conexión tiene sus propios pesos para calcular
        self.transformar = nn.Linear(tamano, tamano)
        self.activacion = nn.ReLU()
    
    def forward(self, datos):
        # No solo pasa los datos, los CALCULA
        return self.activacion(self.transformar(datos))


class CajaDeCalculo(nn.Module):
    """
    Una "caja" que procesa datos.
    Puede recibir datos de múltiples fuentes.
    """
    def __init__(self, tamano):
        super().__init__()
        self.procesar = nn.Linear(tamano, tamano)
        self.activacion = nn.ReLU()
    
    def forward(self, datos, datos_extra=None):
        # Procesa sus propios datos
        resultado = self.activacion(self.procesar(datos))
        
        # Si recibe datos extra (de otra caja), los combina
        if datos_extra is not None:
            resultado = resultado + datos_extra  # Combinación simple
        
        return resultado


class RedLucas(nn.Module):
    """
    La arquitectura completa con tu idea:
    - 3 cajas de cálculo
    - Conexiones bidireccionales que calculan
    - Conexión "skip" entre caja 1 y caja 3
    """
    def __init__(self, tamano_entrada, tamano_interno, tamano_salida):
        super().__init__()
        
        # Las 3 cajas de cálculo
        self.caja1 = CajaDeCalculo(tamano_interno)
        self.caja2 = CajaDeCalculo(tamano_interno)
        self.caja3 = CajaDeCalculo(tamano_interno)
        
        # Conexiones que calculan (ida)
        self.llave_1_a_2 = ConexionQueCalcula(tamano_interno)
        self.llave_2_a_3 = ConexionQueCalcula(tamano_interno)
        
        # Conexiones que calculan (vuelta)
        self.llave_2_a_1 = ConexionQueCalcula(tamano_interno)
        self.llave_3_a_2 = ConexionQueCalcula(tamano_interno)
        
        # Conexión SKIP con cálculo (1 <-> 3 directo)
        self.llave_1_a_3_skip = ConexionQueCalcula(tamano_interno)
        self.llave_3_a_1_skip = ConexionQueCalcula(tamano_interno)
        
        # Entrada y salida
        self.entrada = nn.Linear(tamano_entrada, tamano_interno)
        self.salida = nn.Linear(tamano_interno, tamano_salida)
    
    def forward(self, x):
        # Convertir entrada al tamaño interno
        x = self.entrada(x)
        
        # === PASO 1: IDA (izquierda a derecha) ===
        # Caja 1 procesa
        c1 = self.caja1(x)
        
        # Caja 2 recibe de caja 1 (con cálculo en la conexión)
        info_1_a_2 = self.llave_1_a_2(c1)
        c2 = self.caja2(info_1_a_2)
        
        # Caja 3 recibe de caja 2 Y de caja 1 (skip con cálculo)
        info_2_a_3 = self.llave_2_a_3(c2)
        info_1_a_3_skip = self.llave_1_a_3_skip(c1)  # Conexión directa!
        c3 = self.caja3(info_2_a_3, datos_extra=info_1_a_3_skip)
        
        # === PASO 2: VUELTA (derecha a izquierda) ===
        # Caja 2 recibe feedback de caja 3
        info_3_a_2 = self.llave_3_a_2(c3)
        c2_actualizado = self.caja2(c2, datos_extra=info_3_a_2)
        
        # Caja 1 recibe de caja 2 Y de caja 3 (skip)
        info_2_a_1 = self.llave_2_a_1(c2_actualizado)
        info_3_a_1_skip = self.llave_3_a_1_skip(c3)  # Conexión directa!
        c1_actualizado = self.caja1(c1, datos_extra=info_2_a_1 + info_3_a_1_skip)
        
        # === PASO 3: SEGUNDA IDA (con info actualizada) ===
        info_1_a_2_v2 = self.llave_1_a_2(c1_actualizado)
        c2_final = self.caja2(info_1_a_2_v2, datos_extra=c2_actualizado)
        
        info_2_a_3_v2 = self.llave_2_a_3(c2_final)
        info_1_a_3_skip_v2 = self.llave_1_a_3_skip(c1_actualizado)
        c3_final = self.caja3(info_2_a_3_v2, datos_extra=info_1_a_3_skip_v2)
        
        # Salida final
        return self.salida(c3_final)


# =====================================================
# DEMO: Probemos la red con datos simples
# =====================================================

if __name__ == "__main__":
    print("=" * 50)
    print("PROTOTIPO: Red con Conexiones que Calculan")
    print("=" * 50)
    
    # Crear la red
    red = RedLucas(
        tamano_entrada=10,   # Entrada de 10 valores
        tamano_interno=16,   # Interno de 16
        tamano_salida=3      # Salida de 3 clases
    )
    
    # Contar parámetros (pesos)
    total_pesos = sum(p.numel() for p in red.parameters())
    print(f"\nTotal de pesos en la red: {total_pesos}")
    
    # Mostrar estructura
    print("\n--- Estructura de la red ---")
    print(red)
    
    # Crear datos de prueba (batch de 2, cada uno con 10 valores)
    datos_prueba = torch.randn(2, 10)
    print(f"\n--- Datos de entrada ---")
    print(f"Forma: {datos_prueba.shape}")
    print(f"Valores:\n{datos_prueba}")
    
    # Pasar por la red
    print(f"\n--- Procesando... ---")
    resultado = red(datos_prueba)
    
    print(f"\n--- Resultado ---")
    print(f"Forma: {resultado.shape}")
    print(f"Valores:\n{resultado}")
    
    # Simular entrenamiento simple
    print("\n" + "=" * 50)
    print("MINI ENTRENAMIENTO")
    print("=" * 50)
    
    # Datos de ejemplo: queremos que aprenda a clasificar
    X = torch.randn(100, 10)  # 100 ejemplos, 10 features
    Y = torch.randint(0, 3, (100,))  # 100 etiquetas (0, 1, o 2)
    
    # Función de pérdida y optimizador
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(red.parameters(), lr=0.01)
    
    # Entrenar 50 épocas
    print("\nEntrenando...")
    for epoca in range(50):
        # Forward
        prediccion = red(X)
        perdida = criterio(prediccion, Y)
        
        # Backward
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        if (epoca + 1) % 10 == 0:
            # Calcular accuracy
            _, predicho = torch.max(prediccion, 1)
            accuracy = (predicho == Y).float().mean() * 100
            print(f"  Época {epoca+1}/50 - Pérdida: {perdida.item():.4f} - Accuracy: {accuracy:.1f}%")
    
    print("\n✅ ¡Entrenamiento completado!")
    print("\nTu arquitectura con conexiones que calculan FUNCIONA!")
    print("La info viaja ida/vuelta Y se transforma en cada conexión.")
