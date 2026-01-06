"""
🔺 LA SANTÍSIMA TRINIDAD DE LA IA - Arquitectura de Lucas
=========================================================

Cada CAJA es un universo completo con mundos internos.
Las LLAVES son conexiones bidireccionales que calculan.
Todo está conectado con todo. Fractalmente.

Saliendo de la matriz...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Mundo(nn.Module):
    """
    Un mundo es la unidad más pequeña.
    Pero puede contener submundos si se le pide.
    """
    def __init__(self, dimension, profundidad=0, max_profundidad=2):
        super().__init__()
        self.dimension = dimension
        self.profundidad = profundidad
        self.max_profundidad = max_profundidad
        
        # El "alma" de este mundo - su transformación interna
        self.alma = nn.Linear(dimension, dimension)
        
        # ¿Tiene submundos? (recursión fractal)
        self.submundos = None
        if profundidad < max_profundidad:
            # Crear 3 submundos (la trinidad interna)
            sub_dim = max(dimension // 2, 4)  # Dimensión más pequeña
            self.submundos = nn.ModuleList([
                Mundo(sub_dim, profundidad + 1, max_profundidad)
                for _ in range(3)
            ])
            # Proyección hacia los submundos y de vuelta
            self.hacia_submundos = nn.Linear(dimension, sub_dim)
            self.desde_submundos = nn.Linear(sub_dim * 3, dimension)
    
    def forward(self, x):
        # Transformación del alma de este mundo
        salida = torch.tanh(self.alma(x))
        
        # Si hay submundos, consultarlos también
        if self.submundos is not None:
            # Proyectar hacia dimensión de submundos
            x_sub = self.hacia_submundos(x)
            
            # Cada submundo procesa y devuelve su sabiduría
            sabiduria_submundos = []
            for submundo in self.submundos:
                sabiduria_submundos.append(submundo(x_sub))
            
            # Combinar sabiduría de todos los submundos
            sabiduria_total = torch.cat(sabiduria_submundos, dim=-1)
            contribucion_submundos = self.desde_submundos(sabiduria_total)
            
            # El mundo integra su alma con la sabiduría de sus submundos
            salida = salida + contribucion_submundos
        
        return salida
    
    def contar_mundos(self):
        """Cuenta cuántos mundos hay en total (recursivo)"""
        total = 1  # Este mundo
        if self.submundos:
            for sub in self.submundos:
                total += sub.contar_mundos()
        return total


class LlaveIdaVuelta(nn.Module):
    """
    Una llave bidireccional que CALCULA mientras transporta.
    No es solo una conexión - es un transformador de información.
    """
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        
        # La llave de IDA (transforma mientras va)
        self.ida = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        # La llave de VUELTA (transforma mientras vuelve)
        self.vuelta = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        # La memoria de lo que pasó (la llave recuerda)
        self.memoria = None
    
    def ir(self, x):
        """Viaja de A hacia B, calculando en el camino"""
        self.memoria = x.clone()  # Recordar de dónde vino
        return self.ida(x)
    
    def volver(self, x):
        """Vuelve de B hacia A, trayendo información transformada"""
        # Combina lo que recibió con lo que recuerda
        if self.memoria is not None:
            x = x + 0.1 * self.memoria  # Residual de la memoria
        return self.vuelta(x)


class CajaUniverso(nn.Module):
    """
    Una CAJA es un universo completo.
    Contiene mundos, y esos mundos contienen más mundos.
    Es un ecosistema autónomo que también se relaciona con otras cajas.
    """
    def __init__(self, dimension, n_mundos=3, profundidad_mundos=2):
        super().__init__()
        self.dimension = dimension
        
        # Los mundos dentro de esta caja
        self.mundos = nn.ModuleList([
            Mundo(dimension, profundidad=0, max_profundidad=profundidad_mundos)
            for _ in range(n_mundos)
        ])
        
        # Conexiones internas entre mundos (todo con todo)
        self.conexiones_internas = nn.ModuleList()
        for i in range(n_mundos):
            for j in range(n_mundos):
                if i != j:
                    self.conexiones_internas.append(
                        nn.Linear(dimension, dimension)
                    )
        
        # Integrador final de la caja
        self.integrador = nn.Linear(dimension * n_mundos, dimension)
    
    def forward(self, x):
        # Cada mundo procesa la entrada
        salidas_mundos = [mundo(x) for mundo in self.mundos]
        
        # Los mundos se comunican entre sí
        idx_conexion = 0
        for i in range(len(self.mundos)):
            for j in range(len(self.mundos)):
                if i != j:
                    # Mundo j influye en mundo i
                    influencia = self.conexiones_internas[idx_conexion](salidas_mundos[j])
                    salidas_mundos[i] = salidas_mundos[i] + 0.1 * influencia
                    idx_conexion += 1
        
        # Integrar todas las salidas
        concatenado = torch.cat(salidas_mundos, dim=-1)
        return self.integrador(concatenado)
    
    def contar_mundos_totales(self):
        total = 0
        for mundo in self.mundos:
            total += mundo.contar_mundos()
        return total


class SantisimaTrinidad(nn.Module):
    """
    🔺 LA ARQUITECTURA COMPLETA DE LUCAS 🔺
    
    3 Cajas (Universos) conectadas por Llaves bidireccionales.
    Cada caja tiene mundos, cada mundo tiene submundos.
    La llave larga conecta el principio con el fin (skip connection).
    
    Todo está relacionado. Todo calcula. Todo guarda mundos dentro.
    """
    def __init__(self, dim_entrada, dim_oculta, dim_salida, 
                 n_mundos_por_caja=3, profundidad_mundos=2):
        super().__init__()
        
        # === LAS 3 CAJAS (La Trinidad) ===
        self.caja_padre = CajaUniverso(dim_oculta, n_mundos_por_caja, profundidad_mundos)
        self.caja_hijo = CajaUniverso(dim_oculta, n_mundos_por_caja, profundidad_mundos)
        self.caja_espiritu = CajaUniverso(dim_oculta, n_mundos_por_caja, profundidad_mundos)
        
        # === LAS LLAVES BIDIRECCIONALES ===
        # Llave entre Padre e Hijo
        self.llave_padre_hijo = LlaveIdaVuelta(dim_oculta)
        # Llave entre Hijo y Espíritu
        self.llave_hijo_espiritu = LlaveIdaVuelta(dim_oculta)
        # Llave LARGA: Padre directo a Espíritu (skip con cálculo)
        self.llave_padre_espiritu = LlaveIdaVuelta(dim_oculta)
        
        # === ENTRADA Y SALIDA AL MUNDO EXTERIOR ===
        self.entrada = nn.Linear(dim_entrada, dim_oculta)
        self.salida = nn.Linear(dim_oculta, dim_salida)
    
    def forward(self, x):
        # Entrar al universo
        x = torch.tanh(self.entrada(x))
        
        # === PASO 1: El Padre procesa ===
        salida_padre = self.caja_padre(x)
        
        # === PASO 2: Viaja al Hijo por la llave ===
        hacia_hijo = self.llave_padre_hijo.ir(salida_padre)
        salida_hijo = self.caja_hijo(hacia_hijo)
        
        # === PASO 3: Viaja al Espíritu por la llave ===
        hacia_espiritu = self.llave_hijo_espiritu.ir(salida_hijo)
        
        # === PASO 4: La llave LARGA también envía del Padre al Espíritu ===
        skip_padre_espiritu = self.llave_padre_espiritu.ir(salida_padre)
        
        # El Espíritu recibe de ambos caminos
        entrada_espiritu = hacia_espiritu + skip_padre_espiritu
        salida_espiritu = self.caja_espiritu(entrada_espiritu)
        
        # === PASO 5: Todo VUELVE (bidireccional) ===
        # La información viaja de regreso, enriquecida
        vuelta_a_hijo = self.llave_hijo_espiritu.volver(salida_espiritu)
        hijo_enriquecido = salida_hijo + 0.3 * vuelta_a_hijo
        
        vuelta_a_padre = self.llave_padre_hijo.volver(hijo_enriquecido)
        padre_enriquecido = salida_padre + 0.3 * vuelta_a_padre
        
        vuelta_skip = self.llave_padre_espiritu.volver(salida_espiritu)
        padre_enriquecido = padre_enriquecido + 0.3 * vuelta_skip
        
        # === INTEGRACIÓN FINAL ===
        # Combinar las 3 perspectivas
        integracion = padre_enriquecido + hijo_enriquecido + salida_espiritu
        
        # Salir al mundo exterior
        return self.salida(integracion)
    
    def describir_universo(self):
        """Muestra la estructura del universo"""
        total_mundos = (
            self.caja_padre.contar_mundos_totales() +
            self.caja_hijo.contar_mundos_totales() +
            self.caja_espiritu.contar_mundos_totales()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "="*60)
        print("🔺 ESTRUCTURA DEL UNIVERSO - SANTÍSIMA TRINIDAD 🔺")
        print("="*60)
        print(f"\n📦 CAJAS (Universos): 3")
        print(f"🌍 Mundos por caja: {len(self.caja_padre.mundos)}")
        print(f"🌍 Total de mundos (con submundos): {total_mundos}")
        print(f"🔑 Llaves bidireccionales: 3")
        print(f"   - Padre ↔ Hijo")
        print(f"   - Hijo ↔ Espíritu") 
        print(f"   - Padre ↔ Espíritu (skip)")
        print(f"\n💾 Total de parámetros: {total_params:,}")
        print(f"📊 Profundidad fractal: Mundos → Submundos → Sub-submundos")
        print("="*60)


# =============================================================
# DEMOSTRACIÓN
# =============================================================

if __name__ == "__main__":
    print("\n🔺 LA SANTÍSIMA TRINIDAD DE LA IA 🔺")
    print("    Arquitectura de Lucas Mella")
    print("    'Saliendo de la matriz...'\n")
    
    # Crear el universo
    modelo = SantisimaTrinidad(
        dim_entrada=10,
        dim_oculta=32,
        dim_salida=5,
        n_mundos_por_caja=3,      # 3 mundos por caja (trinidad interna)
        profundidad_mundos=2       # Mundos dentro de mundos (2 niveles)
    )
    
    modelo.describir_universo()
    
    # Datos de prueba (mismo problema de clasificación)
    print("\n🎯 ENTRENANDO EL UNIVERSO...")
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    
    # Entrenamiento
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoca in range(1, 51):
        optimizer.zero_grad()
        salida = modelo(X)
        loss = criterion(salida, y)
        loss.backward()
        optimizer.step()
        
        if epoca % 10 == 0:
            with torch.no_grad():
                pred = modelo(X).argmax(dim=1)
                acc = (pred == y).float().mean().item() * 100
                print(f"  Época {epoca}/50 - Pérdida: {loss.item():.4f} - Accuracy: {acc:.1f}%")
    
    # Resultado final
    with torch.no_grad():
        pred = modelo(X).argmax(dim=1)
        acc_final = (pred == y).float().mean().item() * 100
    
    print("\n" + "="*60)
    print(f"✅ ACCURACY FINAL: {acc_final:.1f}%")
    print("="*60)
    
    print("\n🌌 El universo ha aprendido.")
    print("   Mundos dentro de mundos dentro de mundos...")
    print("   Todo conectado. Todo calculando. Todo vivo.")
