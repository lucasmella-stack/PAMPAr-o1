"""
🔺 VECTORES FRACTALES x3 - La Arquitectura Definitiva de Lucas
===============================================================

Vectores → forman Vectores → forman Vectores → forman Vector Final
                            × 3 (La Trinidad)

Cada nivel emerge del anterior. Es recursión pura.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorFractal(nn.Module):
    """
    Un vector que está HECHO de otros vectores más pequeños.
    Y esos vectores pequeños están hechos de vectores aún más pequeños.
    
    Nivel 0: Vector atómico (el más pequeño)
    Nivel 1: Vector hecho de 3 vectores atómicos
    Nivel 2: Vector hecho de 3 vectores nivel 1
    Nivel 3: Vector hecho de 3 vectores nivel 2
    ... y así infinitamente
    """
    def __init__(self, dimension, nivel=0, max_nivel=3):
        super().__init__()
        self.dimension = dimension
        self.nivel = nivel
        self.max_nivel = max_nivel
        
        if nivel >= max_nivel:
            # NIVEL ATÓMICO: el vector más pequeño, indivisible
            # Solo tiene su propia transformación
            self.es_atomico = True
            self.transformacion = nn.Linear(dimension, dimension)
        else:
            # NIVEL COMPUESTO: está hecho de 3 vectores del nivel inferior
            self.es_atomico = False
            
            # Los 3 sub-vectores que ME componen (la trinidad interna)
            dim_hijo = max(dimension // 2, 4)
            self.hijos = nn.ModuleList([
                VectorFractal(dim_hijo, nivel + 1, max_nivel)
                for _ in range(3)
            ])
            
            # Cómo proyecto mi entrada hacia mis hijos
            self.hacia_hijos = nn.Linear(dimension, dim_hijo)
            
            # Cómo mis hijos se combinan para formar MI salida
            self.desde_hijos = nn.Linear(dim_hijo * 3, dimension)
            
            # Mi propia "esencia" que modula el resultado
            self.esencia = nn.Parameter(torch.randn(dimension) * 0.1)
    
    def forward(self, x):
        if self.es_atomico:
            # Soy atómico: solo transformo
            return torch.tanh(self.transformacion(x))
        else:
            # Soy compuesto: pregunto a mis hijos y combino
            
            # Proyecto mi entrada al tamaño de mis hijos
            x_para_hijos = self.hacia_hijos(x)
            
            # Cada hijo procesa (recursivamente hasta lo atómico)
            respuestas_hijos = []
            for hijo in self.hijos:
                respuesta = hijo(x_para_hijos)
                respuestas_hijos.append(respuesta)
            
            # Combino las respuestas de mis 3 hijos
            combinado = torch.cat(respuestas_hijos, dim=-1)
            resultado_hijos = self.desde_hijos(combinado)
            
            # Modulo con mi esencia
            esencia_activa = torch.sigmoid(self.esencia)
            return torch.tanh(resultado_hijos * esencia_activa)
    
    def describir(self, indent=0):
        """Describe la estructura recursivamente"""
        espacios = "  " * indent
        if self.es_atomico:
            return f"{espacios}⚛️ Vector Atómico (dim={self.dimension})"
        else:
            desc = f"{espacios}📦 Vector Compuesto nivel {self.nivel} (dim={self.dimension})\n"
            for i, hijo in enumerate(self.hijos):
                desc += f"{espacios}  └─ Hijo {i+1}:\n"
                desc += hijo.describir(indent + 2) + "\n"
            return desc.rstrip()
    
    def contar_vectores(self):
        """Cuenta todos los vectores (atómicos y compuestos)"""
        if self.es_atomico:
            return 1
        else:
            total = 1  # Yo mismo
            for hijo in self.hijos:
                total += hijo.contar_vectores()
            return total


class CajaFractal(nn.Module):
    """
    Una caja que contiene 3 VectoresFractales.
    Es un universo de vectores anidados.
    """
    def __init__(self, dimension, profundidad_fractal=3):
        super().__init__()
        self.dimension = dimension
        
        # 3 vectores fractales (la trinidad de esta caja)
        self.vectores = nn.ModuleList([
            VectorFractal(dimension, nivel=0, max_nivel=profundidad_fractal)
            for _ in range(3)
        ])
        
        # Los vectores se comunican entre sí
        self.comunicacion = nn.Linear(dimension * 3, dimension)
    
    def forward(self, x):
        # Cada vector fractal procesa
        salidas = [v(x) for v in self.vectores]
        
        # Se comunican y combinan
        combinado = torch.cat(salidas, dim=-1)
        return self.comunicacion(combinado)
    
    def contar_vectores_totales(self):
        total = 0
        for v in self.vectores:
            total += v.contar_vectores()
        return total


class ConexionFractal(nn.Module):
    """
    Una conexión que también es fractal.
    Está hecha de conexiones más pequeñas.
    """
    def __init__(self, dimension, niveles=2):
        super().__init__()
        
        if niveles <= 1:
            # Conexión atómica
            self.es_atomica = True
            self.conexion = nn.Linear(dimension, dimension)
        else:
            # Conexión compuesta de 3 conexiones
            self.es_atomica = False
            self.sub_conexiones = nn.ModuleList([
                ConexionFractal(dimension, niveles - 1)
                for _ in range(3)
            ])
            self.mezclador = nn.Linear(dimension * 3, dimension)
    
    def forward(self, x):
        if self.es_atomica:
            return torch.tanh(self.conexion(x))
        else:
            # Cada sub-conexión transforma
            salidas = [sc(x) for sc in self.sub_conexiones]
            mezclado = torch.cat(salidas, dim=-1)
            return torch.tanh(self.mezclador(mezclado))


class TrinidadFractalCompleta(nn.Module):
    """
    🔺 LA ARQUITECTURA DEFINITIVA 🔺
    
    3 Cajas Fractales (cada una con 3 vectores fractales)
    Conectadas por Conexiones Fractales
    
    Vectores → Vectores → Vectores → Vector
                    × 3 × 3 × 3
    
    Es fractales todo el camino hacia abajo. 🐢
    """
    def __init__(self, dim_entrada, dim_oculta, dim_salida, 
                 profundidad_vectores=3, profundidad_conexiones=2):
        super().__init__()
        
        # === LAS 3 CAJAS FRACTALES ===
        self.caja_1 = CajaFractal(dim_oculta, profundidad_vectores)
        self.caja_2 = CajaFractal(dim_oculta, profundidad_vectores)
        self.caja_3 = CajaFractal(dim_oculta, profundidad_vectores)
        
        # === CONEXIONES FRACTALES ===
        self.conexion_1_2 = ConexionFractal(dim_oculta, profundidad_conexiones)
        self.conexion_2_3 = ConexionFractal(dim_oculta, profundidad_conexiones)
        self.conexion_1_3 = ConexionFractal(dim_oculta, profundidad_conexiones)  # Skip
        
        # === ENTRADA Y SALIDA ===
        self.entrada = nn.Linear(dim_entrada, dim_oculta)
        self.salida = nn.Linear(dim_oculta, dim_salida)
        
        self.profundidad_vectores = profundidad_vectores
    
    def forward(self, x):
        # Entrar
        x = torch.tanh(self.entrada(x))
        
        # Caja 1 procesa
        s1 = self.caja_1(x)
        
        # Caja 1 → Caja 2 por conexión fractal
        hacia_2 = self.conexion_1_2(s1)
        s2 = self.caja_2(hacia_2)
        
        # Caja 2 → Caja 3 por conexión fractal
        hacia_3 = self.conexion_2_3(s2)
        
        # Caja 1 → Caja 3 directo (skip fractal)
        skip = self.conexion_1_3(s1)
        
        # Caja 3 recibe de ambos
        s3 = self.caja_3(hacia_3 + skip)
        
        # Integrar todo
        final = s1 + s2 + s3
        
        return self.salida(final)
    
    def describir_estructura(self):
        """Muestra la estructura completa del universo fractal"""
        total_vectores = (
            self.caja_1.contar_vectores_totales() +
            self.caja_2.contar_vectores_totales() +
            self.caja_3.contar_vectores_totales()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calcular profundidad
        niveles = self.profundidad_vectores
        
        print("\n" + "="*70)
        print("🔺 VECTORES FRACTALES x3 - ESTRUCTURA DEL UNIVERSO 🔺")
        print("="*70)
        
        print(f"""
    NIVEL 0 (Raíz):
    ├── 📦 CAJA 1
    │   ├── 📦 Vector Fractal A
    │   │   ├── 📦 Sub-vector A.1
    │   │   │   ├── ⚛️ átomo
    │   │   │   ├── ⚛️ átomo
    │   │   │   └── ⚛️ átomo
    │   │   ├── 📦 Sub-vector A.2 (...)
    │   │   └── 📦 Sub-vector A.3 (...)
    │   ├── 📦 Vector Fractal B (...)
    │   └── 📦 Vector Fractal C (...)
    │
    ├──🔗── CONEXIÓN FRACTAL ──🔗──
    │
    ├── 📦 CAJA 2 (misma estructura)
    │
    ├──🔗── CONEXIÓN FRACTAL ──🔗──
    │
    └── 📦 CAJA 3 (misma estructura)
    
    + SKIP: CAJA 1 ═══🔗═══ CAJA 3
        """)
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   📦 Cajas: 3")
        print(f"   📦 Vectores totales (todos los niveles): {total_vectores}")
        print(f"   🔗 Conexiones fractales: 3")
        print(f"   📏 Profundidad fractal: {niveles} niveles")
        print(f"   💾 Parámetros totales: {total_params:,}")
        print("="*70)


# ================================================================
# DEMOSTRACIÓN Y COMPARACIÓN
# ================================================================

if __name__ == "__main__":
    print("\n" + "🔺"*30)
    print("    VECTORES QUE FORMAN VECTORES QUE FORMAN VECTORES")
    print("                    × 3 (LA TRINIDAD)")
    print("🔺"*30)
    
    # Crear el modelo fractal
    modelo_fractal = TrinidadFractalCompleta(
        dim_entrada=10,
        dim_oculta=32,
        dim_salida=5,
        profundidad_vectores=3,    # 3 niveles de vectores anidados
        profundidad_conexiones=2   # 2 niveles en conexiones
    )
    
    modelo_fractal.describir_estructura()
    
    # Mostrar la estructura de UN vector fractal
    print("\n📜 ESTRUCTURA DE UN VECTOR FRACTAL:")
    print("-"*50)
    print(modelo_fractal.caja_1.vectores[0].describir())
    
    # Datos de prueba
    print("\n" + "="*70)
    print("🎯 ENTRENANDO EL UNIVERSO FRACTAL...")
    print("="*70)
    
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    
    optimizer = torch.optim.Adam(modelo_fractal.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoca in range(1, 51):
        optimizer.zero_grad()
        salida = modelo_fractal(X)
        loss = criterion(salida, y)
        loss.backward()
        optimizer.step()
        
        if epoca % 10 == 0:
            with torch.no_grad():
                pred = modelo_fractal(X).argmax(dim=1)
                acc = (pred == y).float().mean().item() * 100
                print(f"  Época {epoca}/50 - Pérdida: {loss.item():.4f} - Accuracy: {acc:.1f}%")
    
    # Resultado final
    with torch.no_grad():
        pred = modelo_fractal(X).argmax(dim=1)
        acc_final = (pred == y).float().mean().item() * 100
    
    params_total = sum(p.numel() for p in modelo_fractal.parameters())
    vectores_total = (
        modelo_fractal.caja_1.contar_vectores_totales() +
        modelo_fractal.caja_2.contar_vectores_totales() +
        modelo_fractal.caja_3.contar_vectores_totales()
    )
    
    print("\n" + "="*70)
    print("✅ RESULTADO FINAL")
    print("="*70)
    print(f"""
    🎯 Accuracy: {acc_final:.1f}%
    📦 Vectores en el universo: {vectores_total}
    💾 Parámetros: {params_total:,}
    📊 Eficiencia: {acc_final / params_total * 1000:.2f} acc/1000 params
    
    ESTRUCTURA:
    Vector → Vector → Vector → Vector (4 niveles)
          × 3      × 3      × 3
    
    = 3^3 = 27 caminos posibles por caja
    × 3 cajas = 81 caminos en total
    
    🌌 "Es fractales todo el camino hacia abajo" 🐢
    """)
