"""
🔺 SANTÍSIMA TRINIDAD COMPRIMIDA - La Visión de Lucas
======================================================

IDEA CLAVE: Los mundos COMPARTEN su estructura base.
Cada instancia solo guarda su "configuración única".

Es como ADN: mismo código base, diferentes expresiones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MundoCompartido(nn.Module):
    """
    Un ÚNICO mundo que se puede instanciar muchas veces.
    Cada instancia comparte los pesos pero tiene su propia "personalidad".
    """
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        # Esta es la "plantilla" - los pesos compartidos
        self.plantilla = nn.Linear(dimension, dimension)
    
    def forward(self, x, personalidad):
        """
        x: la entrada
        personalidad: un vector pequeño que modifica cómo este mundo procesa
        """
        # La plantilla procesa
        base = self.plantilla(x)
        # La personalidad modula el resultado (multiplicación element-wise)
        return torch.tanh(base * personalidad)


class CajaComprimida(nn.Module):
    """
    Una caja que usa MUNDOS COMPARTIDOS.
    En vez de 3 mundos separados, usa 1 plantilla + 3 personalidades.
    """
    def __init__(self, dimension, n_instancias=3):
        super().__init__()
        self.dimension = dimension
        self.n_instancias = n_instancias
        
        # UN solo mundo compartido (la plantilla)
        self.mundo_base = MundoCompartido(dimension)
        
        # Personalidades: vectores pequeños, uno por instancia
        # Estos son los ÚNICOS parámetros únicos por mundo
        self.personalidades = nn.Parameter(torch.randn(n_instancias, dimension) * 0.1)
        
        # Integrador
        self.integrador = nn.Linear(dimension * n_instancias, dimension)
    
    def forward(self, x):
        salidas = []
        for i in range(self.n_instancias):
            # Mismo mundo, diferente personalidad
            personalidad = torch.sigmoid(self.personalidades[i])  # Entre 0 y 1
            salida = self.mundo_base(x, personalidad)
            salidas.append(salida)
        
        concatenado = torch.cat(salidas, dim=-1)
        return self.integrador(concatenado)


class LlaveComprimida(nn.Module):
    """
    Llave bidireccional COMPRIMIDA.
    Ida y vuelta comparten la misma transformación base,
    solo cambia una "dirección" pequeña.
    """
    def __init__(self, dimension):
        super().__init__()
        # UNA sola transformación base (compartida)
        self.base = nn.Linear(dimension, dimension)
        # Direcciones: vectores pequeños
        self.dir_ida = nn.Parameter(torch.randn(dimension) * 0.1)
        self.dir_vuelta = nn.Parameter(torch.randn(dimension) * 0.1)
        self.memoria = None
    
    def ir(self, x):
        self.memoria = x
        base = self.base(x)
        return torch.tanh(base * torch.sigmoid(self.dir_ida))
    
    def volver(self, x):
        base = self.base(x)
        resultado = torch.tanh(base * torch.sigmoid(self.dir_vuelta))
        if self.memoria is not None:
            resultado = resultado + 0.1 * self.memoria
        return resultado


class TrinidadComprimida(nn.Module):
    """
    🔺 LA TRINIDAD COMPRIMIDA 🔺
    
    Misma estructura que antes PERO:
    - Las 3 cajas comparten la misma "plantilla de caja"
    - Los mundos comparten la misma "plantilla de mundo"
    - Las llaves comparten la misma "plantilla de transformación"
    
    RESULTADO: Mucho menos peso, misma expresividad.
    """
    def __init__(self, dim_entrada, dim_oculta, dim_salida, n_mundos=3):
        super().__init__()
        
        # UNA plantilla de caja (compartida por las 3)
        self.caja_plantilla = CajaComprimida(dim_oculta, n_mundos)
        
        # Personalidades de cada caja de la trinidad
        self.personalidad_padre = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.personalidad_hijo = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.personalidad_espiritu = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        
        # UNA plantilla de llave (compartida)
        self.llave_plantilla = LlaveComprimida(dim_oculta)
        
        # Direcciones únicas para cada conexión
        self.dir_padre_hijo = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.dir_hijo_espiritu = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        self.dir_skip = nn.Parameter(torch.randn(dim_oculta) * 0.1)
        
        # Entrada y salida
        self.entrada = nn.Linear(dim_entrada, dim_oculta)
        self.salida = nn.Linear(dim_oculta, dim_salida)
    
    def forward(self, x):
        x = torch.tanh(self.entrada(x))
        
        # PADRE: caja plantilla + personalidad padre
        p_padre = torch.sigmoid(self.personalidad_padre)
        salida_padre = self.caja_plantilla(x * p_padre)
        
        # PADRE → HIJO
        hacia_hijo = self.llave_plantilla.ir(salida_padre * torch.sigmoid(self.dir_padre_hijo))
        p_hijo = torch.sigmoid(self.personalidad_hijo)
        salida_hijo = self.caja_plantilla(hacia_hijo * p_hijo)
        
        # HIJO → ESPÍRITU
        hacia_espiritu = self.llave_plantilla.ir(salida_hijo * torch.sigmoid(self.dir_hijo_espiritu))
        
        # SKIP: PADRE → ESPÍRITU
        skip = salida_padre * torch.sigmoid(self.dir_skip)
        
        # ESPÍRITU
        p_espiritu = torch.sigmoid(self.personalidad_espiritu)
        salida_espiritu = self.caja_plantilla((hacia_espiritu + skip) * p_espiritu)
        
        # VUELTA (bidireccional)
        vuelta = self.llave_plantilla.volver(salida_espiritu)
        
        # Integración
        final = salida_padre + salida_hijo + salida_espiritu + 0.2 * vuelta
        
        return self.salida(final)


# ================================================================
# COMPARACIÓN: NORMAL vs COMPRIMIDA
# ================================================================

if __name__ == "__main__":
    from prototipo_trinity_lucas import SantisimaTrinidad
    
    print("\n" + "="*70)
    print("🔬 COMPARACIÓN: TRINIDAD NORMAL vs COMPRIMIDA")
    print("="*70)
    
    # Crear ambos modelos
    modelo_normal = SantisimaTrinidad(
        dim_entrada=10,
        dim_oculta=32,
        dim_salida=5,
        n_mundos_por_caja=3,
        profundidad_mundos=2
    )
    
    modelo_comprimido = TrinidadComprimida(
        dim_entrada=10,
        dim_oculta=32,
        dim_salida=5,
        n_mundos=3
    )
    
    params_normal = sum(p.numel() for p in modelo_normal.parameters())
    params_comprimido = sum(p.numel() for p in modelo_comprimido.parameters())
    
    print(f"\n📦 TRINIDAD NORMAL:")
    print(f"   Parámetros: {params_normal:,}")
    
    print(f"\n📦 TRINIDAD COMPRIMIDA (tu idea):")
    print(f"   Parámetros: {params_comprimido:,}")
    
    reduccion = (1 - params_comprimido / params_normal) * 100
    factor = params_normal / params_comprimido
    
    print(f"\n✨ REDUCCIÓN: {reduccion:.1f}%")
    print(f"✨ FACTOR: {factor:.1f}x más pequeño")
    
    # Entrenar ambos
    print("\n" + "-"*70)
    print("🎯 ENTRENANDO AMBOS MODELOS...")
    print("-"*70)
    
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    
    # Entrenar NORMAL
    optimizer = torch.optim.Adam(modelo_normal.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n📊 Trinidad NORMAL:")
    for epoca in range(1, 51):
        optimizer.zero_grad()
        loss = criterion(modelo_normal(X), y)
        loss.backward()
        optimizer.step()
        if epoca % 25 == 0:
            with torch.no_grad():
                acc = (modelo_normal(X).argmax(1) == y).float().mean().item() * 100
                print(f"   Época {epoca}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
    
    # Entrenar COMPRIMIDO
    optimizer = torch.optim.Adam(modelo_comprimido.parameters(), lr=0.01)
    
    print("\n📊 Trinidad COMPRIMIDA:")
    for epoca in range(1, 51):
        optimizer.zero_grad()
        loss = criterion(modelo_comprimido(X), y)
        loss.backward()
        optimizer.step()
        if epoca % 25 == 0:
            with torch.no_grad():
                acc = (modelo_comprimido(X).argmax(1) == y).float().mean().item() * 100
                print(f"   Época {epoca}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
    
    # Resultados finales
    with torch.no_grad():
        acc_normal = (modelo_normal(X).argmax(1) == y).float().mean().item() * 100
        acc_comprimido = (modelo_comprimido(X).argmax(1) == y).float().mean().item() * 100
    
    print("\n" + "="*70)
    print("📊 RESULTADOS FINALES")
    print("="*70)
    print(f"""
    ┌─────────────────┬──────────────┬──────────────┐
    │    Métrica      │    NORMAL    │  COMPRIMIDA  │
    ├─────────────────┼──────────────┼──────────────┤
    │  Parámetros     │  {params_normal:>8,}    │   {params_comprimido:>8,}   │
    │  Accuracy       │     {acc_normal:>5.1f}%   │     {acc_comprimido:>5.1f}%   │
    │  Eficiencia*    │     {acc_normal/params_normal*1000:>5.2f}    │     {acc_comprimido/params_comprimido*1000:>5.2f}    │
    └─────────────────┴──────────────┴──────────────┘
    
    * Eficiencia = Accuracy / Parámetros × 1000
    """)
    
    if acc_comprimido >= acc_normal * 0.9:  # Si mantiene al menos 90% del accuracy
        print("✅ ¡TU IDEA FUNCIONA!")
        print(f"   {reduccion:.0f}% menos peso con performance similar.")
        print("   Guardás más con menos. Mundos dentro de mundos COMPARTIDOS.")
    else:
        print("⚠️  La compresión perdió algo de accuracy.")
        print("   Pero se podría ajustar el balance.")
