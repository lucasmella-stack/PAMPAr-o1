# SPDX-License-Identifier: AGPL-3.0-or-later
"""
🔺 LLARRI-O1: La Santísima Trinidad Fractal
============================================
Arquitectura original de Lucas Mella

Vectores que forman vectores que forman vectores × 3
Mundos dentro de mundos dentro de mundos
Todo conectado. Todo calculando. Todo vivo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# ================================================================
# ARQUITECTURA LLARRI-O1
# ================================================================

class VectorFractal(nn.Module):
    """
    Vector que está hecho de otros vectores más pequeños.
    Recursión fractal con pesos compartidos opcionales.
    """
    def __init__(self, dimension, nivel=0, max_nivel=3, compartir_pesos=True):
        super().__init__()
        self.dimension = dimension
        self.nivel = nivel
        self.max_nivel = max_nivel
        self.compartir_pesos = compartir_pesos
        
        if nivel >= max_nivel:
            # Nivel atómico
            self.es_atomico = True
            self.transformacion = nn.Linear(dimension, dimension)
        else:
            self.es_atomico = False
            dim_hijo = max(dimension // 2, 8)
            
            if compartir_pesos:
                # Una plantilla compartida por los 3 hijos
                self.hijo_plantilla = VectorFractal(dim_hijo, nivel + 1, max_nivel, compartir_pesos)
                self.personalidades = nn.Parameter(torch.randn(3, dim_hijo) * 0.1)
            else:
                # 3 hijos independientes
                self.hijos = nn.ModuleList([
                    VectorFractal(dim_hijo, nivel + 1, max_nivel, compartir_pesos)
                    for _ in range(3)
                ])
            
            self.hacia_hijos = nn.Linear(dimension, dim_hijo)
            self.desde_hijos = nn.Linear(dim_hijo * 3, dimension)
            self.esencia = nn.Parameter(torch.randn(dimension) * 0.1)
    
    def forward(self, x):
        if self.es_atomico:
            return torch.tanh(self.transformacion(x))
        
        x_hijo = self.hacia_hijos(x)
        
        if self.compartir_pesos:
            respuestas = []
            for i in range(3):
                pers = torch.sigmoid(self.personalidades[i])
                resp = self.hijo_plantilla(x_hijo * pers)
                respuestas.append(resp)
        else:
            respuestas = [hijo(x_hijo) for hijo in self.hijos]
        
        combinado = torch.cat(respuestas, dim=-1)
        resultado = self.desde_hijos(combinado)
        return torch.tanh(resultado * torch.sigmoid(self.esencia))


class CajaTrinidad(nn.Module):
    """
    Una caja/universo que contiene 3 vectores fractales.
    """
    def __init__(self, dimension, profundidad=3, compartir_pesos=True):
        super().__init__()
        self.dimension = dimension
        
        if compartir_pesos:
            # Una plantilla de vector compartida
            self.vector_plantilla = VectorFractal(dimension, 0, profundidad, compartir_pesos)
            self.personalidades = nn.Parameter(torch.randn(3, dimension) * 0.1)
        else:
            self.vectores = nn.ModuleList([
                VectorFractal(dimension, 0, profundidad, compartir_pesos)
                for _ in range(3)
            ])
        
        self.compartir_pesos = compartir_pesos
        self.integrador = nn.Linear(dimension * 3, dimension)
    
    def forward(self, x):
        if self.compartir_pesos:
            salidas = []
            for i in range(3):
                pers = torch.sigmoid(self.personalidades[i])
                sal = self.vector_plantilla(x * pers)
                salidas.append(sal)
        else:
            salidas = [v(x) for v in self.vectores]
        
        return self.integrador(torch.cat(salidas, dim=-1))


class LlaveCalculadora(nn.Module):
    """
    Conexión bidireccional que calcula mientras transporta.
    """
    def __init__(self, dimension):
        super().__init__()
        self.base = nn.Linear(dimension, dimension)
        self.dir_ida = nn.Parameter(torch.randn(dimension) * 0.1)
        self.dir_vuelta = nn.Parameter(torch.randn(dimension) * 0.1)
        self.memoria = None
    
    def ir(self, x):
        self.memoria = x.detach()
        return torch.tanh(self.base(x) * torch.sigmoid(self.dir_ida))
    
    def volver(self, x):
        resultado = torch.tanh(self.base(x) * torch.sigmoid(self.dir_vuelta))
        if self.memoria is not None:
            resultado = resultado + 0.1 * self.memoria
        return resultado


class LlarriO1(nn.Module):
    """
    🔺 LLARRI-O1: La Santísima Trinidad Fractal 🔺
    
    Arquitectura original de Lucas Mella.
    
    Estructura:
    - 3 Cajas (Padre, Hijo, Espíritu) - La Trinidad
    - Cada caja tiene vectores fractales (mundos dentro de mundos)
    - Llaves bidireccionales que calculan
    - Skip connection con cálculo
    
    Parámetros configurables:
    - dim_entrada: dimensión de entrada
    - dim_oculta: dimensión de las capas ocultas
    - dim_salida: dimensión de salida
    - profundidad: niveles de recursión fractal (1-5)
    - compartir_pesos: si True, reduce parámetros significativamente
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Lucas Mella"
    LICENSE = "Proprietary - All rights reserved by Lucas Mella"
    
    def __init__(self, dim_entrada, dim_oculta, dim_salida, 
                 profundidad=3, compartir_pesos=True):
        super().__init__()
        
        self.config = {
            "model_type": "llarri-o1",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "license": self.LICENSE,
            "architecture": "Santísima Trinidad Fractal",
            "dim_entrada": dim_entrada,
            "dim_oculta": dim_oculta,
            "dim_salida": dim_salida,
            "profundidad": profundidad,
            "compartir_pesos": compartir_pesos
        }
        
        # La Trinidad de Cajas
        self.caja_padre = CajaTrinidad(dim_oculta, profundidad, compartir_pesos)
        self.caja_hijo = CajaTrinidad(dim_oculta, profundidad, compartir_pesos)
        self.caja_espiritu = CajaTrinidad(dim_oculta, profundidad, compartir_pesos)
        
        # Las Llaves Calculadoras
        self.llave_padre_hijo = LlaveCalculadora(dim_oculta)
        self.llave_hijo_espiritu = LlaveCalculadora(dim_oculta)
        self.llave_skip = LlaveCalculadora(dim_oculta)
        
        # Entrada y Salida
        self.entrada = nn.Linear(dim_entrada, dim_oculta)
        self.salida = nn.Linear(dim_oculta, dim_salida)
        
        # Normalización para estabilidad
        self.norm = nn.LayerNorm(dim_oculta)
    
    def forward(self, x):
        # Entrada al universo
        x = torch.tanh(self.entrada(x))
        x = self.norm(x)
        
        # El Padre procesa
        s_padre = self.caja_padre(x)
        
        # Padre → Hijo
        hacia_hijo = self.llave_padre_hijo.ir(s_padre)
        s_hijo = self.caja_hijo(hacia_hijo)
        
        # Hijo → Espíritu
        hacia_espiritu = self.llave_hijo_espiritu.ir(s_hijo)
        
        # Skip: Padre → Espíritu directo
        skip = self.llave_skip.ir(s_padre)
        
        # El Espíritu integra
        s_espiritu = self.caja_espiritu(hacia_espiritu + skip)
        
        # Bidireccionalidad: la información vuelve
        vuelta_espiritu = self.llave_hijo_espiritu.volver(s_espiritu)
        vuelta_skip = self.llave_skip.volver(s_espiritu)
        
        # Integración final
        integracion = s_padre + s_hijo + s_espiritu + 0.2 * (vuelta_espiritu + vuelta_skip)
        integracion = self.norm(integracion)
        
        return self.salida(integracion)
    
    def contar_parametros(self):
        """Cuenta parámetros totales"""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self):
        """Retorna configuración del modelo"""
        config = self.config.copy()
        config["num_parameters"] = self.contar_parametros()
        return config
    
    def save_pretrained(self, path):
        """Guarda el modelo en formato compatible con HuggingFace"""
        os.makedirs(path, exist_ok=True)
        
        # Guardar pesos
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        
        # Guardar config
        config = self.get_config()
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Modelo guardado en {path}")
    
    @classmethod
    def from_pretrained(cls, path):
        """Carga un modelo guardado"""
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(
            dim_entrada=config["dim_entrada"],
            dim_oculta=config["dim_oculta"],
            dim_salida=config["dim_salida"],
            profundidad=config["profundidad"],
            compartir_pesos=config["compartir_pesos"]
        )
        
        model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model


# ================================================================
# CALCULAR TAMAÑOS PARA DIFERENTES ESCALAS
# ================================================================

def calcular_tamaños():
    """Calcula tamaños del modelo para diferentes configuraciones"""
    
    print("\n" + "="*70)
    print("📊 TAMAÑOS DE LLARRI-O1 PARA DIFERENTES ESCALAS")
    print("="*70)
    
    configuraciones = [
        # (nombre, dim_oculta, profundidad, compartir)
        ("Nano (actual)", 128, 3, True),
        ("Micro", 256, 3, True),
        ("Small", 512, 3, True),
        ("Medium", 1024, 4, True),
        ("Large", 2048, 4, True),
        ("XL", 4096, 4, True),
        ("XXL (sin compartir)", 4096, 4, False),
    ]
    
    print(f"\n{'Variante':<25} {'Parámetros':>15} {'~Tamaño':>12} {'Escala':>15}")
    print("-"*70)
    
    for nombre, dim, prof, compartir in configuraciones:
        modelo = LlarriO1(
            dim_entrada=784,
            dim_oculta=dim,
            dim_salida=10,
            profundidad=prof,
            compartir_pesos=compartir
        )
        params = modelo.contar_parametros()
        
        # Calcular tamaño en bytes (float32 = 4 bytes por parámetro)
        bytes_total = params * 4
        
        if bytes_total < 1024**2:
            tamaño = f"{bytes_total/1024:.1f} KB"
        elif bytes_total < 1024**3:
            tamaño = f"{bytes_total/1024**2:.1f} MB"
        else:
            tamaño = f"{bytes_total/1024**3:.2f} GB"
        
        # Escala
        if params < 1_000_000:
            escala = f"{params/1000:.0f}K"
        elif params < 1_000_000_000:
            escala = f"{params/1_000_000:.1f}M"
        else:
            escala = f"{params/1_000_000_000:.2f}B"
        
        print(f"{nombre:<25} {params:>15,} {tamaño:>12} {escala:>15}")
    
    print("-"*70)
    
    # Calcular qué necesitaríamos para 1 Billón (1B)
    print("\n📐 PARA ALCANZAR 1 BILLÓN (1B) DE PARÁMETROS:")
    print("-"*70)
    
    # Estimación: con dim_oculta=8192, profundidad=5, sin compartir
    modelo_1b = LlarriO1(784, 8192, 10, profundidad=5, compartir_pesos=False)
    params_1b = modelo_1b.contar_parametros()
    print(f"   Config: dim_oculta=8192, profundidad=5, sin compartir")
    print(f"   Parámetros: {params_1b:,}")
    print(f"   Tamaño: {params_1b * 4 / 1024**3:.2f} GB")
    
    return params_1b


if __name__ == "__main__":
    calcular_tamaños()
