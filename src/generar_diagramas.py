# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 v3.0 - Generador de Diagramas
========================================

Crea diagramas explicativos de la arquitectura
Trinity Fractal Recursivo Profundo.

Niveles:
- SUPER SIMPLE: Para niños (emojis y colores)
- BÁSICO: Conceptual simple
- INTERMEDIO: Arquitectura detallada
- AVANZADO: Comparación técnica

Autor: Lucas Mella (Segunda Cabeza)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


# ==============================================================================
# NIVEL 1: SUPER SIMPLE (PARA NIÑOS)
# ==============================================================================

def crear_diagrama_super_simple(guardar_en: str = "diagrams"):
    """
    Diagrama SUPER SIMPLE - Para que lo entienda un niño de 5 años
    
    Usa emojis y analogías simples
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Título
    ax.text(7, 9.5, "🧠 LLARRI-O1: ¡Un Cerebro Mágico! 🧠", 
            fontsize=24, ha='center', color='white', fontweight='bold')
    
    # Explicación simple
    ax.text(7, 8.5, "Imagina que tienes 3 cajas mágicas que piensan juntas...",
            fontsize=14, ha='center', color='#00d4ff', style='italic')
    
    # Las 3 cajas como casitas
    cajas = [
        (2, 4.5, "🏠", "Casita 1\n(Padre)", "#ff6b6b"),
        (7, 4.5, "🏠", "Casita 2\n(Hijo)", "#4ecdc4"),
        (12, 4.5, "🏠", "Casita 3\n(Espíritu)", "#ffe66d")
    ]
    
    for x, y, emoji, nombre, color in cajas:
        # Casita
        ax.text(x, y + 1.5, emoji, fontsize=60, ha='center')
        ax.text(x, y, nombre, fontsize=14, ha='center', color=color, fontweight='bold')
    
    # Flechas de conexión (como caminos)
    ax.annotate("", xy=(5.5, 5.5), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='<->', color='#00d4ff', lw=3))
    ax.text(4.5, 6, "🔑", fontsize=20, ha='center')
    
    ax.annotate("", xy=(10.5, 5.5), xytext=(8.5, 5.5),
                arrowprops=dict(arrowstyle='<->', color='#00d4ff', lw=3))
    ax.text(9.5, 6, "🔑", fontsize=20, ha='center')
    
    # Cuadrantes dentro de las cajas (como ventanitas)
    ax.text(7, 2.5, "Cada casita tiene 4 ventanitas (cuadrantes) 🪟🪟🪟🪟",
            fontsize=12, ha='center', color='white')
    
    # Recursión simple
    ax.text(7, 1.5, "¡Y dentro de cada ventanita... hay 4 ventanitas más pequeñas!",
            fontsize=12, ha='center', color='#ffd93d')
    
    ax.text(7, 0.8, "🪟→🔲🔲🔲🔲 →🔳🔳🔳🔳→...", 
            fontsize=14, ha='center', color='white')
    
    ax.text(7, 0.2, "¡Como muñecas rusas pero con ventanas!",
            fontsize=11, ha='center', color='#4ecdc4', style='italic')
    
    # Guardar
    Path(guardar_en).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/01_super_simple_ninos.png", dpi=150, 
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/01_super_simple_ninos.png")


# ==============================================================================
# NIVEL 2: BÁSICO (CONCEPTUAL)
# ==============================================================================

def crear_diagrama_basico(guardar_en: str = "diagrams"):
    """
    Diagrama BÁSICO - Estructura conceptual
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Título
    ax.text(8, 9.5, "LLARRI-O1 v3.0 - Estructura Básica", 
            fontsize=22, ha='center', color='white', fontweight='bold')
    ax.text(8, 8.9, "Trinity Fractal Recursivo Profundo", 
            fontsize=14, ha='center', color='#58a6ff')
    
    # Las 3 cajas
    colores = ['#ff6b6b', '#4ecdc4', '#ffe66d']
    nombres = ['CAJA 1\n(Padre)', 'CAJA 2\n(Hijo)', 'CAJA 3\n(Espíritu)']
    
    for i, (color, nombre) in enumerate(zip(colores, nombres)):
        x = 2 + i * 5
        
        # Caja principal
        rect = FancyBboxPatch((x, 3), 4, 4, boxstyle="round,pad=0.1",
                              facecolor='#21262d', edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        # Título de caja
        ax.text(x + 2, 7.3, nombre, fontsize=12, ha='center', 
                color=color, fontweight='bold')
        
        # Cuadrantes (2x2)
        for qi in range(2):
            for qj in range(2):
                qx = x + 0.3 + qj * 1.8
                qy = 3.3 + (1-qi) * 1.8
                
                qrect = FancyBboxPatch((qx, qy), 1.6, 1.6, 
                                       boxstyle="round,pad=0.05",
                                       facecolor='#30363d', edgecolor='#8b949e',
                                       linewidth=1)
                ax.add_patch(qrect)
                
                letra = ['A', 'B', 'C', 'D'][qi * 2 + qj]
                ax.text(qx + 0.8, qy + 0.8, letra, fontsize=14, 
                       ha='center', va='center', color='white')
    
    # Flechas entre cajas
    # Caja 1 ↔ Caja 2
    ax.annotate("", xy=(6.5, 5), xytext=(6, 5),
                arrowprops=dict(arrowstyle='<->', color='#58a6ff', lw=2))
    ax.text(6.25, 5.5, "🔑", fontsize=16, ha='center')
    
    # Caja 2 → Caja 3
    ax.annotate("", xy=(11.5, 5), xytext=(11, 5),
                arrowprops=dict(arrowstyle='<->', color='#58a6ff', lw=2))
    ax.text(11.25, 5.5, "🔑", fontsize=16, ha='center')
    
    # Leyenda de recursión
    ax.text(8, 2.3, "RECURSIÓN FRACTAL", fontsize=14, ha='center', 
            color='#ffd93d', fontweight='bold')
    ax.text(8, 1.7, "Cada cuadrante A,B,C,D contiene 4 sub-cuadrantes", 
            fontsize=11, ha='center', color='white')
    ax.text(8, 1.2, "Cada sub-cuadrante contiene 4 sub-sub-cuadrantes...", 
            fontsize=11, ha='center', color='#8b949e')
    ax.text(8, 0.7, "¡Hasta llegar al mínimo posible!", 
            fontsize=11, ha='center', color='#4ecdc4')
    
    ax.text(8, 0.2, "TODOS los niveles comparten los mismos pesos = COMPRESIÓN EXTREMA", 
            fontsize=10, ha='center', color='#ff6b6b', fontweight='bold')
    
    # Guardar
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/02_estructura_basica.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/02_estructura_basica.png")


# ==============================================================================
# NIVEL 3: RECURSIÓN FRACTAL (VISUAL)
# ==============================================================================

def crear_diagrama_recursion_fractal(guardar_en: str = "diagrams"):
    """
    Diagrama de RECURSIÓN FRACTAL - Muestra los niveles
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.patch.set_facecolor('#0d1117')
    
    niveles_nombres = ['Nivel 0\n(Cuadrantes)', 'Nivel 1\n(Sub-cuadrantes)', 
                      'Nivel 2\n(Sub-sub)', 'Nivel 3\n(Mínimo)']
    dims = ['256 dims', '64 dims', '16 dims', '4 dims']
    
    for idx, (ax, nombre, dim) in enumerate(zip(axes, niveles_nombres, dims)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor('#0d1117')
        
        # Título del nivel
        ax.text(5, 9.5, nombre, fontsize=14, ha='center', 
                color='#58a6ff', fontweight='bold')
        ax.text(5, 8.8, dim, fontsize=11, ha='center', color='#8b949e')
        
        # Dibujar cuadrícula recursiva
        divisiones = 2 ** idx
        size = 7 / divisiones
        
        colores = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a78bfa']
        
        for i in range(divisiones):
            for j in range(divisiones):
                color_idx = (i + j) % 4
                x = 1.5 + j * size
                y = 1 + i * size
                
                rect = patches.Rectangle(
                    (x, y), size * 0.95, size * 0.95,
                    facecolor=colores[color_idx], alpha=0.6,
                    edgecolor='white', linewidth=0.5
                )
                ax.add_patch(rect)
        
        # Número total
        total = (2 ** idx) ** 2
        ax.text(5, 0.3, f"{total} cuadrantes", fontsize=10, 
                ha='center', color='white')
    
    # Título general
    fig.suptitle("LLARRI-O1: Recursión Fractal - Cada nivel divide en 4", 
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/03_recursion_fractal.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/03_recursion_fractal.png")


# ==============================================================================
# NIVEL 4: COMPARACIÓN TÉCNICA
# ==============================================================================

def crear_diagrama_comparacion_tecnica(guardar_en: str = "diagrams"):
    """
    Diagrama de COMPARACIÓN TÉCNICA - LLARRI vs otros modelos
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor('#0d1117')
    
    modelos = [
        {
            'nombre': 'Transformer\n(Tradicional)',
            'color': '#ff6b6b',
            'params': '~100M',
            'compresion': '0%',
            'estructura': ['Atención', 'FFN', 'LayerNorm'],
            'icono': '🔲'
        },
        {
            'nombre': 'CNN\n(Convolucional)',
            'color': '#4ecdc4',
            'params': '~25M',
            'compresion': '~20%',
            'estructura': ['Conv', 'Pool', 'FC'],
            'icono': '📊'
        },
        {
            'nombre': 'LLARRI-O1 v3.0\n(Trinity Fractal)',
            'color': '#ffe66d',
            'params': '~1M',
            'compresion': '~98%',
            'estructura': ['Fractal', 'Compartido', 'Recursivo'],
            'icono': '🔷'
        }
    ]
    
    for ax, modelo in zip(axes, modelos):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor('#0d1117')
        
        # Caja del modelo
        rect = FancyBboxPatch((0.5, 1), 9, 8, boxstyle="round,pad=0.2",
                              facecolor='#21262d', edgecolor=modelo['color'],
                              linewidth=3)
        ax.add_patch(rect)
        
        # Nombre
        ax.text(5, 8.5, modelo['icono'], fontsize=30, ha='center')
        ax.text(5, 7.2, modelo['nombre'], fontsize=14, ha='center',
                color=modelo['color'], fontweight='bold')
        
        # Stats
        ax.text(5, 5.8, f"Parámetros: {modelo['params']}", 
                fontsize=12, ha='center', color='white')
        ax.text(5, 5.0, f"Compresión: {modelo['compresion']}", 
                fontsize=12, ha='center', color='white')
        
        # Estructura
        ax.text(5, 3.8, "Estructura:", fontsize=11, ha='center', 
                color='#8b949e', fontweight='bold')
        for i, s in enumerate(modelo['estructura']):
            ax.text(5, 3.2 - i * 0.6, f"• {s}", fontsize=10, 
                    ha='center', color='#58a6ff')
    
    fig.suptitle("Comparación: LLARRI-O1 vs Arquitecturas Tradicionales",
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/04_comparacion_tecnica.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/04_comparacion_tecnica.png")


# ==============================================================================
# NIVEL 5: COMPRESIÓN VISUAL
# ==============================================================================

def crear_diagrama_compresion(guardar_en: str = "diagrams"):
    """
    Diagrama de COMPRESIÓN - Muestra el ahorro de parámetros
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Datos de comparación
    categorias = ['Sin\nCompartir', 'LLARRI v1\n(Básico)', 'LLARRI v2\n(Cuadrantes)', 'LLARRI v3\n(Fractal)']
    valores = [100, 50, 10, 2]  # Porcentaje relativo
    colores = ['#ff6b6b', '#ffd93d', '#4ecdc4', '#58a6ff']
    
    bars = ax.barh(categorias, valores, color=colores, edgecolor='white', linewidth=2)
    
    # Añadir valores
    for bar, val in zip(bars, valores):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=14, color='white', fontweight='bold')
    
    # Estilo
    ax.set_xlabel('Uso de Memoria Relativo', fontsize=14, color='white')
    ax.set_title('LLARRI-O1: Evolución de la Compresión de Parámetros',
                fontsize=18, color='white', fontweight='bold', pad=20)
    
    ax.tick_params(colors='white', labelsize=12)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Nota
    ax.text(50, -0.8, "¡98% de reducción en parámetros gracias a la recursión fractal!",
            ha='center', fontsize=12, color='#4ecdc4', style='italic')
    
    # Guardar
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/05_compresion_parametros.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/05_compresion_parametros.png")


# ==============================================================================
# NIVEL 6: ARQUITECTURA COMPLETA (DETALLADA)
# ==============================================================================

def crear_diagrama_arquitectura_completa(guardar_en: str = "diagrams"):
    """
    Diagrama COMPLETO de la arquitectura
    """
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Título
    ax.text(10, 13.5, "LLARRI-O1 v3.0 - Arquitectura Completa", 
            fontsize=24, ha='center', color='white', fontweight='bold')
    ax.text(10, 12.8, "Trinity Fractal Recursivo Profundo | Por Lucas Mella (Segunda Cabeza)",
            fontsize=12, ha='center', color='#58a6ff')
    
    # INPUT
    input_box = FancyBboxPatch((0.5, 5.5), 2, 2, boxstyle="round,pad=0.1",
                               facecolor='#1f6feb', edgecolor='white', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 6.5, "INPUT\n784 dims", fontsize=11, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Las 3 Cajas Trinity
    cajas_x = [4, 9, 14]
    cajas_nombres = ['CAJA 1\n(Padre)', 'CAJA 2\n(Hijo)', 'CAJA 3\n(Espíritu)']
    cajas_colores = ['#ff6b6b', '#4ecdc4', '#ffe66d']
    
    for x, nombre, color in zip(cajas_x, cajas_nombres, cajas_colores):
        # Caja grande
        caja = FancyBboxPatch((x, 3), 4, 6, boxstyle="round,pad=0.15",
                              facecolor='#21262d', edgecolor=color, linewidth=3)
        ax.add_patch(caja)
        
        ax.text(x + 2, 9.3, nombre, fontsize=12, ha='center',
                color=color, fontweight='bold')
        
        # 4 Cuadrantes
        letras = ['A', 'B', 'C', 'D']
        for qi in range(2):
            for qj in range(2):
                qx = x + 0.3 + qj * 1.85
                qy = 3.3 + (1-qi) * 2.8
                
                # Cuadrante
                quad = FancyBboxPatch((qx, qy), 1.7, 2.6, 
                                      boxstyle="round,pad=0.05",
                                      facecolor='#30363d', edgecolor='#58a6ff',
                                      linewidth=1)
                ax.add_patch(quad)
                
                letra = letras[qi * 2 + qj]
                ax.text(qx + 0.85, qy + 2.3, letra, fontsize=11,
                       ha='center', color='white', fontweight='bold')
                
                # Sub-cuadrantes (4 dentro)
                for si in range(2):
                    for sj in range(2):
                        sx = qx + 0.1 + sj * 0.75
                        sy = qy + 0.1 + (1-si) * 1.0
                        
                        sub = patches.Rectangle((sx, sy), 0.65, 0.9,
                                               facecolor='#484f58', 
                                               edgecolor='#8b949e',
                                               linewidth=0.5)
                        ax.add_patch(sub)
    
    # Flechas entre cajas
    # Caja 1 ↔ Caja 2
    ax.annotate("", xy=(8.5, 6), xytext=(8, 6),
                arrowprops=dict(arrowstyle='<->', color='#58a6ff', lw=2.5))
    ax.text(8.25, 6.5, "🔑", fontsize=18, ha='center')
    
    # Caja 2 ↔ Caja 3
    ax.annotate("", xy=(13.5, 6), xytext=(13, 6),
                arrowprops=dict(arrowstyle='<->', color='#58a6ff', lw=2.5))
    ax.text(13.25, 6.5, "🔑", fontsize=18, ha='center')
    
    # Input → Caja 1
    ax.annotate("", xy=(3.8, 6.5), xytext=(2.7, 6.5),
                arrowprops=dict(arrowstyle='->', color='#1f6feb', lw=2))
    
    # OUTPUT
    output_box = FancyBboxPatch((18, 5.5), 1.5, 2, boxstyle="round,pad=0.1",
                                facecolor='#238636', edgecolor='white', linewidth=2)
    ax.add_patch(output_box)
    ax.text(18.75, 6.5, "OUT\n10", fontsize=11, ha='center',
            va='center', color='white', fontweight='bold')
    
    ax.annotate("", xy=(17.8, 6.5), xytext=(18, 6.5),
                arrowprops=dict(arrowstyle='<-', color='#238636', lw=2))
    
    # Retroalimentación Caja3 → Caja1
    ax.annotate("", xy=(6, 2.7), xytext=(16, 2.7),
                arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=2,
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(11, 1.8, "Retroalimentación 🔄", fontsize=10, ha='center', color='#a78bfa')
    
    # Leyenda
    leyenda_y = 0.8
    ax.text(1, leyenda_y, "LEYENDA:", fontsize=11, color='white', fontweight='bold')
    ax.text(3.5, leyenda_y, "🔑 = Llave (conexión)", fontsize=10, color='#58a6ff')
    ax.text(7.5, leyenda_y, "A,B,C,D = Cuadrantes", fontsize=10, color='white')
    ax.text(11.5, leyenda_y, "Cuadrados pequeños = Sub-cuadrantes (fractales)", 
            fontsize=10, color='#8b949e')
    
    # Guardar
    plt.tight_layout()
    plt.savefig(f"{guardar_en}/06_arquitectura_completa.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {guardar_en}/06_arquitectura_completa.png")


# ==============================================================================
# GENERAR TODOS
# ==============================================================================

def generar_todos_los_diagramas(guardar_en: str = "diagrams"):
    """Genera todos los diagramas"""
    Path(guardar_en).mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("  GENERANDO DIAGRAMAS LLARRI-O1 v3.0")
    print("="*60 + "\n")
    
    crear_diagrama_super_simple(guardar_en)
    crear_diagrama_basico(guardar_en)
    crear_diagrama_recursion_fractal(guardar_en)
    crear_diagrama_comparacion_tecnica(guardar_en)
    crear_diagrama_compresion(guardar_en)
    crear_diagrama_arquitectura_completa(guardar_en)
    
    print("\n" + "="*60)
    print("  ✓ TODOS LOS DIAGRAMAS GENERADOS")
    print("="*60 + "\n")


if __name__ == "__main__":
    generar_todos_los_diagramas()
