#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Generador de Diagramas Visuales LLARRI-O1 v4.0 HyperComprimido
==============================================================

Genera imágenes PNG con matplotlib en 3 niveles de dificultad:
1. Básico (para niños/principiantes)
2. Medio (conceptual)
3. Avanzado (técnico completo)

Autor: Lucas Ricardo Mella Chillemi
Coordinador: Alvaro (Segunda Cabeza)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Configuración de estilo
plt.style.use('default')
COLORS = {
    'datos_A': '#3498db',      # Azul
    'datos_B': '#2ecc71',      # Verde
    'datos_C': '#9b59b6',      # Púrpura
    'calc_A': '#e74c3c',       # Rojo
    'calc_B': '#f39c12',       # Naranja
    'calc_C': '#1abc9c',       # Turquesa
    'cache': '#f1c40f',        # Amarillo
    'conexion': '#95a5a6',     # Gris
    'nivel_2': '#e74c3c',      # Rojo (binario)
    'nivel_4': '#f39c12',      # Naranja
    'nivel_8': '#f1c40f',      # Amarillo
    'nivel_16': '#2ecc71',     # Verde
    'nivel_32': '#3498db',     # Azul
    'nivel_64': '#9b59b6',     # Púrpura
    'nivel_128': '#e91e63',    # Rosa
    'nivel_256': '#00bcd4',    # Cyan
}


def crear_diagrama_basico():
    """
    Nivel BÁSICO - Para niños/principiantes
    Explica el concepto con analogías simples
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')
    
    # Título
    ax.text(7, 9.5, '🧠 LLARRI-O1 v4.0 - Explicación Simple', 
            fontsize=20, ha='center', fontweight='bold', color='#2c3e50')
    ax.text(7, 9.0, '"Como guardar 1000 libros en 1 cuaderno"', 
            fontsize=12, ha='center', style='italic', color='#7f8c8d')
    
    # === LADO IZQUIERDO: Las 6 Cajas ===
    ax.text(3.5, 8.2, '📦 Las 6 Cajas Mágicas', fontsize=14, ha='center', 
            fontweight='bold', color='#2c3e50')
    
    # 3 Cajas de datos
    for i, (label, emoji, color) in enumerate([
        ('Caja A', '📊', COLORS['datos_A']),
        ('Caja B', '📈', COLORS['datos_B']),
        ('Caja C', '📉', COLORS['datos_C']),
    ]):
        y = 7 - i * 1.2
        rect = FancyBboxPatch((1, y-0.4), 2, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(2, y, f'{emoji} {label}\n(Guarda datos)', fontsize=9, ha='center', 
                va='center', color='white', fontweight='bold')
    
    # 3 Cajas de cálculos
    for i, (label, emoji, color) in enumerate([
        ('Calc A', '🔢', COLORS['calc_A']),
        ('Calc B', '➗', COLORS['calc_B']),
        ('Calc C', '✖️', COLORS['calc_C']),
    ]):
        y = 7 - i * 1.2
        rect = FancyBboxPatch((4, y-0.4), 2, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(5, y, f'{emoji} {label}\n(Hace cuentas)', fontsize=9, ha='center', 
                va='center', color='white', fontweight='bold')
    
    # Flechas entre cajas
    ax.annotate('', xy=(4, 6.6), xytext=(3.1, 6.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['conexion'], lw=2))
    ax.annotate('', xy=(4, 5.4), xytext=(3.1, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['conexion'], lw=2))
    ax.annotate('', xy=(4, 4.2), xytext=(3.1, 4.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['conexion'], lw=2))
    
    # === LADO DERECHO: Los 8 Niveles ===
    ax.text(10.5, 8.2, '🔬 Los 8 Niveles (Zoom)', fontsize=14, ha='center', 
            fontweight='bold', color='#2c3e50')
    
    niveles = [256, 128, 64, 32, 16, 8, 4, 2]
    colores_niveles = [COLORS[f'nivel_{n}'] if f'nivel_{n}' in COLORS else '#95a5a6' for n in niveles]
    
    for i, (nivel, color) in enumerate(zip(niveles, colores_niveles)):
        y = 7.2 - i * 0.7
        width = 0.5 + (8-i) * 0.4
        x = 10.5 - width/2
        rect = FancyBboxPatch((x, y-0.25), width, 0.5, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        emoji = '🔴' if nivel == 2 else ('🟠' if nivel <= 8 else '🔵')
        ax.text(10.5, y, f'{emoji} {nivel}', fontsize=10, ha='center', 
                va='center', color='white', fontweight='bold')
    
    # Flecha de flujo
    ax.annotate('', xy=(10.5, 1.8), xytext=(10.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    ax.text(12.5, 4.5, 'Comprime\nhasta el\nmínimo', fontsize=10, ha='center', 
            va='center', color='#7f8c8d')
    
    # === ABAJO: Cache Binario ===
    cache_rect = FancyBboxPatch((3, 1), 8, 1.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['cache'], edgecolor='#f39c12', 
                                linewidth=3, alpha=0.3)
    ax.add_patch(cache_rect)
    ax.text(7, 2.1, '💾 Cache Binario (Memoria Rápida)', fontsize=12, ha='center', 
            fontweight='bold', color='#2c3e50')
    ax.text(7, 1.5, 'Guarda las respuestas de 0+0, 0+1, 1+0, 1+1\n¡No tiene que calcularlas de nuevo!', 
            fontsize=9, ha='center', va='center', color='#7f8c8d')
    
    # Leyenda
    ax.text(7, 0.5, '© 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
            fontsize=8, ha='center', color='#bdc3c7')
    
    plt.tight_layout()
    return fig


def crear_diagrama_medio():
    """
    Nivel MEDIO - Conceptual
    Muestra la arquitectura con más detalle técnico
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#fafafa')
    
    # === Panel Izquierdo: Arquitectura de 6 Cajas ===
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_facecolor('#fafafa')
    
    ax1.text(5, 9.5, '🏗️ Arquitectura de 6 Cajas', fontsize=16, ha='center', 
             fontweight='bold', color='#2c3e50')
    
    # Capa de Datos
    ax1.text(2.5, 8.5, 'CAPA DE DATOS', fontsize=11, ha='center', 
             fontweight='bold', color='#2c3e50', 
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    
    cajas_datos = [
        ('Caja A', COLORS['datos_A'], 1, 7),
        ('Caja B', COLORS['datos_B'], 2.5, 5.5),
        ('Caja C', COLORS['datos_C'], 1, 4),
    ]
    
    for name, color, x, y in cajas_datos:
        rect = FancyBboxPatch((x, y), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(x+1.5, y+0.6, f'{name}\n4 Cuadrantes', fontsize=9, ha='center', 
                va='center', color='white', fontweight='bold')
    
    # Capa de Cálculos
    ax1.text(7.5, 8.5, 'CAPA DE CÁLCULOS', fontsize=11, ha='center', 
             fontweight='bold', color='#2c3e50',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    
    cajas_calc = [
        ('Calc A', COLORS['calc_A'], 6, 7),
        ('Calc B', COLORS['calc_B'], 7.5, 5.5),
        ('Calc C', COLORS['calc_C'], 6, 4),
    ]
    
    for name, color, x, y in cajas_calc:
        rect = FancyBboxPatch((x, y), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(x+1.5, y+0.6, f'{name}\nOpera sobre\ndatos + otros calc', 
                fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    
    # Conexiones (Llaves)
    conexiones = [
        (4, 7.6, 6, 7.6),   # A->CalcA
        (4, 6.1, 7.5, 6.1), # B->CalcB
        (4, 4.6, 6, 4.6),   # C->CalcC
        (5.5, 6.1, 6, 7.2), # B->CalcA
        (5.5, 6.1, 6, 4.8), # B->CalcC
    ]
    
    for x1, y1, x2, y2 in conexiones:
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['conexion'], lw=1.5, 
                                   connectionstyle='arc3,rad=0.1'))
    
    # Leyenda de llaves
    ax1.text(5, 3, '🔑 Llaves de Conexión', fontsize=10, ha='center', 
             fontweight='bold', color='#7f8c8d')
    ax1.text(5, 2.5, 'Conectan cajas bidireccionales\nDatos ↔ Cálculos ↔ Cálculos', 
             fontsize=8, ha='center', color='#95a5a6')
    
    # Parámetros
    params_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#2c3e50', edgecolor='#34495e', linewidth=2)
    ax1.add_patch(params_box)
    ax1.text(5, 1.4, '📊 Parámetros: ~30M reales | Compresión: 26.3%', 
             fontsize=10, ha='center', color='white', fontweight='bold')
    ax1.text(5, 0.9, 'hidden_dim=1024 | quad_dim=256 | 6 cajas × 4 cuadrantes', 
             fontsize=8, ha='center', color='#bdc3c7')
    
    # === Panel Derecho: Flujo Fractal ===
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_facecolor('#fafafa')
    
    ax2.text(5, 9.5, '🔬 8 Niveles Fractales', fontsize=16, ha='center', 
             fontweight='bold', color='#2c3e50')
    ax2.text(5, 9.0, 'Flujo Progresivo Secuencial', fontsize=11, ha='center', 
             color='#7f8c8d')
    
    niveles = [
        (256, COLORS['nivel_256'], 'Entrada (quad_dim)'),
        (128, COLORS['nivel_128'], 'Compresión 1'),
        (64, COLORS['nivel_64'], 'Compresión 2'),
        (32, COLORS['nivel_32'], 'Compresión 3'),
        (16, COLORS['nivel_16'], 'Compresión 4'),
        (8, COLORS['nivel_8'], 'Compresión 5'),
        (4, COLORS['nivel_4'], 'Compresión 6'),
        (2, COLORS['nivel_2'], 'BINARIO + Cache'),
    ]
    
    for i, (nivel, color, desc) in enumerate(niveles):
        y = 8.2 - i * 0.9
        width = 1 + (8-i) * 0.8
        x = 5 - width/2
        
        rect = FancyBboxPatch((x, y-0.3), width, 0.6, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax2.add_patch(rect)
        ax2.text(5, y, f'{nivel}', fontsize=12, ha='center', va='center', 
                color='white', fontweight='bold')
        ax2.text(x + width + 0.2, y, desc, fontsize=8, ha='left', va='center', 
                color='#7f8c8d')
    
    # Flechas de flujo
    ax2.annotate('COMPRESIÓN', xy=(2.5, 4.5), fontsize=9, color='#e74c3c', 
                fontweight='bold', rotation=90, ha='center')
    ax2.annotate('', xy=(2.5, 1.5), xytext=(2.5, 8),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    
    ax2.annotate('EXPANSIÓN', xy=(7.5, 4.5), fontsize=9, color='#27ae60', 
                fontweight='bold', rotation=90, ha='center')
    ax2.annotate('', xy=(7.5, 8), xytext=(7.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    
    # Cache binario
    cache_rect = FancyBboxPatch((1.5, 0.3), 7, 1, boxstyle="round,pad=0.1",
                                facecolor=COLORS['cache'], edgecolor='#f39c12', 
                                linewidth=2, alpha=0.5)
    ax2.add_patch(cache_rect)
    ax2.text(5, 0.8, '💾 Cache Binario: Lookup instantáneo para dim=2', 
             fontsize=9, ha='center', fontweight='bold', color='#2c3e50')
    
    plt.tight_layout()
    fig.text(0.5, 0.02, '© 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
             fontsize=8, ha='center', color='#bdc3c7')
    
    return fig


def crear_diagrama_avanzado():
    """
    Nivel AVANZADO - Técnico completo
    Muestra todos los detalles de implementación
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Grid de 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # === Panel 1: Arquitectura Completa ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')
    
    ax1.text(5, 9.5, 'ARQUITECTURA v4.0', fontsize=14, ha='center', 
             fontweight='bold', color='#00ff88')
    
    # Input
    input_rect = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.05",
                                facecolor='#16213e', edgecolor='#00ff88', linewidth=2)
    ax1.add_patch(input_rect)
    ax1.text(1.5, 8.4, 'INPUT\n784→1024', fontsize=8, ha='center', 
            va='center', color='#00ff88', fontweight='bold')
    
    # 6 Cajas
    cajas = [
        ('D_A', COLORS['datos_A'], 0.5, 6),
        ('D_B', COLORS['datos_B'], 2, 6),
        ('D_C', COLORS['datos_C'], 3.5, 6),
        ('C_A', COLORS['calc_A'], 5.5, 6),
        ('C_B', COLORS['calc_B'], 7, 6),
        ('C_C', COLORS['calc_C'], 8.5, 6),
    ]
    
    for name, color, x, y in cajas:
        rect = FancyBboxPatch((x, y), 1.3, 1.5, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(x+0.65, y+0.75, name, fontsize=9, ha='center', 
                va='center', color='white', fontweight='bold')
    
    # Llaves
    ax1.text(5, 5.2, '🔑 LlaveConexion × 9', fontsize=8, ha='center', 
             color='#ffd700')
    
    # Cuadrantes dentro de cada caja
    ax1.text(2.5, 4, '4 CuadranteProgresivo\npor caja', fontsize=8, ha='center', 
             color='#888')
    
    # Output
    output_rect = FancyBboxPatch((3.5, 1.5), 3, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='#16213e', edgecolor='#ff6b6b', linewidth=2)
    ax1.add_patch(output_rect)
    ax1.text(5, 1.9, 'OUTPUT: 10 clases', fontsize=9, ha='center', 
            va='center', color='#ff6b6b', fontweight='bold')
    
    # === Panel 2: Flujo de Datos ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')
    
    ax2.text(5, 9.5, 'FLUJO PROGRESIVO', fontsize=14, ha='center', 
             fontweight='bold', color='#00ff88')
    
    # Mostrar el flujo 2→4→8→...→256
    niveles = [2, 4, 8, 16, 32, 64, 128, 256]
    colores = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#e91e63', '#00bcd4']
    
    for i, (nivel, color) in enumerate(zip(niveles, colores)):
        x = 1 + i * 1
        y = 5
        circle = Circle((x, y), 0.4, facecolor=color, edgecolor='white', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, y, str(nivel), fontsize=8, ha='center', va='center', 
                color='white', fontweight='bold')
        
        if i < len(niveles) - 1:
            ax2.annotate('', xy=(x+0.6, y), xytext=(x+0.4, y),
                        arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
    
    ax2.text(5, 3.5, 'Secuencial: 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256', 
             fontsize=10, ha='center', color='#888')
    ax2.text(5, 2.8, 'NO paralelo - evita OOM', fontsize=9, ha='center', 
             color='#ff6b6b')
    
    # Cache lookup
    cache_rect = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                                facecolor='#f1c40f', edgecolor='#f39c12', 
                                linewidth=2, alpha=0.3)
    ax2.add_patch(cache_rect)
    ax2.text(5, 1.6, 'CacheBinario.lookup(x) → O(1)', fontsize=10, ha='center', 
             color='#f1c40f', fontweight='bold')
    
    # === Panel 3: Parámetros ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_facecolor('#1a1a2e')
    
    ax3.text(5, 9.5, 'CONFIGURACIÓN', fontsize=14, ha='center', 
             fontweight='bold', color='#00ff88')
    
    config_text = """
ConfigHyperComprimido:
├── input_dim: 784
├── hidden_dim: 1024
├── output_dim: 10
├── num_cajas_datos: 3
├── num_cajas_calculos: 3
├── niveles_fractales: [2,4,8,16,32,64,128,256]
└── dropout: 0.1

Parámetros Reales: ~30M
Parámetros Sin Compartir: ~41M
Compresión: 26.3%
Tamaño en memoria: ~120 MB
"""
    ax3.text(1, 7.5, config_text, fontsize=9, ha='left', va='top', 
             color='#00ff88', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#00ff88'))
    
    # === Panel 4: Módulos ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_facecolor('#1a1a2e')
    
    ax4.text(5, 9.5, 'MÓDULOS', fontsize=14, ha='center', 
             fontweight='bold', color='#00ff88')
    
    modules = [
        ('CacheBinario', 'Lookup O(1) para dim=2', '#f1c40f'),
        ('ProcesoNivel', 'Linear→LayerNorm→GELU→Dropout→Linear', '#3498db'),
        ('CuadranteProgresivo', 'Comprime→Cache→Sube secuencial→Expande', '#2ecc71'),
        ('RelacionesCuadrantes', 'Conexiones cruzadas A↔B↔C↔D', '#9b59b6'),
        ('CajaDatos', '4 cuadrantes + relaciones', '#e74c3c'),
        ('CajaCalculos', 'Opera sobre datos + otros cálculos', '#f39c12'),
        ('LlaveConexion', 'Conecta cajas (residual × 0.5)', '#1abc9c'),
    ]
    
    for i, (name, desc, color) in enumerate(modules):
        y = 8 - i * 1.1
        rect = FancyBboxPatch((0.5, y-0.4), 9, 0.8, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.7)
        ax4.add_patch(rect)
        ax4.text(1, y, name, fontsize=9, ha='left', va='center', 
                color='white', fontweight='bold')
        ax4.text(4, y, desc, fontsize=8, ha='left', va='center', color='white')
    
    # Footer
    fig.text(0.5, 0.02, '© 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza | LLARRI-O1 v4.0 HyperComprimido', 
             fontsize=10, ha='center', color='#888')
    
    return fig


def crear_diagrama_flujo_completo():
    """
    Diagrama de flujo completo del forward pass
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')
    
    ax.text(8, 19.5, '🔄 FLUJO COMPLETO - Forward Pass', fontsize=18, ha='center', 
            fontweight='bold', color='#2c3e50')
    
    # Paso 1: Input
    rect = FancyBboxPatch((6, 18), 4, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#3498db', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 18.4, '1. INPUT x: (batch, 784)', fontsize=10, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 2: Proyección
    ax.annotate('', xy=(8, 17), xytext=(8, 17.9),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((5.5, 16.2), 5, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#2ecc71', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 16.6, '2. proj_in: 784 → 1024', fontsize=10, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 3: Dividir en cuadrantes
    ax.annotate('', xy=(8, 15.2), xytext=(8, 16.1),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((4, 14.4), 8, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#9b59b6', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 14.8, '3. Dividir: a,b,c,d = x[:256], x[256:512], ...', fontsize=9, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 4: Cuadrante Progresivo
    ax.annotate('', xy=(8, 13.4), xytext=(8, 14.3),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Caja de cuadrante progresivo
    rect = FancyBboxPatch((2, 8), 12, 5.3, boxstyle="round,pad=0.1",
                          facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 13, '4. CuadranteProgresivo (×4 por caja, ×6 cajas)', fontsize=11, ha='center', 
            fontweight='bold', color='#2c3e50')
    
    # Sub-pasos del cuadrante
    sub_steps = [
        ('4.1 Comprimir: 256 → 2', '#e74c3c'),
        ('4.2 Cache lookup: (batch, 2) → (batch, 7)', '#f1c40f'),
        ('4.3 Fusión: cat([h, cache]) → Linear → (batch, 2)', '#f39c12'),
        ('4.4 Proceso nivel 2', '#e74c3c'),
        ('4.5 Subir: 2→4→8→16→32→64→128→256', '#27ae60'),
        ('4.6 Expandir: 256 → 256 (residual)', '#3498db'),
    ]
    
    for i, (step, color) in enumerate(sub_steps):
        y = 12 - i * 0.7
        rect = FancyBboxPatch((3, y-0.25), 10, 0.5, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.8)
        ax.add_patch(rect)
        ax.text(8, y, step, fontsize=9, ha='center', va='center', 
                color='white', fontweight='bold')
    
    # Paso 5: Relaciones
    ax.annotate('', xy=(8, 7), xytext=(8, 7.9),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((4, 6.2), 8, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#1abc9c', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 6.6, '5. RelacionesCuadrantes: a↔b, c↔d, a↔c, b↔d', fontsize=9, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 6: Llaves
    ax.annotate('', xy=(8, 5.2), xytext=(8, 6.1),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((4, 4.4), 8, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#e67e22', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 4.8, '6. LlaveConexion: datos↔cálculos (×9 llaves)', fontsize=9, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 7: Fusión
    ax.annotate('', xy=(8, 3.4), xytext=(8, 4.3),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((4, 2.6), 8, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#8e44ad', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 3, '7. Fusión: cat([datos_sum, calc_sum]) → (batch, 2048)', fontsize=9, ha='center', 
            va='center', color='white', fontweight='bold')
    
    # Paso 8: Output
    ax.annotate('', xy=(8, 1.6), xytext=(8, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((5, 0.8), 6, 0.8, boxstyle="round,pad=0.05",
                          facecolor='#c0392b', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 1.2, '8. OUTPUT: 2048 → 1024 → 10', fontsize=10, ha='center', 
            va='center', color='white', fontweight='bold')
    
    ax.text(8, 0.3, '© 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
            fontsize=8, ha='center', color='#bdc3c7')
    
    return fig


def main():
    """Genera todos los diagramas y los guarda."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'diagrams', 'v4-current')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERADOR DE DIAGRAMAS LLARRI-O1 v4.0")
    print("=" * 60)
    
    diagramas = [
        ('01_basico_simple.png', crear_diagrama_basico, 'Básico (para principiantes)'),
        ('02_medio_conceptual.png', crear_diagrama_medio, 'Medio (conceptual)'),
        ('03_avanzado_tecnico.png', crear_diagrama_avanzado, 'Avanzado (técnico)'),
        ('04_flujo_completo.png', crear_diagrama_flujo_completo, 'Flujo Forward Pass'),
    ]
    
    for filename, func, desc in diagramas:
        print(f"\n📊 Generando: {desc}...")
        fig = func()
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"   ✓ Guardado: {filepath}")
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS DIAGRAMAS GENERADOS")
    print(f"   Ubicación: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
