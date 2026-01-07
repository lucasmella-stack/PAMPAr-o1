"""
LLARRI-O1 - Generador de Diagramas
==================================

Genera diagramas visuales de la arquitectura Trinity Fractal Cuadrantes

Autor: Lucas Mella (Segunda Cabeza)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np


def crear_diagrama_arquitectura(guardar: bool = True, ruta: str = "diagrams/arquitectura_v2.png"):
    """
    Crea un diagrama completo de la arquitectura LLARRI-O1 v2.0
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colores
    colores = {
        'caja1': '#3498db',
        'caja2': '#e74c3c',
        'caja3': '#2ecc71',
        'cuadrante': '#f39c12',
        'subcuadrante': '#9b59b6',
        'llave_ida': '#1abc9c',
        'llave_vuelta': '#e67e22'
    }
    
    # Título
    ax.text(8, 11.5, 'LLARRI-O1 v2.0 - Trinity Fractal Cuadrantes', 
            fontsize=18, ha='center', fontweight='bold')
    ax.text(8, 11, 'Arquitectura con Pesos Compartidos', 
            fontsize=12, ha='center', style='italic', color='gray')
    
    def dibujar_subcuadrante(x, y, size, label, color):
        """Dibuja un sub-cuadrante"""
        rect = FancyBboxPatch((x, y), size, size, 
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + size/2, y + size/2, label, fontsize=7, ha='center', va='center')
    
    def dibujar_cuadrante(x, y, size, label, color_fondo):
        """Dibuja un cuadrante con 4 sub-cuadrantes"""
        # Fondo del cuadrante
        rect = FancyBboxPatch((x, y), size, size,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=color_fondo, edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
        
        # Sub-cuadrantes
        sub_size = size * 0.4
        margin = size * 0.1
        
        dibujar_subcuadrante(x + margin, y + size - margin - sub_size, sub_size, 'a1', colores['subcuadrante'])
        dibujar_subcuadrante(x + size - margin - sub_size, y + size - margin - sub_size, sub_size, 'a2', colores['subcuadrante'])
        dibujar_subcuadrante(x + margin, y + margin, sub_size, 'a3', colores['subcuadrante'])
        dibujar_subcuadrante(x + size - margin - sub_size, y + margin, sub_size, 'a4', colores['subcuadrante'])
        
        # Label del cuadrante
        ax.text(x + size/2, y + size + 0.15, label, fontsize=9, ha='center', fontweight='bold')
    
    def dibujar_caja(x, y, size, nombre, color):
        """Dibuja una caja con 4 cuadrantes"""
        # Fondo de la caja
        rect = FancyBboxPatch((x, y), size, size,
                              boxstyle="round,pad=0.03,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.2)
        ax.add_patch(rect)
        
        # Cuadrantes
        cuad_size = size * 0.42
        margin = size * 0.05
        
        dibujar_cuadrante(x + margin, y + size - margin - cuad_size, cuad_size, 'A', colores['cuadrante'])
        dibujar_cuadrante(x + size - margin - cuad_size, y + size - margin - cuad_size, cuad_size, 'B', colores['cuadrante'])
        dibujar_cuadrante(x + margin, y + margin, cuad_size, 'C', colores['cuadrante'])
        dibujar_cuadrante(x + size - margin - cuad_size, y + margin, cuad_size, 'D', colores['cuadrante'])
        
        # Nombre de la caja
        ax.text(x + size/2, y + size + 0.4, nombre, fontsize=12, ha='center', fontweight='bold')
    
    # Dibujar las 3 cajas
    caja_size = 3.5
    
    # Caja 1 (izquierda)
    dibujar_caja(1.5, 4, caja_size, 'CAJA 1', colores['caja1'])
    
    # Caja 2 (centro)
    dibujar_caja(6.25, 4, caja_size, 'CAJA 2', colores['caja2'])
    
    # Caja 3 (derecha)
    dibujar_caja(11, 4, caja_size, 'CAJA 3', colores['caja3'])
    
    # Flechas (llaves) entre cajas
    estilo_flecha = dict(arrowstyle='->', mutation_scale=15, lw=2)
    
    # Caja1 ↔ Caja2 (bidireccional)
    ax.annotate('', xy=(6.1, 6.5), xytext=(5.1, 6.5),
                arrowprops=dict(arrowstyle='->', color=colores['llave_ida'], lw=2))
    ax.annotate('', xy=(5.1, 5.5), xytext=(6.1, 5.5),
                arrowprops=dict(arrowstyle='->', color=colores['llave_vuelta'], lw=2))
    ax.text(5.6, 7, 'Bidireccional', fontsize=8, ha='center', color='gray')
    
    # Caja1 → Caja3 (solo ida)
    ax.annotate('', xy=(10.9, 7.5), xytext=(5.1, 7.5),
                arrowprops=dict(arrowstyle='->', color=colores['llave_ida'], lw=2,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(8, 9, 'Solo Ida', fontsize=8, ha='center', color='gray')
    
    # Caja2 → Caja3 (solo ida)
    ax.annotate('', xy=(10.9, 6.5), xytext=(9.9, 6.5),
                arrowprops=dict(arrowstyle='->', color=colores['llave_ida'], lw=2))
    ax.text(10.4, 7, 'Ida', fontsize=8, ha='center', color='gray')
    
    # Caja3 → Caja1 (retroalimentación)
    ax.annotate('', xy=(1.5, 4.5), xytext=(11, 4.5),
                arrowprops=dict(arrowstyle='->', color=colores['llave_vuelta'], lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(6.25, 2.2, 'Retroalimentación', fontsize=8, ha='center', color='gray')
    
    # Leyenda
    leyenda_y = 1
    ax.text(1, leyenda_y, 'LEYENDA:', fontsize=10, fontweight='bold')
    
    elementos = [
        ('Caja Trinity', colores['caja1']),
        ('Cuadrante', colores['cuadrante']),
        ('Sub-cuadrante', colores['subcuadrante']),
        ('Llave Ida →', colores['llave_ida']),
        ('Llave Vuelta ←', colores['llave_vuelta'])
    ]
    
    for i, (texto, color) in enumerate(elementos):
        x_pos = 3 + i * 2.5
        rect = patches.Rectangle((x_pos, leyenda_y - 0.15), 0.3, 0.3, 
                                  facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos + 0.4, leyenda_y, texto, fontsize=8, va='center')
    
    # Nota de compresión
    ax.text(8, 0.3, '⚡ Pesos compartidos: ~99% compresión vs arquitecturas tradicionales', 
            fontsize=10, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    if guardar:
        import os
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Diagrama guardado en {ruta}")
    
    return fig


def crear_diagrama_comparativo(guardar: bool = True, ruta: str = "diagrams/comparativa_v2.png"):
    """
    Crea un diagrama comparativo: Transformer vs LLARRI-O1 v2.0
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # === TRANSFORMER (izquierda) ===
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Transformer Tradicional', fontsize=14, fontweight='bold', pad=20)
    
    # Capas del transformer
    colores_trans = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    capas = [
        ('Input Embedding', 1),
        ('Multi-Head Attention', 2.5),
        ('Feed Forward', 4),
        ('Layer Norm', 5.5),
        ('Output', 7)
    ]
    
    for i, (nombre, y) in enumerate(capas):
        rect = FancyBboxPatch((2, y), 6, 1,
                              boxstyle="round,pad=0.02",
                              facecolor=colores_trans[i], alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(5, y + 0.5, nombre, fontsize=10, ha='center', va='center', color='white', fontweight='bold')
        
        if i < len(capas) - 1:
            ax1.annotate('', xy=(5, capas[i+1][1]), xytext=(5, y + 1),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Estadísticas Transformer
    ax1.text(5, 9, '❌ Sin compartir pesos', fontsize=10, ha='center')
    ax1.text(5, 8.5, '❌ O(n²) complejidad attention', fontsize=10, ha='center')
    ax1.text(5, 8, '❌ ~125M+ parámetros típicos', fontsize=10, ha='center')
    
    # === LLARRI-O1 v2.0 (derecha) ===
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('LLARRI-O1 v2.0 Trinity Fractal', fontsize=14, fontweight='bold', pad=20)
    
    # Las 3 cajas simplificadas
    colores_llarri = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (color, x_offset) in enumerate(zip(colores_llarri, [1, 3.5, 6])):
        # Caja
        rect = FancyBboxPatch((x_offset, 3), 2.5, 3,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.3, edgecolor='black', lw=2)
        ax2.add_patch(rect)
        
        # Mini-cuadrantes
        for j, (dx, dy) in enumerate([(0.1, 1.6), (1.3, 1.6), (0.1, 0.2), (1.3, 0.2)]):
            mini = FancyBboxPatch((x_offset + dx, 3 + dy), 1, 1.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, alpha=0.6, edgecolor='black')
            ax2.add_patch(mini)
            ax2.text(x_offset + dx + 0.5, 3 + dy + 0.6, ['A','B','C','D'][j], 
                    fontsize=8, ha='center', va='center')
        
        ax2.text(x_offset + 1.25, 6.3, f'Caja {i+1}', fontsize=10, ha='center', fontweight='bold')
    
    # Flechas entre cajas
    ax2.annotate('', xy=(3.4, 4.5), xytext=(3.6, 4.5),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#1abc9c'))
    ax2.annotate('', xy=(5.9, 4.5), xytext=(6.1, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#1abc9c'))
    
    # Retroalimentación
    ax2.annotate('', xy=(1.5, 3), xytext=(8, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#e67e22',
                               connectionstyle='arc3,rad=-0.4'))
    ax2.text(4.75, 1.8, 'Retroalimentación', fontsize=8, ha='center', color='#e67e22')
    
    # Estadísticas LLARRI
    ax2.text(5, 9, '✅ Pesos compartidos (99% compresión)', fontsize=10, ha='center')
    ax2.text(5, 8.5, '✅ O(n) complejidad fractal', fontsize=10, ha='center')
    ax2.text(5, 8, '✅ ~500K parámetros efectivos', fontsize=10, ha='center')
    
    # Nota del "genio"
    ax2.text(5, 0.7, '🧠 "Un modelo de 14GB terminará pesando nada"', 
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    if guardar:
        import os
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Diagrama guardado en {ruta}")
    
    return fig


def crear_diagrama_estructura_fractal(guardar: bool = True, ruta: str = "diagrams/estructura_fractal.png"):
    """
    Diagrama que muestra la estructura fractal recursiva
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Estructura Fractal Recursiva LLARRI-O1', 
            fontsize=16, ha='center', fontweight='bold')
    
    # Nivel 2: Caja completa
    ax.text(1, 8.5, 'Nivel 2: CAJA', fontsize=12, fontweight='bold', color='#3498db')
    caja = FancyBboxPatch((1, 5), 3, 3, boxstyle="round,pad=0.02",
                          facecolor='#3498db', alpha=0.3, edgecolor='black', lw=2)
    ax.add_patch(caja)
    
    # Cuadrantes dentro de la caja
    for i, (dx, dy, label) in enumerate([(0.1, 1.6, 'A'), (1.6, 1.6, 'B'), (0.1, 0.1, 'C'), (1.6, 0.1, 'D')]):
        q = FancyBboxPatch((1 + dx, 5 + dy), 1.3, 1.3, boxstyle="round,pad=0.02",
                           facecolor='#f39c12', alpha=0.5, edgecolor='black')
        ax.add_patch(q)
        ax.text(1 + dx + 0.65, 5 + dy + 0.65, label, fontsize=10, ha='center', va='center')
    
    # Flecha nivel 2 -> nivel 1
    ax.annotate('', xy=(5.5, 6.5), xytext=(4.2, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(4.85, 7, 'Expandir', fontsize=9, ha='center', color='gray')
    
    # Nivel 1: Cuadrante
    ax.text(6, 8.5, 'Nivel 1: CUADRANTE', fontsize=12, fontweight='bold', color='#f39c12')
    cuad = FancyBboxPatch((6, 5), 3, 3, boxstyle="round,pad=0.02",
                          facecolor='#f39c12', alpha=0.3, edgecolor='black', lw=2)
    ax.add_patch(cuad)
    
    # Sub-cuadrantes dentro
    for i, (dx, dy, label) in enumerate([(0.1, 1.6, 'a1'), (1.6, 1.6, 'a2'), (0.1, 0.1, 'a3'), (1.6, 0.1, 'a4')]):
        sq = FancyBboxPatch((6 + dx, 5 + dy), 1.3, 1.3, boxstyle="round,pad=0.02",
                            facecolor='#9b59b6', alpha=0.5, edgecolor='black')
        ax.add_patch(sq)
        ax.text(6 + dx + 0.65, 5 + dy + 0.65, label, fontsize=10, ha='center', va='center')
    
    # Flecha nivel 1 -> nivel 0
    ax.annotate('', xy=(10.5, 6.5), xytext=(9.2, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(9.85, 7, 'Expandir', fontsize=9, ha='center', color='gray')
    
    # Nivel 0: Sub-cuadrante
    ax.text(11, 8.5, 'Nivel 0: SUB-CUADRANTE', fontsize=12, fontweight='bold', color='#9b59b6')
    sub = FancyBboxPatch((11, 5), 2, 2, boxstyle="round,pad=0.02",
                         facecolor='#9b59b6', alpha=0.3, edgecolor='black', lw=2)
    ax.add_patch(sub)
    ax.text(12, 6, 'Cálculo\nInterno', fontsize=10, ha='center', va='center')
    
    # Explicación de compresión
    ax.text(7, 3.5, '🗜️ COMPRESIÓN POR COMPARTIR PESOS', fontsize=12, ha='center', fontweight='bold')
    
    explicacion = [
        'Sin compartir: 3 cajas × 4 cuad × 4 sub = 48 sets de pesos',
        'Con compartir: 1 sub-cuadrante + relaciones = 1 set reutilizado',
        '➡️ Factor de compresión: ~48x en arquitectura base',
        '➡️ Con modelos grandes: compresión > 99%'
    ]
    
    for i, texto in enumerate(explicacion):
        ax.text(7, 2.8 - i*0.5, texto, fontsize=10, ha='center')
    
    # Nota importante
    ax.text(7, 0.5, '💡 "La información se relaciona por POSICIÓN dentro de cada nivel,\n'
            'y por LLAVES entre niveles diferentes"',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    if guardar:
        import os
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Diagrama guardado en {ruta}")
    
    return fig


if __name__ == "__main__":
    print("Generando diagramas LLARRI-O1 v2.0...")
    
    crear_diagrama_arquitectura()
    crear_diagrama_comparativo()
    crear_diagrama_estructura_fractal()
    
    print("\n✓ Todos los diagramas generados!")
