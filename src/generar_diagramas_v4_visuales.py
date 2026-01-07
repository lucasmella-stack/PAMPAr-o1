#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Generador de Diagramas LLARRI-O1 v4.0
=====================================

3 niveles de dificultad:
1. Para niños / principiantes
2. Nivel medio
3. Nivel avanzado

Autor: Lucas Ricardo Mella Chillemi
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Colores
AZUL = '#3498db'
VERDE = '#27ae60'
ROJO = '#e74c3c'
NARANJA = '#f39c12'
MORADO = '#9b59b6'
AMARILLO = '#f1c40f'
GRIS = '#95a5a6'
OSCURO = '#2c3e50'


def diagrama_nivel1_ninos():
    """
    NIVEL 1 - Para niños o gente sin conocimiento técnico
    Analogía simple: Robot con 6 cajas que comparten todo
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#fffef0')
    fig.patch.set_facecolor('#fffef0')
    
    # Título
    ax.text(7, 9.3, 'LLARRI-O1: El Robot que Aprende a Ver Numeros', 
            fontsize=18, ha='center', fontweight='bold', color=OSCURO)
    
    # ===== PASO 1: ENTRADA =====
    ax.text(1.5, 8, 'PASO 1:', fontsize=12, fontweight='bold', color=OSCURO)
    ax.text(1.5, 7.5, 'Llega un dibujo', fontsize=10, color=GRIS)
    
    # Dibujo del número
    caja = FancyBboxPatch((0.5, 5.5), 2, 1.8, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor=AZUL, linewidth=2)
    ax.add_patch(caja)
    ax.text(1.5, 6.4, '7', fontsize=36, ha='center', va='center', 
            color=AZUL, fontweight='bold')
    
    # Flecha
    ax.annotate('', xy=(3.5, 6.4), xytext=(2.7, 6.4),
                arrowprops=dict(arrowstyle='->', color=OSCURO, lw=2))
    
    # ===== PASO 2: 6 CAJAS =====
    ax.text(6.5, 8, 'PASO 2: Pasa por 6 cajas', fontsize=12, 
            fontweight='bold', color=OSCURO)
    
    # 3 cajas arriba (MIRAN)
    ax.text(6.5, 7.5, '3 cajas que MIRAN:', fontsize=10, color=GRIS)
    for i, (nombre, color) in enumerate([('A', AZUL), ('B', VERDE), ('C', MORADO)]):
        x = 4 + i * 2
        caja = FancyBboxPatch((x, 6), 1.5, 1, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(caja)
        ax.text(x + 0.75, 6.5, nombre, fontsize=14, ha='center', va='center', 
                color='white', fontweight='bold')
    
    # Flechas IDA entre cajas de arriba
    for i in range(2):
        x = 5.6 + i * 2
        ax.annotate('', xy=(x + 0.3, 6.5), xytext=(x, 6.5),
                    arrowprops=dict(arrowstyle='->', color=OSCURO, lw=1.5))
    
    # Flechas hacia abajo
    for i in range(3):
        x = 4.75 + i * 2
        ax.annotate('', xy=(x, 4.8), xytext=(x, 5.9),
                    arrowprops=dict(arrowstyle='<->', color=GRIS, lw=1.5))
    
    # 3 cajas abajo (PIENSAN)
    ax.text(6.5, 4.5, '3 cajas que PIENSAN:', fontsize=10, color=GRIS)
    for i, (nombre, color) in enumerate([('D', ROJO), ('E', NARANJA), ('F', '#1abc9c')]):
        x = 4 + i * 2
        caja = FancyBboxPatch((x, 3), 1.5, 1, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(caja)
        ax.text(x + 0.75, 3.5, nombre, fontsize=14, ha='center', va='center', 
                color='white', fontweight='bold')
    
    # Flechas IDA entre cajas de abajo
    for i in range(2):
        x = 5.6 + i * 2
        ax.annotate('', xy=(x + 0.3, 3.5), xytext=(x, 3.5),
                    arrowprops=dict(arrowstyle='->', color=OSCURO, lw=1.5))
    
    # Conexiones explicación
    ax.text(10.5, 5.5, 'COMPARTEN\nTODO:', fontsize=10, ha='center', 
            fontweight='bold', color=OSCURO)
    ax.text(10.5, 4.8, '1. Ida', fontsize=9, ha='center', color=VERDE)
    ax.text(10.5, 4.4, '2. Vuelta', fontsize=9, ha='center', color=ROJO)
    ax.text(10.5, 4.0, '3. Ida+Vuelta', fontsize=9, ha='center', color=MORADO)
    
    # ===== PASO 3: RESPUESTA =====
    ax.annotate('', xy=(7, 1.8), xytext=(7, 2.9),
                arrowprops=dict(arrowstyle='->', color=OSCURO, lw=2))
    
    caja = FancyBboxPatch((5.5, 0.5), 3, 1.2, boxstyle="round,pad=0.05",
                          facecolor=VERDE, edgecolor='white', linewidth=2)
    ax.add_patch(caja)
    ax.text(7, 1.3, 'RESPUESTA', fontsize=11, ha='center', 
            fontweight='bold', color='white')
    ax.text(7, 0.8, 'Es un 7!', fontsize=10, ha='center', color='white')
    
    # Explicación
    ax.text(7, -0.3, 'El secreto: Todas las cajas comparten info en 3 direcciones', 
            fontsize=9, ha='center', color=GRIS, style='italic')
    
    # Créditos
    ax.text(7, -0.8, '(c) 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
            fontsize=7, ha='center', color='#bdc3c7')
    
    plt.tight_layout()
    return fig


def diagrama_nivel2_medio():
    """
    NIVEL 2 - Nivel medio / conceptual
    Muestra la estructura con flujo IDA/VUELTA/BIDI
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')
    
    # Título
    ax.text(8, 10.3, 'LLARRI-O1 v4.0 - Flujo Completo', fontsize=18, 
            ha='center', fontweight='bold', color=OSCURO)
    ax.text(8, 9.7, '6 Cajas con intercambio IDA + VUELTA + BIDIRECCIONAL', fontsize=11, 
            ha='center', color=GRIS)
    
    # ===== INPUT =====
    caja = FancyBboxPatch((0.5, 5.5), 2, 1.2, boxstyle="round,pad=0.05",
                          facecolor=AZUL, edgecolor='white', linewidth=2)
    ax.add_patch(caja)
    ax.text(1.5, 6.1, 'INPUT', fontsize=11, ha='center', 
            fontweight='bold', color='white')
    ax.text(1.5, 5.7, '784 px', fontsize=9, ha='center', color='white')
    
    ax.annotate('', xy=(3, 6.1), xytext=(2.6, 6.1),
                arrowprops=dict(arrowstyle='->', color=OSCURO, lw=2))
    
    # ===== 6 CAJAS EN LINEA =====
    cajas_info = [
        ('A', AZUL, 3.2), ('B', VERDE, 4.8), ('C', MORADO, 6.4),
        ('D', ROJO, 8.0), ('E', NARANJA, 9.6), ('F', '#1abc9c', 11.2)
    ]
    
    for nombre, color, x in cajas_info:
        caja = FancyBboxPatch((x, 5.2), 1.4, 1.8, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(caja)
        ax.text(x + 0.7, 6.3, nombre, fontsize=14, ha='center', 
                fontweight='bold', color='white')
        ax.text(x + 0.7, 5.6, '4 cuad', fontsize=8, ha='center', color='white')
    
    # Etiquetas de capas
    ax.text(5.3, 7.3, 'CAPA DATOS', fontsize=10, ha='center', 
            fontweight='bold', color=AZUL)
    ax.text(10.1, 7.3, 'CAPA CALCULOS', fontsize=10, ha='center', 
            fontweight='bold', color=ROJO)
    
    # ===== FLUJO IDA (arriba) =====
    ax.text(8, 8.5, 'FASE 1 - IDA', fontsize=11, ha='center', 
            fontweight='bold', color=VERDE)
    for i in range(5):
        x = 4.7 + i * 1.6
        ax.annotate('', xy=(x + 0.5, 7.8), xytext=(x, 7.8),
                    arrowprops=dict(arrowstyle='->', color=VERDE, lw=2))
    ax.text(8, 7.5, 'A -> B -> C -> D -> E -> F', fontsize=9, 
            ha='center', color=VERDE, family='monospace')
    
    # ===== FLUJO VUELTA (abajo) =====
    ax.text(8, 3.8, 'FASE 2 - VUELTA', fontsize=11, ha='center', 
            fontweight='bold', color=ROJO)
    for i in range(5):
        x = 4.7 + i * 1.6
        ax.annotate('', xy=(x, 4.3), xytext=(x + 0.5, 4.3),
                    arrowprops=dict(arrowstyle='->', color=ROJO, lw=2))
    ax.text(8, 4.0, 'F -> E -> D -> C -> B -> A', fontsize=9, 
            ha='center', color=ROJO, family='monospace')
    
    # ===== FLUJO BIDI (en medio) =====
    ax.text(8, 2.5, 'FASE 3 - BIDIRECCIONAL', fontsize=11, ha='center', 
            fontweight='bold', color=MORADO)
    ax.text(8, 2.1, 'A<->B  B<->C  C<->D  D<->E  E<->F', fontsize=9, 
            ha='center', color=MORADO, family='monospace')
    ax.text(8, 1.7, '(todos intercambian simultaneamente)', fontsize=8, 
            ha='center', color=GRIS)
    
    # ===== OUTPUT =====
    ax.annotate('', xy=(8, 0.9), xytext=(8, 1.4),
                arrowprops=dict(arrowstyle='->', color=OSCURO, lw=2))
    
    caja = FancyBboxPatch((6.5, 0.2), 3, 0.7, boxstyle="round,pad=0.05",
                          facecolor=VERDE, edgecolor='white', linewidth=2)
    ax.add_patch(caja)
    ax.text(8, 0.55, 'OUTPUT: 10 clases', fontsize=10, ha='center', 
            fontweight='bold', color='white')
    
    # ===== PANEL DERECHO: Dentro de cada caja =====
    ax.text(14, 8.8, 'Cada caja:', fontsize=11, 
            ha='center', fontweight='bold', color=OSCURO)
    
    caja_ext = FancyBboxPatch((13, 6.5), 2.5, 2, boxstyle="round,pad=0.1",
                               facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(caja_ext)
    ax.text(14.25, 8.1, '4 Cuadrantes', fontsize=9, ha='center', color=OSCURO)
    ax.text(14.25, 7.6, '8 niveles:', fontsize=8, ha='center', color=GRIS)
    ax.text(14.25, 7.2, '2->4->8->...->256', fontsize=7, ha='center', color=GRIS)
    ax.text(14.25, 6.8, '+ Cache O(1)', fontsize=8, ha='center', color=AMARILLO)
    
    # Stats
    ax.text(14.25, 5.8, '~49M params', fontsize=9, ha='center', 
            fontweight='bold', color=OSCURO)
    ax.text(14.25, 5.4, '18% compresion', fontsize=8, ha='center', color=GRIS)
    
    ax.text(8, -0.3, '(c) 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
            fontsize=7, ha='center', color='#bdc3c7')
    
    plt.tight_layout()
    return fig


def diagrama_nivel3_avanzado():
    """
    NIVEL 3 - Avanzado / técnico completo
    Forward pass con flujo completo y módulos
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # ===== PANEL IZQUIERDO: FORWARD PASS =====
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')
    
    ax1.text(5, 11.5, 'FORWARD PASS', fontsize=16, ha='center', 
             fontweight='bold', color='#00ff88')
    
    pasos = [
        ('1. Cajas Datos: A,B,C = proceso(x)', AZUL),
        ('2. Cajas Calc: D,E,F = combina(datos)', ROJO),
        ('3. FASE IDA: A->B->C->D->E->F', VERDE),
        ('4. FASE VUELTA: F->E->D->C->B->A', NARANJA),
        ('5. FASE BIDI: A<->B, B<->C, C<->D...', MORADO),
        ('6. RETRO: Calc -> Datos', GRIS),
        ('7. FUSION: cat([datos, calc])', '#1abc9c'),
        ('8. OUTPUT: MLP(fusion) -> 10', '#00bcd4'),
    ]
    
    for i, (texto, color) in enumerate(pasos):
        y = 10 - i * 1.1
        caja = FancyBboxPatch((0.5, y-0.3), 9, 0.8, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.7)
        ax1.add_patch(caja)
        ax1.text(5, y+0.1, texto, fontsize=10, ha='center', va='center', 
                color='white', family='monospace')
    
    # ===== PANEL DERECHO: FLUJO DETALLADO =====
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')
    
    ax2.text(5, 11.5, 'SISTEMA DE FLUJO', fontsize=16, ha='center', 
             fontweight='bold', color='#00ff88')
    
    # IDA
    ax2.text(5, 10.5, 'FASE 1: IDA', fontsize=12, ha='center', 
             fontweight='bold', color=VERDE)
    ax2.text(5, 10, 'A -> B -> C -> D -> E -> F', fontsize=10, 
             ha='center', color=VERDE, family='monospace')
    ax2.text(5, 9.5, '5 LlaveConexion unidireccionales', fontsize=8, 
             ha='center', color='#888')
    
    # VUELTA
    ax2.text(5, 8.5, 'FASE 2: VUELTA', fontsize=12, ha='center', 
             fontweight='bold', color=NARANJA)
    ax2.text(5, 8, 'F -> E -> D -> C -> B -> A', fontsize=10, 
             ha='center', color=NARANJA, family='monospace')
    ax2.text(5, 7.5, '5 LlaveConexion inversas', fontsize=8, 
             ha='center', color='#888')
    
    # BIDI
    ax2.text(5, 6.5, 'FASE 3: BIDIRECCIONAL', fontsize=12, ha='center', 
             fontweight='bold', color=MORADO)
    ax2.text(5, 6, 'A<->B  B<->C  C<->D  D<->E  E<->F', fontsize=10, 
             ha='center', color=MORADO, family='monospace')
    ax2.text(5, 5.5, '5 LlaveBidireccional con gate adaptativo', fontsize=8, 
             ha='center', color='#888')
    
    # Modulos
    ax2.text(5, 4.3, 'MODULOS CLAVE:', fontsize=11, ha='center', 
             fontweight='bold', color='#00ff88')
    
    modulos = [
        ('LlaveConexion', 'Linear + residual*0.5', GRIS),
        ('LlaveBidireccional', 'Gate adaptativo, pesos compartidos', MORADO),
        ('SistemaFlujoCompleto', 'IDA + VUELTA + BIDI + fusion', VERDE),
    ]
    
    for i, (nombre, desc, color) in enumerate(modulos):
        y = 3.5 - i * 0.9
        caja = FancyBboxPatch((0.5, y-0.2), 9, 0.7, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.7)
        ax2.add_patch(caja)
        ax2.text(0.8, y+0.15, nombre, fontsize=9, ha='left', va='center', 
                color='white', fontweight='bold')
        ax2.text(0.8, y-0.1, desc, fontsize=8, ha='left', va='center', color='white')
    
    # Config
    ax2.text(5, 0.8, 'Params: ~49M | 18% compresion | 6 cajas | 8 niveles', 
             fontsize=8, ha='center', color='#888')
    
    fig.text(0.5, 0.02, '(c) 2024-2026 Lucas Ricardo Mella Chillemi - Segunda Cabeza', 
             fontsize=8, ha='center', color='#666')
    
    plt.tight_layout()
    return fig


def main():
    """Genera los 3 diagramas."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'diagrams', 'v4-current')
    os.makedirs(output_dir, exist_ok=True)
    
    # Limpiar PNGs anteriores
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(output_dir, f))
    
    print("=" * 50)
    print("GENERANDO DIAGRAMAS LLARRI-O1 v4.0")
    print("=" * 50)
    
    diagramas = [
        ('nivel1_ninos.png', diagrama_nivel1_ninos, 'Nivel 1 - Para ninos'),
        ('nivel2_medio.png', diagrama_nivel2_medio, 'Nivel 2 - Medio'),
        ('nivel3_avanzado.png', diagrama_nivel3_avanzado, 'Nivel 3 - Avanzado'),
    ]
    
    for filename, func, desc in diagramas:
        print(f"\nGenerando: {desc}...")
        fig = func()
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"   -> {filepath}")
    
    print("\n" + "=" * 50)
    print("LISTO! 3 diagramas generados")
    print("=" * 50)


if __name__ == "__main__":
    main()
