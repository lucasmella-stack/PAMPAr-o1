"""
🧠 COMPARACIÓN: Cerebro de Modelo Normal vs LLARRI-O1
=====================================================

Visualización de cómo fluye la información en:
1. Transformer tradicional (secuencial)
2. LLARRI-O1 Trinity Fractal (multidireccional)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor('#0d1117')

# ================================================================
# LADO IZQUIERDO: TRANSFORMER TRADICIONAL
# ================================================================

ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_facecolor('#0d1117')
ax1.set_title('TRANSFORMER TRADICIONAL\n(GPT, LLaMA, BERT, etc.)', 
              fontsize=14, fontweight='bold', color='white', pad=20)

# Capas del transformer (secuenciales)
layers = [
    ("ENTRADA\n(Tokens/Pixels)", 5, 1, '#4a9eff', 2.5),
    ("EMBEDDING\n+ Positional", 5, 3, '#6b7280', 2),
    ("ATTENTION\nLayer 1", 5, 5, '#ef4444', 2),
    ("FFN\nLayer 1", 5, 6.5, '#f97316', 1.5),
    ("ATTENTION\nLayer 2", 5, 8, '#ef4444', 2),
    ("FFN\nLayer 2", 5, 9.5, '#f97316', 1.5),
    ("SALIDA\n(Logits)", 5, 11, '#22c55e', 2),
]

for name, x, y, color, width in layers:
    rect = FancyBboxPatch(
        (x - width/2, y - 0.4), width, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
    )
    ax1.add_patch(rect)
    ax1.text(x, y, name, ha='center', va='center', fontsize=8, 
             fontweight='bold', color='white')

# Flechas secuenciales (UNA dirección)
arrow_props = dict(arrowstyle='->', color='#9ca3af', lw=2, mutation_scale=15)
positions = [(5, 1.5), (5, 3.5), (5, 5.5), (5, 7), (5, 8.5), (5, 10)]
for i in range(len(positions) - 1):
    ax1.annotate('', xy=(positions[i+1][0], positions[i+1][1] - 0.1), 
                 xytext=(positions[i][0], positions[i][1] + 0.3),
                 arrowprops=arrow_props)

# Texto explicativo
ax1.text(5, -0.5, 'Flujo UNIDIRECCIONAL', ha='center', fontsize=11, 
         color='#9ca3af', style='italic')
ax1.text(5, -1.2, 'La información solo va hacia adelante', ha='center', 
         fontsize=9, color='#6b7280')

# Nota lateral
ax1.text(8.5, 6, 'Cada capa\nprocesa y\npasa al\nsiguiente', ha='center', 
         fontsize=8, color='#6b7280', bbox=dict(boxstyle='round', facecolor='#1f2937', edgecolor='#374151'))

# ================================================================
# LADO DERECHO: LLARRI-O1 TRINITY FRACTAL
# ================================================================

ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_facecolor('#0d1117')
ax2.set_title('LLARRI-O1 TRINITY FRACTAL\n(Arquitectura de Lucas Mella)', 
              fontsize=14, fontweight='bold', color='#ffd700', pad=20)

# Las 3 CAJAS (Trinidad)
cajas = [
    ("CAJA 1\nPADRE\n(Vision)", 2.5, 8, '#ffd700', 2.5),
    ("CAJA 2\nHIJO\n(Texto)", 7.5, 8, '#ffa500', 2.5),
    ("CAJA 3\nESPIRITU\n(Fusion)", 5, 4, '#ff6b6b', 3),
]

for name, x, y, color, size in cajas:
    # Caja principal
    rect = FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=color, edgecolor='white', linewidth=3, alpha=0.85
    )
    ax2.add_patch(rect)
    ax2.text(x, y + 0.3, name, ha='center', va='center', fontsize=9, 
             fontweight='bold', color='black')
    
    # Mundos internos (fractales)
    for dx, dy in [(-0.5, -0.7), (0.5, -0.7), (0, -0.3)]:
        circle = Circle((x + dx, y + dy - 0.3), 0.25, 
                        facecolor='white', edgecolor=color, linewidth=2, alpha=0.7)
        ax2.add_patch(circle)

# Entrada y Salida
entrada = FancyBboxPatch((3.5, 10.5), 3, 0.8, boxstyle="round,pad=0.02",
                          facecolor='#4a9eff', edgecolor='white', linewidth=2)
ax2.add_patch(entrada)
ax2.text(5, 10.9, 'ENTRADA\n(Multimodal)', ha='center', va='center', 
         fontsize=8, fontweight='bold', color='white')

salida = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.02",
                         facecolor='#22c55e', edgecolor='white', linewidth=2)
ax2.add_patch(salida)
ax2.text(5, 0.9, 'SALIDA\n(Unificada)', ha='center', va='center', 
         fontsize=8, fontweight='bold', color='white')

# CONEXIONES BIDIRECCIONALES (el corazón de LLARRI)
# Doble flecha = bidireccional
def draw_bidirectional(ax, start, end, color='#00ff88'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='<->', color=color, lw=3, 
                               mutation_scale=20, connectionstyle="arc3,rad=0.1"))

# Caja 1 <-> Caja 2
draw_bidirectional(ax2, (3.5, 8), (6, 8), '#00ff88')
ax2.text(5, 8.5, 'Llave\nIda/Vuelta', ha='center', fontsize=7, color='#00ff88')

# Caja 1 <-> Caja 3
draw_bidirectional(ax2, (2.5, 6.5), (4, 5.2), '#00ff88')

# Caja 2 <-> Caja 3
draw_bidirectional(ax2, (7.5, 6.5), (6, 5.2), '#00ff88')

# Skip Connection Caja 1 <-> Caja 3 (la llave larga)
ax2.annotate('', xy=(3.2, 5), xytext=(2.5, 6.5),
             arrowprops=dict(arrowstyle='<->', color='#ff00ff', lw=3, 
                            mutation_scale=20, connectionstyle="arc3,rad=-0.3"))
ax2.text(1.5, 5.5, 'SKIP\n(Llave Larga)', ha='center', fontsize=7, color='#ff00ff')

# Entrada a Cajas
ax2.annotate('', xy=(2.5, 9.3), xytext=(4.2, 10.5),
             arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=2))
ax2.annotate('', xy=(7.5, 9.3), xytext=(5.8, 10.5),
             arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=2))

# Caja 3 a Salida
ax2.annotate('', xy=(5, 1.3), xytext=(5, 2.5),
             arrowprops=dict(arrowstyle='->', color='#22c55e', lw=2))

# Texto explicativo
ax2.text(5, -0.5, 'Flujo MULTIDIRECCIONAL', ha='center', fontsize=11, 
         color='#ffd700', style='italic', fontweight='bold')
ax2.text(5, -1.2, 'La informacion fluye en TODAS direcciones', ha='center', 
         fontsize=9, color='#ffa500')

# Detalle de mundos fractales
ax2.text(9, 4, 'Cada caja\ncontiene\nMUNDOS\ndentro de\nmundos', 
         ha='center', fontsize=8, color='#ffd700', 
         bbox=dict(boxstyle='round', facecolor='#1f2937', edgecolor='#ffd700'))

# ================================================================
# CUADRO COMPARATIVO EN EL CENTRO INFERIOR
# ================================================================

# Crear texto de comparación
comparison_text = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    DIFERENCIAS FUNDAMENTALES                                    ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  TRANSFORMER                        │  LLARRI-O1 TRINITY FRACTAL               ║
║  ─────────────────────────────────  │  ───────────────────────────────────     ║
║  • Flujo secuencial (A→B→C→D)       │  • Flujo multidireccional (A↔B↔C)        ║
║  • Capas independientes             │  • Cajas interconectadas (Trinity)       ║
║  • Sin estructura interna           │  • Mundos fractales dentro de mundos     ║
║  • Pesos unicos por capa            │  • Pesos compartidos + personalidades    ║
║  • Atencion dentro de cada capa     │  • Cross-attention entre modalidades     ║
║  • Una modalidad por modelo         │  • Multimodal nativo (Vision+Text+Audio) ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

fig.text(0.5, 0.02, comparison_text, ha='center', va='bottom', fontsize=9,
         family='monospace', color='white',
         bbox=dict(boxstyle='round', facecolor='#1f2937', edgecolor='#374151', pad=1))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# Guardar
plt.savefig('llarri_vs_transformer_brain.png', dpi=200, facecolor='#0d1117', 
            edgecolor='none', bbox_inches='tight')
print("✅ Grafico guardado: llarri_vs_transformer_brain.png")

plt.show()
