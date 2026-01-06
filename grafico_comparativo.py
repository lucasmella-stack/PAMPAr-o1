"""
🔺 LLARRI-O1: Gráfico Comparativo con Otros Modelos
===================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configurar estilo
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ================================================================
# DATOS DE MODELOS
# ================================================================

modelos = {
    # Nombre: (Parámetros en B, Tamaño GB, Modalidades, Color, Es LLARRI)
    
    # LLARRI Family
    "LLARRI-O1\nSmall": (0.0002, 0.00075, 1, "#FFD700", True),
    "LLARRI-O1\n100M": (0.058, 0.22, 1, "#FFD700", True),
    "LLARRI-O1\nMultimodal": (0.132, 0.49, 3, "#FFD700", True),
    "LLARRI-O1\n500M": (0.296, 1.1, 3, "#FFD700", True),
    "LLARRI-O1\n1B": (0.564, 2.1, 3, "#FFD700", True),
    "LLARRI-O1\n7B (proj)": (7.0, 14, 3, "#FFA500", True),
    
    # Otros modelos - Solo Texto
    "GPT-2\nSmall": (0.117, 0.5, 1, "#808080", False),
    "GPT-2\nLarge": (0.774, 3, 1, "#808080", False),
    "LLaMA\n7B": (7.0, 13, 1, "#4169E1", False),
    "Mistral\n7B": (7.0, 14, 1, "#9370DB", False),
    
    # Multimodales
    "CLIP\nViT-B": (0.151, 0.6, 2, "#32CD32", False),
    "CLIP\nViT-L": (0.428, 1.7, 2, "#32CD32", False),
    "BLIP": (0.446, 1.8, 2, "#00CED1", False),
    "LLaVA\n7B": (7.0, 13, 2, "#FF6347", False),
}

# ================================================================
# GRÁFICO 1: PARÁMETROS (Escala Log)
# ================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔺 LLARRI-O1 vs Otros Modelos - Comparativa', fontsize=16, fontweight='bold', color='white')

# Subplot 1: Parámetros
ax1 = axes[0, 0]

nombres = list(modelos.keys())
params = [modelos[m][0] for m in nombres]
colores = [modelos[m][3] for m in nombres]

bars = ax1.barh(nombres, params, color=colores, edgecolor='white', linewidth=0.5)
ax1.set_xscale('log')
ax1.set_xlabel('Parámetros (Billones)', fontsize=12)
ax1.set_title('📊 Parámetros por Modelo', fontsize=14, fontweight='bold')
ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='1B')
ax1.axvline(x=7, color='orange', linestyle='--', alpha=0.5, label='7B')

# Agregar valores
for bar, param in zip(bars, params):
    if param >= 1:
        label = f'{param:.1f}B'
    elif param >= 0.001:
        label = f'{param*1000:.0f}M'
    else:
        label = f'{param*1000000:.0f}K'
    ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2, 
             label, va='center', fontsize=8, color='white')

# ================================================================
# GRÁFICO 2: TAMAÑO EN GB
# ================================================================

ax2 = axes[0, 1]

tamaños = [modelos[m][1] for m in nombres]

bars2 = ax2.barh(nombres, tamaños, color=colores, edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Tamaño (GB)', fontsize=12)
ax2.set_title('💾 Tamaño del Modelo (GB)', fontsize=14, fontweight='bold')

# Líneas de referencia (memoria GPU común)
ax2.axvline(x=4, color='cyan', linestyle='--', alpha=0.5, label='GTX 1650 (4GB)')
ax2.axvline(x=8, color='lime', linestyle='--', alpha=0.5, label='RTX 3070 (8GB)')
ax2.axvline(x=24, color='yellow', linestyle='--', alpha=0.5, label='RTX 4090 (24GB)')

for bar, size in zip(bars2, tamaños):
    if size >= 1:
        label = f'{size:.1f} GB'
    else:
        label = f'{size*1000:.0f} MB'
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
             label, va='center', fontsize=8, color='white')

# ================================================================
# GRÁFICO 3: MODALIDADES
# ================================================================

ax3 = axes[1, 0]

modalidades = [modelos[m][2] for m in nombres]
modal_colors = ['#FF6B6B' if m == 1 else '#4ECDC4' if m == 2 else '#FFD93D' for m in modalidades]

bars3 = ax3.barh(nombres, modalidades, color=modal_colors, edgecolor='white', linewidth=0.5)
ax3.set_xlabel('Número de Modalidades', fontsize=12)
ax3.set_title('🎭 Modalidades Soportadas', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 4)

# Etiquetas
modal_labels = {1: '📝 Solo Texto', 2: '📝🖼️ Text+Vision', 3: '📝🖼️🎵 Text+Vision+Audio'}
for bar, modal in zip(bars3, modalidades):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             modal_labels[modal], va='center', fontsize=8, color='white')

# ================================================================
# GRÁFICO 4: DIAGRAMA DE CAJAS - Familia LLARRI
# ================================================================

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_aspect('equal')
ax4.axis('off')
ax4.set_title('🔺 Familia LLARRI-O1', fontsize=14, fontweight='bold')

# Dibujar cajas para cada variante de LLARRI
llarri_variants = [
    ("Small\n195K", 0.5, 8, 0.8, "#FFD700", "0.75 MB"),
    ("100M\n58M", 2, 8, 1.2, "#FFD700", "222 MB"),
    ("Multimodal\n132M", 4, 8, 1.5, "#FFD700", "490 MB"),
    ("500M\n296M", 6, 8, 1.8, "#FFA500", "1.1 GB"),
    ("1B\n564M", 8, 8, 2.0, "#FFA500", "2.1 GB"),
    ("7B\n(proj)", 5, 4, 3.0, "#FF4500", "14 GB"),
]

for name, x, y, size, color, label in llarri_variants:
    # Caja
    rect = mpatches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.8
    )
    ax4.add_patch(rect)
    
    # Texto
    ax4.text(x, y + 0.1, name, ha='center', va='center', fontsize=8, 
             fontweight='bold', color='black')
    ax4.text(x, y - size/2 - 0.3, label, ha='center', va='top', fontsize=7, color='white')

# Flechas de evolución
arrow_style = dict(arrowstyle='->', color='white', lw=1.5)
ax4.annotate('', xy=(1.8, 8), xytext=(1.1, 8), arrowprops=arrow_style)
ax4.annotate('', xy=(3.5, 8), xytext=(2.8, 8), arrowprops=arrow_style)
ax4.annotate('', xy=(5.5, 8), xytext=(4.7, 8), arrowprops=arrow_style)
ax4.annotate('', xy=(7.5, 8), xytext=(6.7, 8), arrowprops=arrow_style)
ax4.annotate('', xy=(5, 5.5), xytext=(5, 7), arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Texto de escala
ax4.text(5, 2, '🚀 ESCALANDO...', ha='center', fontsize=12, color='red', fontweight='bold')
ax4.text(5, 1, 'Trinity Fractal Architecture', ha='center', fontsize=10, color='white', style='italic')

# ================================================================
# LEYENDA
# ================================================================

legend_elements = [
    mpatches.Patch(facecolor='#FFD700', edgecolor='white', label='LLARRI-O1 (disponible)'),
    mpatches.Patch(facecolor='#FFA500', edgecolor='white', label='LLARRI-O1 (próximo)'),
    mpatches.Patch(facecolor='#FF4500', edgecolor='white', label='LLARRI-O1 7B (proyectado)'),
    mpatches.Patch(facecolor='#808080', edgecolor='white', label='Modelos solo texto'),
    mpatches.Patch(facecolor='#32CD32', edgecolor='white', label='CLIP (Vision+Text)'),
    mpatches.Patch(facecolor='#FF6347', edgecolor='white', label='LLaVA (Vision+Text)'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           framealpha=0.8, edgecolor='white')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Guardar
plt.savefig('llarri_o1_comparison.png', dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
print("✅ Gráfico guardado: llarri_o1_comparison.png")

plt.show()
