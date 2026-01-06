"""
DIAGRAMA SUPER SIMPLE - Tipo explicacion para ninos
====================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('#FAFAFA')

# ================================================================
# 1. MODELO NORMAL = FILA DE PERSONAS
# ================================================================
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('MODELO NORMAL\n"Fila pasando un papel"', fontsize=14, fontweight='bold', color='#333')

# Personas en fila
personas = [(1.5, 3), (3.5, 3), (5.5, 3), (7.5, 3)]
colores = ['#64B5F6', '#64B5F6', '#64B5F6', '#64B5F6']

for i, ((x, y), color) in enumerate(zip(personas, colores)):
    # Cabeza
    circle = Circle((x, y+0.5), 0.4, facecolor='#FFE0B2', edgecolor='#333', linewidth=2)
    ax1.add_patch(circle)
    # Cuerpo
    rect = FancyBboxPatch((x-0.3, y-0.8), 0.6, 1, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='#333', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, y-1.2, f'#{i+1}', ha='center', fontsize=10, color='#666')

# Flechas entre personas
for i in range(len(personas)-1):
    ax1.annotate('', xy=(personas[i+1][0]-0.5, 3), xytext=(personas[i][0]+0.5, 3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))

# Papel viajando
ax1.text(2.5, 4, 'papel', fontsize=8, ha='center', color='#999')
ax1.plot([2.5], [3.8], 'rs', markersize=10)

# Explicacion
ax1.text(5, 0.5, 'Cada persona pasa el papel al siguiente.\nNO pueden hablar con los de atras.',
         ha='center', fontsize=10, color='#666', style='italic')

# ================================================================
# 2. LLARRI = GRUPO DE AMIGOS
# ================================================================
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('LLARRI-O1\n"Grupo de amigos hablando"', fontsize=14, fontweight='bold', color='#FF6B00')

# Personas en circulo
personas_llarri = [(3, 4), (7, 4), (5, 1.5)]
colores_llarri = ['#FFD700', '#FF6B00', '#E91E63']
nombres = ['Papa', 'Hijo', 'Union']

for (x, y), color, nombre in zip(personas_llarri, colores_llarri, nombres):
    # Cabeza
    circle = Circle((x, y+0.5), 0.5, facecolor='#FFE0B2', edgecolor='#333', linewidth=2)
    ax2.add_patch(circle)
    # Cuerpo
    rect = FancyBboxPatch((x-0.4, y-0.9), 0.8, 1.1, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='#333', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x, y-1.3, nombre, ha='center', fontsize=10, color='#333', fontweight='bold')

# Lineas de comunicacion (bidireccionales)
ax2.annotate('', xy=(6, 4), xytext=(4, 4),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))
ax2.annotate('', xy=(4, 2), xytext=(3, 3.2),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))
ax2.annotate('', xy=(6, 2), xytext=(7, 3.2),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))

# Burbujas de dialogo
ax2.text(5, 5.2, '"Oye, que piensas?"', fontsize=9, ha='center', color='#00C853',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#00C853'))

# Explicacion
ax2.text(5, 0, 'Todos pueden hablar con todos.\nSe preguntan y se ayudan.',
         ha='center', fontsize=10, color='#666', style='italic')

# ================================================================
# 3. FLUJO NORMAL (Cascada)
# ================================================================
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)
ax3.axis('off')
ax3.set_title('COMO FLUYE LA INFO\n(Modelo Normal)', fontsize=12, fontweight='bold', color='#333')

# Cascada de agua
niveles = [(5, 5), (5, 4), (5, 3), (5, 2), (5, 1)]
for i, (x, y) in enumerate(niveles):
    rect = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, boxstyle="round,pad=0.02",
                          facecolor='#2196F3', edgecolor='#1565C0', linewidth=2)
    ax3.add_patch(rect)
    if i < len(niveles)-1:
        ax3.annotate('', xy=(5, niveles[i+1][1]+0.3), xytext=(5, y-0.3),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=3))

# Agua cayendo
ax3.text(7, 3, 'Como agua\ncayendo\npor escalones', ha='center', fontsize=10, 
         color='#1565C0', style='italic')

# X roja para "no sube"
ax3.text(3, 3, 'X', fontsize=30, ha='center', color='#d32f2f', fontweight='bold')
ax3.text(3, 2, 'No puede\nsubir', ha='center', fontsize=9, color='#d32f2f')

# ================================================================
# 4. FLUJO LLARRI (Red)
# ================================================================
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('COMO FLUYE LA INFO\n(LLARRI-O1)', fontsize=12, fontweight='bold', color='#FF6B00')

# Nodos conectados
nodos = [(3, 4.5), (7, 4.5), (5, 2)]
colores_nodos = ['#FFD700', '#FF6B00', '#E91E63']

for (x, y), color in zip(nodos, colores_nodos):
    circle = Circle((x, y), 0.6, facecolor=color, edgecolor='#333', linewidth=3)
    ax4.add_patch(circle)

# Conexiones bidireccionales
ax4.annotate('', xy=(6, 4.5), xytext=(4, 4.5),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=5))
ax4.annotate('', xy=(4.3, 2.5), xytext=(3, 3.9),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=5))
ax4.annotate('', xy=(5.7, 2.5), xytext=(7, 3.9),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=5))

# Check verde
ax4.text(8, 3, 'OK', fontsize=25, ha='center', color='#00C853', fontweight='bold')
ax4.text(8, 2.2, 'Va y\nvuelve!', ha='center', fontsize=9, color='#00C853')

# Explicacion central
ax4.text(5, 0.5, 'La info puede ir en CUALQUIER direccion', 
         ha='center', fontsize=11, color='#00C853', fontweight='bold')

# ================================================================
# MENSAJE FINAL
# ================================================================
fig.text(0.5, 0.01, 
         'RESUMEN: En modelos normales la info solo baja. En LLARRI, todos se comunican!',
         ha='center', fontsize=12, fontweight='bold', color='#333',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#00C853', pad=0.5))

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.savefig('diagrama_super_simple.png', dpi=150, facecolor='#FAFAFA', bbox_inches='tight')
print("Guardado: diagrama_super_simple.png")
plt.show()
