"""
DIAGRAMA SUPER BASICO: Modelo Normal vs LLARRI-O1
=================================================

Para personas que NO saben de IA
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# Crear figura con 2 diagramas
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('white')

# ================================================================
# DIAGRAMA 1: MODELO NORMAL (Como una fabrica en linea)
# ================================================================

ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
ax1.set_facecolor('white')

# Titulo
ax1.text(5, 13.5, 'MODELO NORMAL', fontsize=18, fontweight='bold', 
         ha='center', color='#333333')
ax1.text(5, 12.8, '(Como una fabrica en linea recta)', fontsize=12, 
         ha='center', color='#666666', style='italic')

# Cajas en linea vertical
cajas_normal = [
    ("ENTRADA\n(foto/texto)", 5, 11, '#4CAF50', "Entra la informacion"),
    ("CAJA 1", 5, 9, '#2196F3', "Procesa un poco"),
    ("CAJA 2", 5, 7, '#2196F3', "Procesa mas"),
    ("CAJA 3", 5, 5, '#2196F3', "Procesa mas"),
    ("CAJA 4", 5, 3, '#2196F3', "Ultimo proceso"),
    ("SALIDA\n(respuesta)", 5, 1, '#FF9800', "Sale la respuesta"),
]

for nombre, x, y, color, descripcion in cajas_normal:
    # Caja
    rect = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#333', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, y, nombre, ha='center', va='center', fontsize=10, 
             fontweight='bold', color='white')
    
    # Descripcion al lado
    ax1.text(x+2, y, descripcion, ha='left', va='center', fontsize=9, color='#666')

# Flechas (solo hacia abajo)
for i in range(len(cajas_normal)-1):
    y_start = cajas_normal[i][2] - 0.5
    y_end = cajas_normal[i+1][2] + 0.5
    ax1.annotate('', xy=(5, y_end), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))

# Nota importante
ax1.text(5, -0.5, 'La info SOLO va para abajo', fontsize=11, ha='center',
         color='#d32f2f', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#ffebee', edgecolor='#d32f2f'))

# Icono de flecha unidireccional
ax1.annotate('', xy=(8.5, 6), xytext=(8.5, 10),
            arrowprops=dict(arrowstyle='->', color='#999', lw=5))
ax1.text(9, 8, 'UNA\nDIRECCION', ha='left', va='center', fontsize=9, color='#999')

# ================================================================
# DIAGRAMA 2: LLARRI-O1 (Como un equipo que se comunica)
# ================================================================

ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis('off')
ax2.set_facecolor('white')

# Titulo
ax2.text(5, 13.5, 'LLARRI-O1', fontsize=18, fontweight='bold', 
         ha='center', color='#FF6B00')
ax2.text(5, 12.8, '(Como un equipo que habla entre si)', fontsize=12, 
         ha='center', color='#666666', style='italic')

# Entrada
rect_entrada = FancyBboxPatch((3.5, 10.5), 3, 1, boxstyle="round,pad=0.05",
                              facecolor='#4CAF50', edgecolor='#333', linewidth=2)
ax2.add_patch(rect_entrada)
ax2.text(5, 11, 'ENTRADA', ha='center', va='center', fontsize=10, 
         fontweight='bold', color='white')

# Las 3 cajas principales (triangulo)
cajas_llarri = [
    ("CAJA 1\n(Papa)", 2.5, 7.5, '#FFD700', 1.8),
    ("CAJA 2\n(Hijo)", 7.5, 7.5, '#FF6B00', 1.8),
    ("CAJA 3\n(Union)", 5, 4, '#E91E63', 2),
]

for nombre, x, y, color, size in cajas_llarri:
    rect = FancyBboxPatch((x-size/2, y-size/2), size, size, 
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#333', linewidth=3)
    ax2.add_patch(rect)
    ax2.text(x, y, nombre, ha='center', va='center', fontsize=10, 
             fontweight='bold', color='white')

# Salida
rect_salida = FancyBboxPatch((3.5, 0.5), 3, 1, boxstyle="round,pad=0.05",
                             facecolor='#FF9800', edgecolor='#333', linewidth=2)
ax2.add_patch(rect_salida)
ax2.text(5, 1, 'SALIDA', ha='center', va='center', fontsize=10, 
         fontweight='bold', color='white')

# Flechas de entrada
ax2.annotate('', xy=(2.5, 8.4), xytext=(4, 10.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax2.annotate('', xy=(7.5, 8.4), xytext=(6, 10.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# FLECHAS BIDIRECCIONALES (lo importante!)
# Caja 1 <-> Caja 2
ax2.annotate('', xy=(6.5, 7.5), xytext=(3.5, 7.5),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))
ax2.text(5, 8.3, 'HABLAN!', ha='center', fontsize=9, color='#00C853', fontweight='bold')

# Caja 1 <-> Caja 3
ax2.annotate('', xy=(4, 4.8), xytext=(2.5, 6.6),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))

# Caja 2 <-> Caja 3
ax2.annotate('', xy=(6, 4.8), xytext=(7.5, 6.6),
            arrowprops=dict(arrowstyle='<->', color='#00C853', lw=4))

# Flecha a salida
ax2.annotate('', xy=(5, 1.5), xytext=(5, 3),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Nota importante
ax2.text(5, -0.5, 'Las cajas SE COMUNICAN entre si!', fontsize=11, ha='center',
         color='#00C853', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#00C853'))

# Descripcion al lado
ax2.text(9.5, 7.5, 'Cada caja\npuede PREGUNTAR\na las otras', ha='center', 
         va='center', fontsize=9, color='#666',
         bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#FF6B00'))

# ================================================================
# CUADRO COMPARATIVO ABAJO
# ================================================================

comparison = """
    MODELO NORMAL                           LLARRI-O1
    ─────────────────                       ─────────────────
    Info va en UNA direccion  -->           Info va en TODAS direcciones  <-->
    
    Las cajas NO se hablan                  Las cajas SI se hablan
    
    Como una fila de personas               Como un grupo de amigos
    pasandose un papel                      discutiendo juntos
"""

fig.text(0.5, 0.02, comparison, ha='center', va='bottom', fontsize=10,
         family='monospace', color='#333',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc', pad=0.8))

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

# Guardar
plt.savefig('diagrama_basico_llarri.png', dpi=150, facecolor='white', 
            edgecolor='none', bbox_inches='tight')
print("Guardado: diagrama_basico_llarri.png")

plt.show()
