# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi 
"""
PAMPAr-o1 v9 - Modelo de Lenguaje Cerebral con Territorios

============================================================
DEFINICIÓN EN UNA FRASE:
============================================================
"PampaR es un cerebro artificial donde el tálamo orquesta tokens hacia 
territorios especializados (Expresivo, Contextual, Formal, Estructural) 
que colaboran via fronteras bidireccionales, combinando reglas explícitas 
(LLAVES 70%) con atención aprendida (30%) para generar lenguaje."
============================================================

Arquitectura:
- TÁLAMO: Orquestador central con LLAVES (reglas + aprendizaje)
- 4 TERRITORIOS: Agrupaciones funcionales de 6 módulos
  · Expresivo: Lenguaje + Creatividad
  · Contextual: Contexto (memoria de trabajo)
  · Formal: Lógica
  · Estructural: Patrones + Matemáticas
- 6 FRONTERAS: Conexiones bidireccionales entre territorios
- AXIOMAS: Razonamiento deductivo (modus ponens, silogismo) [opcional]
- MEMORIA: Experiencia acumulada [opcional]

Flujo de procesamiento:
1. Token → Embedding + Posición
2. Tálamo: LLAVES (70%) + Atención (30%) → Pesos por territorio/módulo
3. Territorios procesan (buffer compartido intra-territorio)
4. Fronteras intercambian señales (solo si ambos territorios activos > umbral)
5. Axiomas aplican razonamiento lógico [si habilitado]
6. LM Head → Siguiente token

Innovaciones técnicas:
- LLAVES: Routing híbrido con conocimiento a priori inyectado
- Territorios: De 18 sinapsis O(n²) a 6 fronteras eficientes
- Fronteras gateadas: Solo comunican cuando es necesario
- Axiomas diferenciables: Lógica simbólica en tensores

Versión: 9.0
Fecha: Enero 2026
"""

# Re-exportar desde model_v9
from .model_v9 import PampaR, BloqueTerrritorial

__all__ = ['PampaR', 'BloqueTerrritorial']
