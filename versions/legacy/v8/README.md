# PampaR v8 (Legacy)

Esta carpeta contiene los archivos de la versión 8 del modelo PampaR, que usaba:
- **Sinapsis**: 18 conexiones individuales entre módulos
- **Módulos planos**: 6 módulos sin agrupación territorial

## Archivos

- `model.py` - Modelo principal con CerebralBlock y sinapsis
- `sinapsis.py` - Sistema de conexiones sinápticas entre módulos

## Razón del cambio a v9

La arquitectura v8 tenía problemas de eficiencia:
1. **18 sinapsis por forward pass** era computacionalmente costoso
2. **Sin agrupación funcional** - todos los módulos tratados por igual
3. **Comunicación O(n²)** entre módulos

La v9 introduce:
- **Territorios**: Agrupaciones funcionales (4 territorios vs 6 módulos individuales)
- **Fronteras**: 6 conexiones bidireccionales (vs 18 sinapsis)
- **Comunicación eficiente**: Solo se activan fronteras cuando ambos territorios están activos
