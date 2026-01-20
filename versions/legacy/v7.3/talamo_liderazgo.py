# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Tálamo con Liderazgo Dinámico

En el cerebro real, cuando realizamos una tarea:
- Un área cortical DOMINA el procesamiento
- Las otras áreas se ACOPLAN al líder
- El tálamo orquesta esta coordinación

En LLARRI v7.3:
- El tálamo elige un módulo LÍDER según la tarea
- Los otros módulos SIGUEN al líder (reciben su señal)
- El acoplamiento es dinámico y diferenciado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TalamoConLiderazgo(nn.Module):
    """
    Tálamo que selecciona un módulo líder y coordina el acoplamiento.
    
    Principios:
    1. Liderazgo claro: un módulo domina según la tarea
    2. Acoplamiento diferenciado: cada módulo sigue al líder de forma distinta
    3. Señal de coordinación: el líder emite señal que otros reciben
    """
    
    # Mapeo de tipos de tarea a módulo líder
    TAREAS = {
        'lenguaje': 0,      # Sintaxis, gramática
        'logica': 1,        # Razonamiento
        'matematicas': 2,   # Números
        'patrones': 3,      # Secuencias
        'contexto': 4,      # Comprensión global
        'creatividad': 5,   # Generación libre
    }
    
    def __init__(
        self, 
        dim: int,
        n_modulos: int = 6,
        actividad_basal: float = 0.15,
        temperatura_seleccion: float = 0.5,  # Qué tan "afilada" es la selección de líder
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        self.actividad_basal = actividad_basal
        self.temperatura = temperatura_seleccion
        
        # === SELECTOR DE LÍDER ===
        # Analiza el input y decide qué módulo debe liderar
        self.detector_tarea = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_modulos),
        )
        
        # === MATRIZ DE ACOPLAMIENTO ===
        # Cómo cada módulo sigue a cada posible líder
        # acoplamiento[i, j] = cómo módulo j se acopla cuando i es líder
        self.matriz_acoplamiento = nn.Parameter(
            self._init_acoplamiento_natural()
        )
        
        # === MODULACIÓN DE INTENSIDAD ===
        # Qué tan fuerte es el acoplamiento (contextual)
        self.intensidad = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Estado para tracking
        self.ultimo_lider = None
        self.historia_lideres = []
        
    def _init_acoplamiento_natural(self) -> torch.Tensor:
        """
        Inicializa matriz de acoplamiento con relaciones "naturales".
        
        Ejemplo: cuando Lenguaje lidera, Lógica y Contexto lo siguen fuerte,
        pero Matemáticas tiene acoplamiento más débil.
        """
        n = self.n_modulos
        matriz = torch.zeros(n, n)
        
        # Definir acoplamientos naturales entre módulos
        # Líder (fila) -> Seguidor (columna), valor = intensidad acoplamiento
        
        # Cuando LENGUAJE lidera (0)
        matriz[0] = torch.tensor([1.0, 0.6, 0.2, 0.5, 0.8, 0.5])  # Lógica, Contexto siguen fuerte
        
        # Cuando LÓGICA lidera (1)
        matriz[1] = torch.tensor([0.6, 1.0, 0.7, 0.4, 0.5, 0.3])  # Matemáticas sigue
        
        # Cuando MATEMÁTICAS lidera (2)
        matriz[2] = torch.tensor([0.3, 0.8, 1.0, 0.7, 0.3, 0.2])  # Lógica, Patrones siguen
        
        # Cuando PATRONES lidera (3)
        matriz[3] = torch.tensor([0.5, 0.4, 0.5, 1.0, 0.6, 0.7])  # Contexto, Creatividad siguen
        
        # Cuando CONTEXTO lidera (4)
        matriz[4] = torch.tensor([0.8, 0.5, 0.3, 0.6, 1.0, 0.6])  # Lenguaje sigue fuerte
        
        # Cuando CREATIVIDAD lidera (5)
        matriz[5] = torch.tensor([0.5, 0.3, 0.3, 0.7, 0.6, 1.0])  # Patrones sigue
        
        return matriz
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Determina módulo líder y modulación de seguidores.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            liderazgo: (batch, n_modulos) probabilidad de ser líder (soft)
            modulacion: (batch, n_modulos) intensidad de cada módulo
            stats: diccionario con información del procesamiento
        """
        batch = x.shape[0]
        
        # Contexto global para decidir liderazgo
        contexto = x.mean(dim=1)  # (batch, dim)
        
        # 1. SELECCIÓN DE LÍDER
        # Scores de liderazgo para cada módulo
        scores_lider = self.detector_tarea(contexto)  # (batch, n_modulos)
        
        # Softmax con temperatura para selección más o menos "dura"
        # temperatura baja = selección más definida (un líder claro)
        # temperatura alta = selección más difusa
        liderazgo = F.softmax(scores_lider / self.temperatura, dim=-1)
        
        # 2. CALCULAR MODULACIÓN BASADA EN ACOPLAMIENTO
        # Cada seguidor ajusta según quién es el líder
        # modulacion[j] = sum_i(liderazgo[i] * acoplamiento[i,j])
        
        # Normalizar matriz de acoplamiento (softmax por columna del seguidor)
        acoplamiento_norm = F.softmax(self.matriz_acoplamiento, dim=0)
        
        # Modulación = producto de liderazgo × acoplamiento
        # (batch, n) × (n, n) -> (batch, n)
        modulacion = liderazgo @ acoplamiento_norm
        
        # 3. INTENSIDAD CONTEXTUAL
        # Puede subir o bajar el acoplamiento general según contexto
        intensidad = self.intensidad(contexto)  # (batch, 1)
        
        # Escalar modulación por intensidad (mantener actividad basal)
        modulacion = (
            self.actividad_basal + 
            modulacion * (1.0 - self.actividad_basal) * intensidad
        )
        
        # 4. GUARDAR ESTADÍSTICAS
        lider_idx = liderazgo.argmax(dim=-1)  # (batch,)
        self.ultimo_lider = lider_idx
        
        nombres = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
        stats = {
            'lider_idx': lider_idx[0].item() if batch > 0 else -1,
            'lider_nombre': nombres[lider_idx[0].item()] if batch > 0 else 'ninguno',
            'lider_confianza': liderazgo.max(dim=-1)[0].mean().item(),
            'intensidad_media': intensidad.mean().item(),
        }
        
        # Agregar modulación por módulo
        for i, nombre in enumerate(nombres):
            stats[f'mod_{nombre}'] = modulacion[:, i].mean().item()
        
        return liderazgo, modulacion, stats
    
    def get_matriz_acoplamiento(self) -> torch.Tensor:
        """Retorna la matriz de acoplamiento aprendida (para visualización)."""
        return F.softmax(self.matriz_acoplamiento, dim=0).detach()
    
    def reset_estado(self):
        """Reinicia el estado para nueva secuencia."""
        self.ultimo_lider = None
        self.historia_lideres = []


class SenalLiderazgo(nn.Module):
    """
    Genera la señal que el líder envía a sus seguidores.
    
    El líder no solo activa a otros módulos, sino que les envía
    información sobre CÓMO procesar (qué buscar, qué enfatizar).
    """
    
    def __init__(self, dim: int, n_modulos: int = 6):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        
        # Cada módulo genera una señal diferente como líder
        self.generadores_senal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )
            for _ in range(n_modulos)
        ])
        
        # Cada módulo tiene un receptor diferente
        self.receptores = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),  # input + señal
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for _ in range(n_modulos)
        ])
        
    def generar_senal(
        self,
        outputs_modulos: list,
        liderazgo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Genera la señal de coordinación del líder.
        
        Args:
            outputs_modulos: Lista de outputs de cada módulo
            liderazgo: (batch, n_modulos) probabilidades de liderazgo
            
        Returns:
            senal: (batch, seq, dim) señal de coordinación
        """
        batch, seq, dim = outputs_modulos[0].shape
        
        # Calcular señal de cada módulo (como si fuera líder)
        senales = []
        for i, (output, generador) in enumerate(zip(outputs_modulos, self.generadores_senal)):
            # Señal basada en el procesamiento del módulo
            senal_i = generador(output.mean(dim=1))  # (batch, dim)
            senales.append(senal_i)
        
        # Combinar señales ponderadas por probabilidad de liderazgo
        # senal = sum_i(liderazgo[i] * senal[i])
        senal_combinada = torch.zeros(batch, dim, device=outputs_modulos[0].device)
        for i, senal_i in enumerate(senales):
            senal_combinada += liderazgo[:, i:i+1] * senal_i
            
        return senal_combinada
    
    def aplicar_senal(
        self,
        outputs_modulos: list,
        senal: torch.Tensor,
        modulacion: torch.Tensor,
    ) -> list:
        """
        Aplica la señal del líder a cada módulo seguidor.
        
        Args:
            outputs_modulos: Lista de outputs de cada módulo
            senal: (batch, dim) señal de coordinación
            modulacion: (batch, n_modulos) intensidad de cada módulo
            
        Returns:
            outputs_acoplados: Lista de outputs modificados
        """
        batch, seq, dim = outputs_modulos[0].shape
        outputs_acoplados = []
        
        # Expandir señal a secuencia
        senal_expanded = senal.unsqueeze(1).expand(-1, seq, -1)  # (batch, seq, dim)
        
        for i, (output, receptor) in enumerate(zip(outputs_modulos, self.receptores)):
            # Concatenar output con señal
            combinado = torch.cat([output, senal_expanded], dim=-1)
            
            # Aplicar receptor (cada módulo interpreta la señal diferente)
            ajuste = receptor(combinado)
            
            # Mezclar: original + ajuste modulado
            # modulación alta = sigue más al líder
            mod_i = modulacion[:, i:i+1].unsqueeze(1)  # (batch, 1, 1)
            output_acoplado = output + mod_i * 0.3 * ajuste
            
            outputs_acoplados.append(output_acoplado)
            
        return outputs_acoplados
