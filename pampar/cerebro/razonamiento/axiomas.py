# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Axiomas - Reglas fundamentales de razonamiento deductivo

Los axiomas son reglas que SIEMPRE son verdaderas y permiten
derivar conclusiones válidas a partir de premisas.

Implementados:
1. Modus Ponens: Si A→B y A, entonces B
2. Modus Tollens: Si A→B y ¬B, entonces ¬A  
3. Silogismo: Si A→B y B→C, entonces A→C
4. Contradicción: No puede ser A y ¬A simultáneamente
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TipoAxioma(Enum):
    """Tipos de axiomas lógicos."""
    MODUS_PONENS = "modus_ponens"       # A→B, A ⊢ B
    MODUS_TOLLENS = "modus_tollens"     # A→B, ¬B ⊢ ¬A
    SILOGISMO = "silogismo"             # A→B, B→C ⊢ A→C
    CONTRADICCION = "contradiccion"      # ¬(A ∧ ¬A)
    IDENTIDAD = "identidad"              # A ⊢ A
    DOBLE_NEGACION = "doble_negacion"   # ¬¬A ⊢ A


@dataclass
class Proposicion:
    """
    Representa una proposición lógica en forma vectorial.
    
    El vector `contenido` codifica el significado semántico.
    `es_negacion` indica si es la negación de algo.
    `confianza` indica qué tan seguro está el modelo.
    """
    contenido: torch.Tensor  # (dim,) vector de significado
    es_negacion: bool = False
    confianza: float = 1.0
    
    def negar(self) -> 'Proposicion':
        """Retorna la negación de esta proposición."""
        return Proposicion(
            contenido=self.contenido,
            es_negacion=not self.es_negacion,
            confianza=self.confianza * 0.95  # Ligera pérdida de confianza
        )


class Axioma(nn.Module):
    """
    Base para implementar axiomas lógicos.
    
    Cada axioma:
    - Detecta si es aplicable al contexto actual
    - Si es aplicable, deriva una conclusión
    - Tiene un peso de confianza
    """
    
    def __init__(self, dim: int, tipo: TipoAxioma):
        super().__init__()
        self.dim = dim
        self.tipo = tipo
        
        # Detector de aplicabilidad
        self.detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
    
    def es_aplicable(self, premisa1: torch.Tensor, premisa2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Determina si el axioma es aplicable."""
        raise NotImplementedError
    
    def aplicar(self, premisa1: torch.Tensor, premisa2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aplica el axioma y retorna la conclusión."""
        raise NotImplementedError


class ModusPonens(Axioma):
    """
    Modus Ponens: Si A implica B, y A es verdadero, entonces B es verdadero.
    
    A → B
    A
    ∴ B
    """
    
    def __init__(self, dim: int):
        super().__init__(dim, TipoAxioma.MODUS_PONENS)
        
        # Detector de implicación
        self.detector_implicacion = nn.Bilinear(dim, dim, 1)
        
        # Derivador de conclusión
        self.derivador = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def es_aplicable(
        self, 
        antecedente: torch.Tensor,      # A
        consecuente: torch.Tensor,       # B
        afirmacion: torch.Tensor         # Afirmamos A
    ) -> torch.Tensor:
        """
        Verifica si podemos aplicar modus ponens.
        
        Necesitamos:
        1. Que exista implicación A→B
        2. Que se afirme A
        """
        # ¿Hay implicación?
        impl_score = torch.sigmoid(self.detector_implicacion(antecedente, consecuente))
        
        # ¿Se afirma el antecedente?
        similitud = F.cosine_similarity(antecedente, afirmacion, dim=-1, eps=1e-6)
        afirma_score = (similitud + 1) / 2  # Normalizar a [0, 1]
        
        # Aplicable si ambas condiciones se cumplen
        return impl_score.squeeze(-1) * afirma_score
    
    def aplicar(
        self, 
        antecedente: torch.Tensor, 
        consecuente: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica modus ponens: dado A→B y A, deriva B.
        """
        combinado = torch.cat([antecedente, consecuente], dim=-1)
        return self.derivador(combinado)


class ModusTollens(Axioma):
    """
    Modus Tollens: Si A implica B, y B es falso, entonces A es falso.
    
    A → B
    ¬B
    ∴ ¬A
    """
    
    def __init__(self, dim: int):
        super().__init__(dim, TipoAxioma.MODUS_TOLLENS)
        
        # Detector de implicación
        self.detector_implicacion = nn.Bilinear(dim, dim, 1)
        
        # Negador
        self.negador = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),  # Tanh porque negación puede invertir signos
        )
    
    def es_aplicable(
        self, 
        antecedente: torch.Tensor,
        consecuente: torch.Tensor,
        negacion_consecuente: torch.Tensor
    ) -> torch.Tensor:
        """Verifica si podemos aplicar modus tollens."""
        # ¿Hay implicación?
        impl_score = torch.sigmoid(self.detector_implicacion(antecedente, consecuente))
        
        # ¿Se niega el consecuente?
        similitud = F.cosine_similarity(consecuente, negacion_consecuente, dim=-1, eps=1e-6)
        # Si son "opuestos", la similitud debería ser baja/negativa
        niega_score = (1 - similitud) / 2
        
        return impl_score.squeeze(-1) * niega_score
    
    def aplicar(self, antecedente: torch.Tensor) -> torch.Tensor:
        """Aplica modus tollens: deriva ¬A."""
        return self.negador(antecedente)


class Silogismo(Axioma):
    """
    Silogismo Hipotético: Si A→B y B→C, entonces A→C.
    
    A → B
    B → C
    ∴ A → C
    """
    
    def __init__(self, dim: int):
        super().__init__(dim, TipoAxioma.SILOGISMO)
        
        # Detector de cadena
        self.detector_cadena = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        
        # Encadenador
        self.encadenador = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def es_aplicable(
        self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        c: torch.Tensor
    ) -> torch.Tensor:
        """Verifica si existe cadena A→B→C."""
        combinado = torch.cat([a, b, c], dim=-1)
        return self.detector_cadena(combinado).squeeze(-1)
    
    def aplicar(self, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Deriva la implicación A→C."""
        return self.encadenador(torch.cat([a, c], dim=-1))


class MotorAxiomas(nn.Module):
    """
    Motor que aplica axiomas al procesamiento.
    
    Integra los axiomas con el flujo de la red neuronal:
    1. Detecta oportunidades de razonamiento deductivo
    2. Aplica axiomas cuando es apropiado
    3. Combina resultados con el procesamiento normal
    """
    
    def __init__(self, dim: int, usar_axiomas: List[TipoAxioma] = None):
        super().__init__()
        self.dim = dim
        
        # Axiomas a usar
        if usar_axiomas is None:
            usar_axiomas = [
                TipoAxioma.MODUS_PONENS, 
                TipoAxioma.MODUS_TOLLENS,
                TipoAxioma.SILOGISMO,
            ]
        
        # Inicializar axiomas
        self.axiomas = nn.ModuleDict()
        for tipo in usar_axiomas:
            if tipo == TipoAxioma.MODUS_PONENS:
                self.axiomas['modus_ponens'] = ModusPonens(dim)
            elif tipo == TipoAxioma.MODUS_TOLLENS:
                self.axiomas['modus_tollens'] = ModusTollens(dim)
            elif tipo == TipoAxioma.SILOGISMO:
                self.axiomas['silogismo'] = Silogismo(dim)
        
        # Detector de contexto lógico
        self.detector_contexto_logico = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Combinador de razonamiento con entrada original
        self.combinador = nn.Linear(dim * 2, dim)
        
        # Peso aprendible de cuánto contribuyen los axiomas
        self.peso_axiomas = nn.Parameter(torch.tensor(0.3))
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Aplica razonamiento axiomático al tensor de entrada.
        
        Args:
            x: (batch, seq, dim) embeddings a procesar
            mask: máscara de atención opcional
            
        Returns:
            x_razonado: tensor con razonamiento aplicado
            stats: estadísticas de qué axiomas se aplicaron
        """
        batch, seq, dim = x.shape
        stats = {k: 0.0 for k in self.axiomas.keys()}
        
        # Detectar si el contexto es propicio para razonamiento lógico
        contexto_logico = self.detector_contexto_logico(x)  # (batch, seq, 1)
        
        # Si no hay contexto lógico, retornar sin cambios
        if contexto_logico.mean() < 0.2:
            return x, stats
        
        # Preparar para razonamiento
        razonamiento = torch.zeros_like(x)
        
        # Aplicar Modus Ponens
        if 'modus_ponens' in self.axiomas:
            mp = self.axiomas['modus_ponens']
            # Usar posiciones consecutivas como premisas
            if seq >= 2:
                antecedente = x[:, :-1, :]
                consecuente = x[:, 1:, :]
                
                # Calcular aplicabilidad
                aplicable = mp.es_aplicable(antecedente, consecuente, antecedente)
                
                # Derivar conclusiones
                conclusion = mp.aplicar(antecedente, consecuente)
                
                # Añadir al razonamiento (con peso por aplicabilidad)
                razonamiento[:, 1:, :] += conclusion * aplicable.unsqueeze(-1)
                stats['modus_ponens'] = aplicable.mean().item()
        
        # Aplicar Silogismo
        if 'silogismo' in self.axiomas and seq >= 3:
            sil = self.axiomas['silogismo']
            a = x[:, :-2, :]
            b = x[:, 1:-1, :]
            c = x[:, 2:, :]
            
            aplicable = sil.es_aplicable(a, b, c)
            conclusion = sil.aplicar(a, c)
            
            razonamiento[:, 2:, :] += conclusion * aplicable.unsqueeze(-1)
            stats['silogismo'] = aplicable.mean().item()
        
        # Combinar razonamiento con entrada original
        peso = torch.sigmoid(self.peso_axiomas) * contexto_logico
        x_razonado = x + peso * razonamiento
        
        return x_razonado, stats
    
    def obtener_peso_axiomas(self) -> float:
        """Retorna el peso actual de los axiomas en el procesamiento."""
        return torch.sigmoid(self.peso_axiomas).item()
