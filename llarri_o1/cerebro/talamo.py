# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
Tálamo - El Orquestador Central

El tálamo cerebral es la estación de relevo que:
- Recibe TODA la información sensorial
- Decide QUÉ módulo debe procesarla
- Usa LLAVES (detectores de dominio) no solo redes neuronales

Filosofía: "No mezclar peras con manzanas"
- Cada módulo tiene tokens que SON SU DOMINIO
- El tálamo SABE de antemano qué va a cada módulo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LlaveModulo:
    """
    LLAVE: Define el dominio de un módulo.
    
    Una llave contiene:
    - Tokens específicos que activan este módulo
    - Patrones regex/prefijos que lo activan
    - Peso base de activación
    """
    nombre: str
    tokens_clave: List[int]  # IDs de tokens que activan este módulo
    patrones: List[str]      # Patrones de texto para detección
    peso_base: float = 1.0   # Peso cuando se detecta coincidencia
    peso_fondo: float = 0.1  # Peso mínimo (siempre algo de activación)


class Talamo(nn.Module):
    """
    Orquestador central que distribuye señales a los módulos.
    
    Combina:
    1. LLAVES (reglas explícitas): detectores de dominio
    2. Atención aprendida: para casos ambiguos
    
    La proporción es configurable pero LLAVES tienen prioridad.
    """
    
    def __init__(
        self, 
        dim: int,
        n_modulos: int = 6,
        nombres_modulos: Optional[List[str]] = None,
        peso_llaves: float = 0.7,  # 70% reglas, 30% aprendido
        vocab_size: int = 8000,  # Tamaño del vocabulario
    ):
        super().__init__()
        self.dim = dim
        self.n_modulos = n_modulos
        self.peso_llaves = peso_llaves
        self.vocab_size = vocab_size
        
        # Nombres de los módulos
        self.nombres = nombres_modulos or [
            'lenguaje', 'logica', 'matematicas', 
            'patrones', 'contexto', 'creatividad'
        ]
        
        # ============================================
        # LLAVES - Reglas explícitas de dominio
        # ============================================
        self.llaves = self._crear_llaves()
        
        # Buffer para mapeo rápido token -> módulo
        # Se poblará cuando tengamos el tokenizer
        self.register_buffer('token_a_modulo', torch.zeros(vocab_size, n_modulos))
        
        # ============================================
        # Atención aprendida (para casos ambiguos)
        # ============================================
        self.query = nn.Linear(dim, n_modulos)
        
        # Normalización
        self.norm = nn.LayerNorm(dim)
        
        # Estadísticas de activación (para debug)
        self.register_buffer('activaciones_acumuladas', torch.zeros(n_modulos))
        self.register_buffer('n_llamadas', torch.tensor(0))
    
    def _crear_llaves(self) -> Dict[str, LlaveModulo]:
        """
        Crea las LLAVES para cada módulo.
        
        Estas son reglas EXPLÍCITAS basadas en conocimiento del dominio.
        """
        return {
            'lenguaje': LlaveModulo(
                nombre='lenguaje',
                tokens_clave=[],  # Se poblarán con tokenizer
                patrones=[
                    # Artículos, preposiciones, conectores
                    'el', 'la', 'los', 'las', 'un', 'una',
                    'de', 'en', 'por', 'para', 'con', 'sin',
                    'que', 'como', 'cuando', 'donde', 'quien',
                    'y', 'o', 'pero', 'aunque', 'porque',
                    # Verbos comunes
                    'es', 'son', 'fue', 'ser', 'estar',
                    'tiene', 'tiene', 'hay', 'puede',
                    # Palabras de texto
                    'texto', 'palabra', 'frase', 'párrafo',
                ],
                peso_base=1.0,
                peso_fondo=0.2,  # Siempre algo activo
            ),
            
            'logica': LlaveModulo(
                nombre='logica',
                tokens_clave=[],
                patrones=[
                    # Conectores lógicos
                    'si', 'entonces', 'porque', 'por lo tanto',
                    'sin embargo', 'aunque', 'mientras',
                    'implica', 'deduce', 'concluye',
                    # Términos lógicos
                    'verdadero', 'falso', 'correcto', 'incorrecto',
                    'válido', 'inválido', 'posible', 'imposible',
                    'necesario', 'suficiente', 'condición',
                    # Operadores
                    'y', 'o', 'no', 'ni', 'ambos', 'ninguno',
                ],
                peso_base=1.2,  # Más peso porque es importante
                peso_fondo=0.1,
            ),
            
            'matematicas': LlaveModulo(
                nombre='matematicas',
                tokens_clave=[],
                patrones=[
                    # Números
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    # Operadores
                    '+', '-', '*', '/', '=', '<', '>', '≤', '≥',
                    '×', '÷', '±', '∑', '∫', '√', '^', '%',
                    # Términos matemáticos
                    'suma', 'resta', 'multiplica', 'divide',
                    'igual', 'mayor', 'menor', 'resultado',
                    'ecuación', 'fórmula', 'cálculo', 'número',
                    'promedio', 'total', 'cantidad', 'valor',
                ],
                peso_base=1.5,  # Alto peso - muy específico
                peso_fondo=0.05,
            ),
            
            'patrones': LlaveModulo(
                nombre='patrones',
                tokens_clave=[],
                patrones=[
                    # Palabras de secuencia
                    'secuencia', 'patrón', 'serie', 'repetir',
                    'siguiente', 'anterior', 'primero', 'último',
                    'siempre', 'nunca', 'cada', 'todos', 'ninguno',
                    # Estructuras repetitivas
                    '...', '→', '↔', '∴',
                    # Palabras de orden
                    'orden', 'ordenar', 'clasificar', 'agrupar',
                ],
                peso_base=1.0,
                peso_fondo=0.1,
            ),
            
            'contexto': LlaveModulo(
                nombre='contexto',
                tokens_clave=[],
                patrones=[
                    # Referencias temporales/espaciales
                    'antes', 'después', 'durante', 'mientras',
                    'aquí', 'allí', 'donde', 'cuando',
                    'ahora', 'luego', 'pronto', 'tarde',
                    # Referencias
                    'este', 'ese', 'aquel', 'esto', 'eso',
                    'él', 'ella', 'ellos', 'su', 'sus',
                    # Memoria
                    'anterior', 'previo', 'mencionado', 'dicho',
                ],
                peso_base=0.8,
                peso_fondo=0.15,  # Contexto siempre importa
            ),
            
            'creatividad': LlaveModulo(
                nombre='creatividad',
                tokens_clave=[],
                patrones=[
                    # Palabras creativas
                    'imagina', 'inventa', 'crea', 'diseña',
                    'nuevo', 'original', 'único', 'diferente',
                    # Preguntas abiertas
                    'qué pasaría', 'y si', 'cómo sería',
                    'podrías', 'puedes', 'intenta',
                    # Metáforas
                    'como', 'parece', 'similar', 'metáfora',
                    # Emociones
                    'siente', 'piensa', 'quiere', 'desea',
                ],
                peso_base=0.9,
                peso_fondo=0.1,
            ),
        }
    
    def registrar_tokenizer(self, tokenizer, vocab_size: int):
        """
        Registra el tokenizer para mapear tokens a módulos.
        
        Debe llamarse después de cargar el tokenizer.
        """
        # Crear nuevo buffer con tamaño correcto
        token_a_modulo = torch.zeros(vocab_size, self.n_modulos)
        
        # Para cada módulo, encontrar los IDs de sus tokens clave
        for i, nombre in enumerate(self.nombres):
            if nombre not in self.llaves:
                continue
            
            llave = self.llaves[nombre]
            
            for patron in llave.patrones:
                try:
                    # Tokenizar el patrón
                    ids = tokenizer.encode(patron)
                    for token_id in ids:
                        if 0 <= token_id < vocab_size:
                            token_a_modulo[token_id, i] = llave.peso_base
                except:
                    pass
            
            # Peso de fondo para todos los tokens
            token_a_modulo[:, i] = torch.clamp(
                token_a_modulo[:, i], 
                min=llave.peso_fondo
            )
        
        # Normalizar por fila (cada token suma ~1 en pesos)
        sumas = token_a_modulo.sum(dim=1, keepdim=True)
        sumas = torch.clamp(sumas, min=1e-6)
        token_a_modulo = token_a_modulo / sumas
        
        # Actualizar buffer
        self.register_buffer('token_a_modulo', token_a_modulo)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula pesos de distribución para cada módulo.
        
        Args:
            x: Embeddings de entrada (batch, seq, dim)
            token_ids: IDs de tokens (batch, seq) para usar llaves
            
        Returns:
            pesos: (batch, seq, n_modulos) - pesos por posición y módulo
        """
        batch, seq, _ = x.shape
        
        # ============================================
        # 1. Pesos por LLAVES (reglas explícitas)
        # ============================================
        if token_ids is not None and self.token_a_modulo.shape[0] > 1:
            # Lookup directo: token_id -> pesos de módulos
            token_ids_clamped = token_ids.clamp(0, self.token_a_modulo.shape[0] - 1)
            pesos_llaves = self.token_a_modulo[token_ids_clamped]  # (batch, seq, n_modulos)
        else:
            # Sin token_ids, usar distribución uniforme
            pesos_llaves = torch.ones(
                batch, seq, self.n_modulos, 
                device=x.device
            ) / self.n_modulos
        
        # ============================================
        # 2. Pesos por ATENCIÓN (aprendidos)
        # ============================================
        x_norm = self.norm(x)
        pesos_atencion = F.softmax(self.query(x_norm), dim=-1)  # (batch, seq, n_modulos)
        
        # ============================================
        # 3. Combinar: LLAVES tienen prioridad
        # ============================================
        pesos = (
            self.peso_llaves * pesos_llaves + 
            (1 - self.peso_llaves) * pesos_atencion
        )
        
        # Normalizar para que sumen 1
        pesos = pesos / (pesos.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Estadísticas (para análisis)
        if self.training:
            self.activaciones_acumuladas += pesos.mean(dim=(0, 1)).detach()
            self.n_llamadas += 1
        
        return pesos
    
    def obtener_estadisticas(self) -> Dict[str, float]:
        """Obtiene estadísticas de activación de módulos."""
        if self.n_llamadas == 0:
            return {n: 0.0 for n in self.nombres}
        
        promedio = self.activaciones_acumuladas / self.n_llamadas
        return {
            nombre: promedio[i].item() 
            for i, nombre in enumerate(self.nombres)
        }
    
    def reset_estadisticas(self):
        """Reinicia las estadísticas de activación."""
        self.activaciones_acumuladas.zero_()
        self.n_llamadas.zero_()
