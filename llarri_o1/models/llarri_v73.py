# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7.3 - Sistema de Liderazgo Dinámico entre Módulos Cerebrales

Arquitectura:
- 6 módulos especializados que trabajan en EQUIPO
- Un módulo LIDERA según el tipo de tarea
- Los otros se ACOPLAN al líder
- El Tálamo coordina quién lidera

Esto simula cómo funcionan las áreas cerebrales:
- Una área domina según la tarea (lenguaje para hablar, visual para ver)
- Las otras se sincronizan y apoyan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ModuloAcoplable(nn.Module):
    """
    Módulo cerebral que puede actuar como LÍDER o SEGUIDOR.
    
    Como líder: procesa independientemente
    Como seguidor: recibe señal del líder y se acopla
    """
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.15, nombre: str = ""):
        super().__init__()
        self.nombre = nombre
        self.dim = dim
        
        # === Procesamiento propio ===
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # FFN reducida (2x en vez de 4x para ahorrar memoria)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # === Sistema de acoplamiento ===
        # Proyección de la señal del líder
        self.proyeccion_lider = nn.Linear(dim, dim)
        
        # Gate adaptativo: cuánto seguir al líder vs procesar independiente
        self.gate_acople = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        señal_lider: Optional[torch.Tensor] = None,
        intensidad_acople: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Procesa el input, opcionalmente acoplándose a un líder.
        
        Args:
            x: Input [batch, seq, dim]
            mask: Máscara causal
            señal_lider: Output del módulo líder (si existe)
            intensidad_acople: Cuánto seguir al líder (0-1)
            
        Returns:
            output: [batch, seq, dim]
            stats: Estadísticas del procesamiento
        """
        # === Procesamiento propio ===
        # Atención
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        h = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        
        stats = {'procesamiento_propio': 1.0}
        
        # === Acoplamiento al líder (si hay uno) ===
        if señal_lider is not None:
            # Proyectar señal del líder
            lider_proj = self.proyeccion_lider(señal_lider)
            
            # Calcular gate: cuánto de mi salida vs cuánto del líder
            gate_input = torch.cat([h, lider_proj], dim=-1)
            gate = self.gate_acople(gate_input)  # [batch, seq, 1]
            
            # Aplicar acoplamiento modulado por intensidad
            acople_efectivo = gate * intensidad_acople
            h = h * (1 - acople_efectivo) + lider_proj * acople_efectivo
            
            stats['acoplamiento'] = acople_efectivo.mean().item()
            stats['gate_mean'] = gate.mean().item()
        
        return h, stats


class TalamoLiderazgo(nn.Module):
    """
    Tálamo que decide qué módulo LIDERA y coordina el equipo.
    
    Funciones:
    1. Detectar tipo de tarea (lingüística, lógica, etc.)
    2. Seleccionar módulo líder
    3. Calcular intensidad de acoplamiento para seguidores
    """
    
    NOMBRES_MODULOS = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
    
    def __init__(self, dim: int, n_modulos: int = 6, actividad_basal: float = 0.2):
        super().__init__()
        self.n_modulos = n_modulos
        self.actividad_basal = actividad_basal
        
        # Detector de tipo de tarea → qué módulo debería liderar
        self.detector_tarea = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, n_modulos),
        )
        
        # Calculador de intensidad de acoplamiento
        self.calc_acoplamiento = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decide el liderazgo y coordinación.
        
        Args:
            x: Input embeddings [batch, seq, dim]
            
        Returns:
            Dict con:
                - liderazgo: probabilidades de ser líder [batch, n_modulos]
                - lider_idx: índice del líder [batch]
                - participacion: cuánto participa cada módulo [batch, n_modulos]
                - intensidad_acople: cuánto deben acoplarse los seguidores [batch, 1]
        """
        # Usar representación global (promedio de secuencia)
        h_global = x.mean(dim=1)  # [batch, dim]
        
        # 1. Scores de liderazgo
        scores = self.detector_tarea(h_global)  # [batch, n_modulos]
        
        # 2. Softmax con temperatura baja = decisión más clara sobre líder
        liderazgo = F.softmax(scores * 3.0, dim=-1)  # Temperatura 1/3
        
        # 3. Índice del líder (argmax)
        lider_idx = liderazgo.argmax(dim=-1)  # [batch]
        
        # 4. Participación de cada módulo (todos participan algo)
        # El líder tiene peso máximo, los demás según su relevancia
        participacion = F.softmax(scores, dim=-1)
        participacion = participacion * (1 - self.actividad_basal) + self.actividad_basal
        
        # 5. Intensidad de acoplamiento
        intensidad_acople = self.calc_acoplamiento(h_global)  # [batch, 1]
        
        return {
            'liderazgo': liderazgo,
            'lider_idx': lider_idx,
            'participacion': participacion,
            'intensidad_acople': intensidad_acople,
            'scores_raw': scores,
        }


class IntegradorCerebral(nn.Module):
    """
    Integra las salidas de todos los módulos ponderadas por participación.
    """
    
    def __init__(self, dim: int, n_modulos: int = 6):
        super().__init__()
        
        # Proyección final
        self.proyeccion = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # Gate de confianza (si hay mucho consenso, más confianza)
        self.gate_confianza = nn.Linear(n_modulos, 1)
        
    def forward(
        self, 
        salidas: Dict[str, torch.Tensor],
        participacion: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Integra salidas de módulos.
        
        Args:
            salidas: Dict nombre_modulo -> output [batch, seq, dim]
            participacion: Pesos de participación [batch, n_modulos]
            
        Returns:
            integrado: [batch, seq, dim]
            stats: Estadísticas
        """
        nombres = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
        
        # Combinar salidas ponderadas
        batch, seq_len, dim = list(salidas.values())[0].shape
        integrado = torch.zeros(batch, seq_len, dim, device=participacion.device)
        
        for i, nombre in enumerate(nombres):
            peso = participacion[:, i].view(-1, 1, 1)  # [batch, 1, 1]
            integrado = integrado + salidas[nombre] * peso
        
        # Proyección y normalización
        integrado = self.norm(self.proyeccion(integrado))
        
        # Calcular consenso (varianza entre módulos)
        outputs_stack = torch.stack([salidas[n] for n in nombres], dim=0)  # [6, batch, seq, dim]
        varianza = outputs_stack.var(dim=0).mean()  # Menor varianza = más consenso
        consenso = 1.0 / (1.0 + varianza)
        
        stats = {
            'consenso': consenso.item(),
            'participacion_max': participacion.max(dim=-1)[0].mean().item(),
            'participacion_min': participacion.min(dim=-1)[0].mean().item(),
        }
        
        return integrado, stats


class LLARRIv73(nn.Module):
    """
    LLARRI v7.3 - Modelo con Sistema de Liderazgo Dinámico
    
    Arquitectura:
    1. Embeddings (token + posición)
    2. Tálamo decide qué módulo lidera
    3. Líder procesa primero
    4. Seguidores se acoplan al líder
    5. Integrador combina todo
    6. Output layer
    """
    
    VERSION = "7.3"
    NOMBRES_MODULOS = ['lenguaje', 'logica', 'matematicas', 'patrones', 'contexto', 'creatividad']
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        dropout: float = 0.15,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        
        # === Embeddings ===
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # === Tálamo (coordinador) ===
        self.talamo = TalamoLiderazgo(dim, n_modulos=6)
        
        # === 6 Módulos Cerebrales ===
        self.modulos = nn.ModuleDict({
            nombre: ModuloAcoplable(dim, n_heads, dropout, nombre)
            for nombre in self.NOMBRES_MODULOS
        })
        
        # === Integrador ===
        self.integrador = IntegradorCerebral(dim, n_modulos=6)
        
        # === Output ===
        self.norm_final = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        
        # Para estadísticas
        self._ultimo_lider = None
        self._ultimas_stats = {}
        
        # Inicialización
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización de pesos estilo GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con sistema de liderazgo.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            targets: Target IDs para loss [batch, seq_len]
            
        Returns:
            Dict con logits, loss, y estadísticas
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # === 1. Embeddings ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embedding(input_ids) + self.pos_embedding(positions)
        h = self.dropout(h)
        
        # === 2. Máscara causal ===
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # === 3. Tálamo decide liderazgo ===
        talamo_out = self.talamo(h)
        lider_idx = talamo_out['lider_idx']
        participacion = talamo_out['participacion']
        intensidad_acople = talamo_out['intensidad_acople']
        
        # Obtener nombre del líder (moda del batch)
        lider_comun = lider_idx.mode().values.item()
        nombre_lider = self.NOMBRES_MODULOS[lider_comun]
        self._ultimo_lider = nombre_lider
        
        # === 4. Líder procesa primero ===
        salida_lider, stats_lider = self.modulos[nombre_lider](
            h, mask=mask, señal_lider=None, intensidad_acople=0.0
        )
        
        # === 5. Seguidores se acoplan al líder ===
        salidas = {nombre_lider: salida_lider}
        stats_modulos = {nombre_lider: stats_lider}
        
        for nombre in self.NOMBRES_MODULOS:
            if nombre != nombre_lider:
                salida, stats = self.modulos[nombre](
                    h, 
                    mask=mask, 
                    señal_lider=salida_lider,
                    intensidad_acople=intensidad_acople.mean().item(),
                )
                salidas[nombre] = salida
                stats_modulos[nombre] = stats
        
        # === 6. Integrar ===
        h_integrado, stats_integracion = self.integrador(salidas, participacion)
        
        # === 7. Output ===
        h_final = self.norm_final(h_integrado)
        logits = self.output(h_final)
        
        # === 8. Loss (si hay targets) ===
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
        
        # Compilar estadísticas
        self._ultimas_stats = {
            'lider': nombre_lider,
            'liderazgo': talamo_out['liderazgo'],
            'participacion': participacion,
            'consenso': stats_integracion['consenso'],
            'intensidad_acople': intensidad_acople.mean().item(),
        }
        
        return {
            'logits': logits,
            'loss': loss,
            'lider': nombre_lider,
            'stats': self._ultimas_stats,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Genera texto con sampling.
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Limitar contexto
            context = generated[:, -256:]
            
            with torch.no_grad():
                output = self(context)
                logits = output['logits'][:, -1, :] / temperature
                
                # Repetition penalty
                for i in range(generated.shape[0]):
                    for token_id in generated[i].unique():
                        logits[i, token_id] /= repetition_penalty
                
                # Top-k
                if top_k > 0:
                    indices_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_remove] = float('-inf')
                
                # Top-p (nucleus sampling)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_remove = cumulative_probs > top_p
                sorted_indices_remove[:, 1:] = sorted_indices_remove[:, :-1].clone()
                sorted_indices_remove[:, 0] = 0
                
                for i in range(logits.shape[0]):
                    indices_remove = sorted_indices[i, sorted_indices_remove[i]]
                    logits[i, indices_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_stats(self) -> Dict:
        """Retorna las últimas estadísticas del forward."""
        return self._ultimas_stats
    
    def get_ultimo_lider(self) -> str:
        """Retorna el nombre del último módulo líder."""
        return self._ultimo_lider
