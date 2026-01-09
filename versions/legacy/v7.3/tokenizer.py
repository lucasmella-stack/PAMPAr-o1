# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Tokenizador Fractal LLARRI.

Tokenización jerárquica donde:
- Nivel 2: caracteres básicos (256 bytes)
- Nivel 4: pares de caracteres (4 tokens nivel 2 → 1 token nivel 4)
- Nivel 8: cuádruplos (4 tokens nivel 4 → 1 token nivel 8)
- ... hasta nivel 256

Los cuadrantes (4 elementos) se combinan para subir de nivel.
Cache guarda tokens frecuentes para lookup directo.

Author: Lucas Ricardo Mella Chillemi (Segunda Cabeza)
Coordinator: Alvaro (Segunda Cabeza)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


@dataclass
class TokenizerConfig:
    """Configuración del tokenizador fractal."""
    niveles: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256])
    vocab_base: int = 256  # Bytes (nivel 2)
    max_cache_size: int = 100000  # Tokens frecuentes cacheados
    min_freq_cache: int = 10  # Frecuencia mínima para cachear
    pad_token: int = 0
    unk_token: int = 1
    bos_token: int = 2  # Begin of sequence
    eos_token: int = 3  # End of sequence


class VocabNivel:
    """
    Vocabulario para un nivel específico.
    
    Cada token de nivel N está compuesto por 4 tokens de nivel N/2.
    """
    
    def __init__(self, nivel: int, nivel_inferior: Optional['VocabNivel'] = None):
        self.nivel = nivel
        self.nivel_inferior = nivel_inferior
        
        # Mapeos
        self.token_to_id: Dict[tuple, int] = {}
        self.id_to_token: Dict[int, tuple] = {}
        self.frecuencias: Dict[int, int] = defaultdict(int)
        
        # Para nivel 2 (base), inicializar con bytes
        if nivel == 2:
            for i in range(256):
                self.token_to_id[(i,)] = i
                self.id_to_token[i] = (i,)
        
        self.next_id = 256 if nivel == 2 else 0
    
    def add_token(self, componentes: tuple) -> int:
        """
        Agrega un token compuesto por 4 elementos del nivel inferior.
        
        Args:
            componentes: Tupla de 4 IDs del nivel inferior
            
        Returns:
            ID del nuevo token
        """
        if componentes in self.token_to_id:
            return self.token_to_id[componentes]
        
        new_id = self.next_id
        self.token_to_id[componentes] = new_id
        self.id_to_token[new_id] = componentes
        self.next_id += 1
        return new_id
    
    def get_id(self, componentes: tuple) -> Optional[int]:
        """Obtiene ID si existe, None si no."""
        return self.token_to_id.get(componentes)
    
    def get_componentes(self, token_id: int) -> Optional[tuple]:
        """Obtiene los componentes de un token."""
        return self.id_to_token.get(token_id)
    
    def incrementar_freq(self, token_id: int):
        """Incrementa la frecuencia de un token."""
        self.frecuencias[token_id] += 1
    
    def __len__(self):
        return len(self.token_to_id)


class TokenizadorFractal:
    """
    Tokenizador Fractal Jerárquico.
    
    Convierte texto en tokens organizados por niveles fractales:
    - Nivel 2: bytes individuales
    - Nivel 4: combinaciones de 4 bytes
    - Nivel 8: combinaciones de 4 tokens nivel 4
    - etc.
    
    La tokenización puede detenerse en cualquier nivel según
    si existe un token cacheado o hay que construirlo.
    
    Example:
        >>> tokenizer = TokenizadorFractal()
        >>> tokens = tokenizer.encode("Hola mundo")
        >>> print(tokens)
        {'nivel_2': [...], 'nivel_4': [...], ...}
        >>> texto = tokenizer.decode(tokens)
        >>> print(texto)
        "Hola mundo"
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        
        # Crear vocabularios por nivel
        self.vocabs: Dict[int, VocabNivel] = {}
        prev_vocab = None
        for nivel in self.config.niveles:
            self.vocabs[nivel] = VocabNivel(nivel, prev_vocab)
            prev_vocab = self.vocabs[nivel]
        
        # Cache de tokens frecuentes (nivel alto → embedding directo)
        self.cache_frecuentes: Dict[str, Dict[int, int]] = {}  # texto → {nivel: token_id}
        
        # Estadísticas
        self.stats = {
            'total_tokens': 0,
            'cache_hits': 0,
            'construidos': 0,
        }
    
    def _bytes_to_nivel2(self, texto: str) -> List[int]:
        """Convierte texto a tokens nivel 2 (bytes)."""
        return list(texto.encode('utf-8'))
    
    def _agrupar_en_cuadrantes(self, tokens: List[int], pad_value: int = 0) -> List[tuple]:
        """
        Agrupa tokens en cuadrantes de 4.
        
        Args:
            tokens: Lista de tokens
            pad_value: Valor de padding si no es múltiplo de 4
            
        Returns:
            Lista de tuplas de 4 elementos
        """
        # Padding para que sea múltiplo de 4
        while len(tokens) % 4 != 0:
            tokens.append(pad_value)
        
        cuadrantes = []
        for i in range(0, len(tokens), 4):
            cuadrante = tuple(tokens[i:i+4])
            cuadrantes.append(cuadrante)
        
        return cuadrantes
    
    def _construir_nivel(self, tokens_nivel_inferior: List[int], nivel: int) -> List[int]:
        """
        Construye tokens del nivel actual a partir del nivel inferior.
        
        4 tokens nivel N/2 → 1 token nivel N
        """
        vocab = self.vocabs[nivel]
        cuadrantes = self._agrupar_en_cuadrantes(tokens_nivel_inferior)
        
        tokens_nivel = []
        for cuadrante in cuadrantes:
            # Buscar si ya existe
            token_id = vocab.get_id(cuadrante)
            if token_id is None:
                # Crear nuevo token
                token_id = vocab.add_token(cuadrante)
            
            vocab.incrementar_freq(token_id)
            tokens_nivel.append(token_id)
        
        return tokens_nivel
    
    def encode(self, texto: str, max_nivel: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Tokeniza texto en todos los niveles fractales.
        
        Args:
            texto: Texto a tokenizar
            max_nivel: Nivel máximo a construir (None = todos)
            
        Returns:
            Dict con tokens por nivel: {2: [...], 4: [...], 8: [...], ...}
        """
        if max_nivel is None:
            max_nivel = self.config.niveles[-1]
        
        resultado = {}
        
        # Nivel 2: bytes
        tokens_actuales = self._bytes_to_nivel2(texto)
        resultado[2] = tokens_actuales.copy()
        
        # Construir niveles superiores
        for nivel in self.config.niveles[1:]:
            if nivel > max_nivel:
                break
            
            tokens_actuales = self._construir_nivel(tokens_actuales, nivel)
            resultado[nivel] = tokens_actuales.copy()
        
        self.stats['total_tokens'] += 1
        
        return resultado
    
    def encode_to_embeddings(
        self, 
        texto: str, 
        nivel_objetivo: int = 64,
        embed_dim: int = 256
    ) -> torch.Tensor:
        """
        Tokeniza y devuelve tensor listo para el modelo.
        
        Args:
            texto: Texto a tokenizar
            nivel_objetivo: Nivel de tokens a usar
            embed_dim: Dimensión de embedding
            
        Returns:
            Tensor de shape (1, seq_len, niveles_info)
        """
        tokens_por_nivel = self.encode(texto, max_nivel=nivel_objetivo)
        
        # Usar el nivel objetivo o el más alto disponible
        nivel_usar = min(nivel_objetivo, max(tokens_por_nivel.keys()))
        tokens = tokens_por_nivel[nivel_usar]
        
        # Crear tensor con información de todos los niveles
        # Cada token incluye: [token_id, nivel, componente1, comp2, comp3, comp4]
        seq_len = len(tokens)
        
        # Tensor básico con IDs
        tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        
        return tensor, tokens_por_nivel
    
    def decode_nivel(self, tokens: List[int], nivel: int) -> List[int]:
        """
        Descompone tokens de un nivel a nivel 2 (bytes).
        """
        if nivel == 2:
            return tokens
        
        vocab = self.vocabs[nivel]
        nivel_inferior = self.config.niveles[self.config.niveles.index(nivel) - 1]
        
        tokens_inferiores = []
        for token_id in tokens:
            componentes = vocab.get_componentes(token_id)
            if componentes:
                tokens_inferiores.extend(componentes)
            else:
                # Token desconocido
                tokens_inferiores.extend([self.config.unk_token] * 4)
        
        # Recursivamente bajar hasta nivel 2
        return self.decode_nivel(tokens_inferiores, nivel_inferior)
    
    def decode(self, tokens_por_nivel: Dict[int, List[int]], usar_nivel: int = 2) -> str:
        """
        Decodifica tokens a texto.
        
        Args:
            tokens_por_nivel: Dict con tokens por nivel
            usar_nivel: Nivel desde el cual decodificar (default=2, bytes directos)
            
        Returns:
            Texto decodificado
            
        Note:
            Por defecto usa nivel 2 (bytes) que es la representación exacta.
            Los niveles superiores son representaciones comprimidas que
            pueden perder información si no se almacenaron correctamente.
        """
        # Usar nivel 2 (bytes) para decodificación exacta
        if usar_nivel == 2 and 2 in tokens_por_nivel:
            bytes_list = tokens_por_nivel[2]
        else:
            # Usar nivel especificado y bajar a nivel 2
            if usar_nivel not in tokens_por_nivel:
                usar_nivel = min(tokens_por_nivel.keys())
            tokens = tokens_por_nivel[usar_nivel]
            bytes_list = self.decode_nivel(tokens, usar_nivel)
        
        # Filtrar padding y convertir a texto
        bytes_list = [b for b in bytes_list if b != self.config.pad_token and b < 256]
        
        try:
            return bytes(bytes_list).decode('utf-8')
        except:
            # Si falla UTF-8, intentar con errores ignorados
            return bytes(bytes_list).decode('utf-8', errors='ignore')
    
    def get_vocab_sizes(self) -> Dict[int, int]:
        """Retorna tamaño de vocabulario por nivel."""
        return {nivel: len(vocab) for nivel, vocab in self.vocabs.items()}
    
    def save(self, path: str):
        """Guarda el tokenizador."""
        data = {
            'config': {
                'niveles': self.config.niveles,
                'vocab_base': self.config.vocab_base,
                'max_cache_size': self.config.max_cache_size,
            },
            'vocabs': {},
            'stats': self.stats,
        }
        
        for nivel, vocab in self.vocabs.items():
            data['vocabs'][nivel] = {
                'token_to_id': {str(k): v for k, v in vocab.token_to_id.items()},
                'frecuencias': dict(vocab.frecuencias),
                'next_id': vocab.next_id,
            }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TokenizadorFractal':
        """Carga un tokenizador guardado."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = TokenizerConfig(**data['config'])
        tokenizer = cls(config)
        tokenizer.stats = data['stats']
        
        for nivel_str, vocab_data in data['vocabs'].items():
            nivel = int(nivel_str)
            vocab = tokenizer.vocabs[nivel]
            vocab.token_to_id = {eval(k): v for k, v in vocab_data['token_to_id'].items()}
            vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}
            vocab.frecuencias = defaultdict(int, vocab_data['frecuencias'])
            vocab.next_id = vocab_data['next_id']
        
        return tokenizer
    
    def print_stats(self):
        """Imprime estadísticas del tokenizador."""
        print("\n" + "="*60)
        print("TOKENIZADOR FRACTAL - ESTADÍSTICAS")
        print("="*60)
        print(f"Niveles: {self.config.niveles}")
        print(f"\nVocabulario por nivel:")
        for nivel, size in self.get_vocab_sizes().items():
            print(f"  Nivel {nivel:3d}: {size:,} tokens")
        print(f"\nTokenizaciones: {self.stats['total_tokens']:,}")
        print("="*60)


class EmbeddingComposicional(nn.Module):
    """
    Embedding Composicional Fractal.
    
    En lugar de tablas enormes por nivel, solo almacena:
    - 256 embeddings base (bytes)
    - MLPs pequeños para combinar 4→1 en cada nivel
    
    Memoria: ~600KB vs ~700MB de tablas tradicionales
    
    Funcionamiento:
    1. Nivel 2: lookup directo (256 bytes → 64 dims)
    2. Nivel 4: combina 4 embeddings nivel 2 → 128 dims
    3. Nivel 8: combina 4 embeddings nivel 4 → 256 dims
    ...y así sucesivamente
    """
    
    def __init__(
        self,
        tokenizer: TokenizadorFractal,
        base_dim: int = 64,
        max_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_dim = base_dim
        self.max_dim = max_dim
        self.niveles = tokenizer.config.niveles
        
        # Embedding base: solo 256 bytes
        self.emb_base = nn.Embedding(256, base_dim)
        
        # Dimensiones por nivel (crece hasta max_dim)
        self.dims = {}
        dim = base_dim
        for nivel in self.niveles:
            self.dims[nivel] = min(dim, max_dim)
            dim = min(dim * 2, max_dim)
        
        # Combinadores: 4 embeddings nivel N → 1 embedding nivel N*2
        self.combinadores = nn.ModuleDict()
        for i, nivel in enumerate(self.niveles[1:], 1):
            nivel_inferior = self.niveles[i-1]
            dim_in = self.dims[nivel_inferior] * 4  # 4 tokens concatenados
            dim_out = self.dims[nivel]
            
            self.combinadores[str(nivel)] = nn.Sequential(
                nn.Linear(dim_in, dim_out * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_out * 2, dim_out),
                nn.LayerNorm(dim_out)
            )
        
        # Proyección final (opcional, para uniformizar dimensión)
        self.proyecto_final = nn.Linear(max_dim, max_dim)
        
        # Cache para evitar recálculos
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _embed_nivel2(self, tokens: List[int]) -> torch.Tensor:
        """Embedding directo de bytes."""
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.emb_base.weight.device)
        tokens_tensor = tokens_tensor.clamp(0, 255)
        return self.emb_base(tokens_tensor)  # (seq_len, base_dim)
    
    def _combinar_cuadrante(self, embeddings: torch.Tensor, nivel: int) -> torch.Tensor:
        """
        Combina 4 embeddings consecutivos en 1.
        
        Args:
            embeddings: (seq_len, dim_inferior)
            nivel: Nivel destino
            
        Returns:
            (seq_len//4, dim_nivel)
        """
        seq_len = embeddings.size(0)
        dim = embeddings.size(1)
        
        # Padding si no es múltiplo de 4
        if seq_len % 4 != 0:
            pad_len = 4 - (seq_len % 4)
            padding = torch.zeros(pad_len, dim, device=embeddings.device)
            embeddings = torch.cat([embeddings, padding], dim=0)
            seq_len = embeddings.size(0)
        
        # Reshape: (seq_len, dim) → (seq_len//4, 4*dim)
        grupos = embeddings.view(seq_len // 4, 4 * dim)
        
        # Pasar por combinador
        combinador = self.combinadores[str(nivel)]
        return combinador(grupos)  # (seq_len//4, dim_nivel)
    
    def embed_tokens(
        self, 
        tokens_por_nivel: Dict[int, List[int]], 
        nivel_objetivo: int = 8
    ) -> torch.Tensor:
        """
        Genera embeddings composicionales hasta el nivel objetivo.
        
        Args:
            tokens_por_nivel: Output de tokenizer.encode()
            nivel_objetivo: Nivel de representación deseado
            
        Returns:
            Tensor (seq_len, embed_dim)
        """
        # Empezar desde nivel 2 (bytes)
        if 2 not in tokens_por_nivel:
            raise ValueError("Se requieren tokens de nivel 2 (bytes)")
        
        # Embedding base
        current_emb = self._embed_nivel2(tokens_por_nivel[2])
        current_nivel = 2
        
        # Subir niveles hasta el objetivo
        for nivel in self.niveles[1:]:
            if nivel > nivel_objetivo:
                break
            
            current_emb = self._combinar_cuadrante(current_emb, nivel)
            current_nivel = nivel
        
        # Si la dimensión no es max_dim, proyectar
        if current_emb.size(-1) != self.max_dim:
            # Pad o truncate
            if current_emb.size(-1) < self.max_dim:
                padding = torch.zeros(
                    current_emb.size(0), 
                    self.max_dim - current_emb.size(-1),
                    device=current_emb.device
                )
                current_emb = torch.cat([current_emb, padding], dim=-1)
            else:
                current_emb = current_emb[:, :self.max_dim]
        
        return self.proyecto_final(current_emb)
    
    def forward(
        self, 
        texto: str, 
        nivel_objetivo: int = 8
    ) -> torch.Tensor:
        """
        Embedding end-to-end: texto → tensor.
        
        Args:
            texto: Texto a embeder
            nivel_objetivo: Nivel de compresión
            
        Returns:
            Tensor (1, seq_len, max_dim)
        """
        # Tokenizar
        tokens = self.tokenizer.encode(texto, max_nivel=nivel_objetivo)
        
        # Generar embeddings
        emb = self.embed_tokens(tokens, nivel_objetivo)
        
        # Agregar batch dimension
        return emb.unsqueeze(0)
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria usada por el modelo."""
        memoria = {
            'emb_base': self.emb_base.weight.numel() * 4,  # bytes
            'combinadores': sum(
                sum(p.numel() for p in comb.parameters()) * 4
                for comb in self.combinadores.values()
            ),
            'proyecto_final': sum(p.numel() for p in self.proyecto_final.parameters()) * 4,
        }
        memoria['total'] = sum(memoria.values())
        return memoria
    
    def print_arquitectura(self):
        """Imprime detalles de la arquitectura."""
        print("\n" + "="*60)
        print("EMBEDDING COMPOSICIONAL FRACTAL")
        print("="*60)
        
        print(f"\nDimensiones por nivel:")
        for nivel, dim in self.dims.items():
            print(f"  Nivel {nivel:3d}: {dim} dims")
        
        print(f"\nCombinadores:")
        for nivel, comb in self.combinadores.items():
            params = sum(p.numel() for p in comb.parameters())
            print(f"  Nivel {nivel:3s}: {params:,} params")
        
        memoria = self.get_memoria_usada()
        print(f"\nMemoria total: {memoria['total'] / 1024:.1f} KB")
        print(f"  - Embedding base (256 bytes): {memoria['emb_base'] / 1024:.1f} KB")
        print(f"  - Combinadores: {memoria['combinadores'] / 1024:.1f} KB")
        print(f"  - Proyección final: {memoria['proyecto_final'] / 1024:.1f} KB")
        print("="*60)


class EmbeddingPosicionalFractal(nn.Module):
    """
    Embedding Posicional Fractal.
    
    Combina:
    1. Sinusoidal base (posición absoluta, 0 parámetros)
    2. Posición jerárquica por nivel (la estructura fractal ya la tiene)
    
    Para posición 7 en una secuencia:
    - pos_nivel2 = 7          (byte 7)
    - pos_nivel4 = 7 // 4 = 1 (bloque 1 de nivel 4)
    - pos_nivel8 = 7 // 16 = 0 (bloque 0 de nivel 8)
    
    El embedding combina todas estas escalas.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        max_seq_len: int = 8192,
        niveles: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.niveles = niveles or [2, 4, 8, 16, 32, 64, 128, 256]
        
        # 1. Sinusoidal base (posición absoluta) - 0 parámetros
        self.register_buffer('sinusoidal', self._crear_sinusoidal(max_seq_len, embed_dim))
        
        # 2. Embeddings aprendidos por nivel (pequeños, solo para bloques)
        # Nivel 4: max 2048 bloques (8192/4), Nivel 8: max 512 bloques, etc.
        self.pos_por_nivel = nn.ModuleDict()
        dim_por_nivel = embed_dim // len(self.niveles)  # Dividir dimensión entre niveles
        
        for nivel in self.niveles[1:]:  # Skip nivel 2 (usa sinusoidal puro)
            max_bloques = max_seq_len // nivel + 1
            # Embedding pequeño: pocos bloques, pocas dimensiones
            self.pos_por_nivel[str(nivel)] = nn.Embedding(
                max_bloques, 
                dim_por_nivel
            )
        
        # Combinador de escalas
        # Input: sinusoidal (embed_dim) + niveles (dim_por_nivel * (len-1))
        total_dim = embed_dim + dim_por_nivel * (len(self.niveles) - 1)
        self.combinar = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dim_por_nivel = dim_por_nivel
    
    def _crear_sinusoidal(self, max_len: int, dim: int) -> torch.Tensor:
        """
        Crea embeddings sinusoidales (fórmula del Transformer original).
        
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2] if dim % 2 else div_term)
        
        return pe
    
    def _calcular_posiciones_fractal(self, seq_len: int) -> Dict[int, torch.Tensor]:
        """
        Calcula posiciones por nivel para una secuencia.
        
        Para seq_len=16:
        - nivel 4: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
        - nivel 8: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
        """
        posiciones = {}
        pos_base = torch.arange(seq_len)
        
        for nivel in self.niveles[1:]:
            # Posición = en qué bloque de 'nivel' estamos
            pos_nivel = pos_base // nivel
            posiciones[nivel] = pos_nivel
        
        return posiciones
    
    def forward(
        self, 
        seq_len: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Genera embeddings posicionales fractales.
        
        Args:
            seq_len: Longitud de secuencia
            device: Dispositivo
            
        Returns:
            Tensor (seq_len, embed_dim)
        """
        if device is None:
            device = self.sinusoidal.device
        
        # 1. Sinusoidal base
        pos_sin = self.sinusoidal[:seq_len].to(device)  # (seq_len, embed_dim)
        
        # 2. Posiciones por nivel
        posiciones = self._calcular_posiciones_fractal(seq_len)
        
        embeddings_nivel = []
        for nivel in self.niveles[1:]:
            pos = posiciones[nivel].to(device)
            pos = pos.clamp(0, self.pos_por_nivel[str(nivel)].num_embeddings - 1)
            emb_nivel = self.pos_por_nivel[str(nivel)](pos)  # (seq_len, dim_por_nivel)
            embeddings_nivel.append(emb_nivel)
        
        # 3. Concatenar todo
        # sinusoidal + todos los niveles
        all_pos = torch.cat([pos_sin] + embeddings_nivel, dim=-1)  # (seq_len, total_dim)
        
        # 4. Combinar a dimensión final
        return self.combinar(all_pos)  # (seq_len, embed_dim)
    
    def forward_batch(
        self, 
        batch_size: int,
        seq_len: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Genera embeddings posicionales para un batch.
        
        Returns:
            Tensor (batch_size, seq_len, embed_dim)
        """
        pos_emb = self.forward(seq_len, device)  # (seq_len, embed_dim)
        return pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, embed_dim)
    
    def get_memoria_usada(self) -> Dict[str, int]:
        """Calcula memoria usada."""
        memoria = {
            'sinusoidal': self.sinusoidal.numel() * 4,
            'pos_por_nivel': sum(
                emb.weight.numel() * 4 
                for emb in self.pos_por_nivel.values()
            ),
            'combinar': sum(p.numel() for p in self.combinar.parameters()) * 4,
        }
        memoria['total'] = sum(memoria.values())
        return memoria
    
    def print_arquitectura(self):
        """Imprime detalles de la arquitectura."""
        print("\n" + "="*60)
        print("EMBEDDING POSICIONAL FRACTAL")
        print("="*60)
        
        print(f"\nDimensión: {self.embed_dim}")
        print(f"Max secuencia: {self.max_seq_len}")
        print(f"Niveles: {self.niveles}")
        print(f"Dims por nivel: {self.dim_por_nivel}")
        
        print(f"\nEmbeddings por nivel:")
        for nivel, emb in self.pos_por_nivel.items():
            print(f"  Nivel {nivel:3s}: {emb.num_embeddings} posiciones × {emb.embedding_dim} dims")
        
        memoria = self.get_memoria_usada()
        print(f"\nMemoria total: {memoria['total'] / 1024:.1f} KB")
        print(f"  - Sinusoidal (buffer): {memoria['sinusoidal'] / 1024:.1f} KB")
        print(f"  - Pos por nivel: {memoria['pos_por_nivel'] / 1024:.1f} KB")
        print(f"  - Combinador: {memoria['combinar'] / 1024:.1f} KB")
        print("="*60)


# Alias para compatibilidad
EmbeddingFractal = EmbeddingComposicional


# Test rápido
if __name__ == "__main__":
    print("="*60)
    print("TEST TOKENIZADOR FRACTAL")
    print("="*60)
    
    # Crear tokenizador
    tokenizer = TokenizadorFractal()
    
    # Textos de prueba
    textos = [
        "Hola",
        "Hola mundo",
        "Hello world",
        "La inteligencia artificial es fascinante",
        "LLARRI-O1 es un modelo fractal",
    ]
    
    for texto in textos:
        print(f"\n{'─'*50}")
        print(f"Texto: '{texto}'")
        
        # Tokenizar
        tokens = tokenizer.encode(texto)
        
        print(f"\nTokens por nivel:")
        for nivel, toks in sorted(tokens.items()):
            print(f"  Nivel {nivel:3d}: {len(toks):3d} tokens → {toks[:10]}{'...' if len(toks) > 10 else ''}")
        
        # Decodificar
        decoded = tokenizer.decode(tokens)
        print(f"\nDecodificado: '{decoded}'")
        print(f"Match: {'✓' if decoded == texto else '✗'}")
    
    # Estadísticas
    tokenizer.print_stats()
    
    # ========================================
    # TEST EMBEDDING COMPOSICIONAL
    # ========================================
    print("\n" + "="*60)
    print("TEST EMBEDDING COMPOSICIONAL")
    print("="*60)
    
    # Crear embedding
    embedding = EmbeddingComposicional(
        tokenizer=tokenizer,
        base_dim=64,
        max_dim=256
    )
    
    # Mostrar arquitectura
    embedding.print_arquitectura()
    
    # Test de embedding
    print("\nTest de embeddings:")
    for texto in textos[:3]:
        emb = embedding(texto, nivel_objetivo=8)
        print(f"  '{texto[:20]}...' → shape {tuple(emb.shape)}")
    
    # Comparar con tabla tradicional
    print("\n" + "-"*40)
    print("COMPARACIÓN DE MEMORIA:")
    print("-"*40)
    
    # Nuestro modelo composicional
    mem_comp = embedding.get_memoria_usada()['total']
    
    # Tabla tradicional equivalente (estimación)
    vocab_sizes = {4: 10000, 8: 50000, 16: 100000, 32: 200000}
    mem_tabla = sum(vs * 256 * 4 for vs in vocab_sizes.values())
    
    print(f"Composicional: {mem_comp / 1024:.1f} KB")
    print(f"Tabla tradicional: {mem_tabla / 1024 / 1024:.1f} MB")
    print(f"Ahorro: {mem_tabla / mem_comp:.0f}x menos memoria")
    
    # ========================================
    # TEST EMBEDDING POSICIONAL FRACTAL
    # ========================================
    print("\n" + "="*60)
    print("TEST EMBEDDING POSICIONAL FRACTAL")
    print("="*60)
    
    # Crear embedding posicional
    pos_embedding = EmbeddingPosicionalFractal(
        embed_dim=256,
        max_seq_len=1024
    )
    
    # Mostrar arquitectura
    pos_embedding.print_arquitectura()
    
    # Test de posiciones
    print("\nTest de embeddings posicionales:")
    for seq_len in [4, 16, 64, 256]:
        pos_emb = pos_embedding(seq_len)
        print(f"  seq_len={seq_len:3d} → shape {tuple(pos_emb.shape)}")
    
    # Mostrar cómo las posiciones se mapean por nivel
    print("\n" + "-"*40)
    print("POSICIONES FRACTALES (ejemplo seq_len=16):")
    print("-"*40)
    posiciones = pos_embedding._calcular_posiciones_fractal(16)
    print("Posición byte:  ", list(range(16)))
    for nivel, pos in sorted(posiciones.items()):
        print(f"Posición nivel {nivel:3d}:", pos.tolist())
    
    # Test combinado: token embedding + posicional
    print("\n" + "-"*40)
    print("TEST COMBINADO: Token + Posicional")
    print("-"*40)
    
    texto_test = "Hola mundo"
    tokens = tokenizer.encode(texto_test)
    seq_len = len(tokens[2])  # Longitud en nivel 2
    
    # Embedding de tokens
    token_emb = embedding.embed_tokens(tokens, nivel_objetivo=2)  # Sin comprimir para test
    
    # Embedding posicional
    pos_emb = pos_embedding(seq_len)
    
    # Combinar (como haría un transformer)
    combined = token_emb + pos_emb
    
    print(f"Texto: '{texto_test}'")
    print(f"Token embedding: {tuple(token_emb.shape)}")
    print(f"Pos embedding:   {tuple(pos_emb.shape)}")
    print(f"Combinado:       {tuple(combined.shape)}")
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)
