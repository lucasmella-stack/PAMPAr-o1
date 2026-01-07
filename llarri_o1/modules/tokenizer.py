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


class EmbeddingFractal(nn.Module):
    """
    Capa de embedding que usa la estructura fractal.
    
    Combina embeddings de diferentes niveles para crear
    representaciones ricas que capturan estructura jerárquica.
    """
    
    def __init__(
        self,
        tokenizer: TokenizadorFractal,
        embed_dim: int = 256,
        niveles_usar: Optional[List[int]] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.niveles = niveles_usar or [2, 4, 8, 16]
        
        # Embedding por nivel (tamaño dinámico)
        self.embeddings = nn.ModuleDict()
        for nivel in self.niveles:
            # Estimar vocab size (256 base + espacio para nuevos)
            vocab_size = max(512, len(tokenizer.vocabs[nivel]) * 2)
            self.embeddings[str(nivel)] = nn.Embedding(vocab_size, embed_dim)
        
        # Combinador de niveles
        self.combinar = nn.Linear(embed_dim * len(self.niveles), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens_por_nivel: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Combina embeddings de múltiples niveles.
        
        Args:
            tokens_por_nivel: Dict de tensores por nivel
            
        Returns:
            Tensor combinado (batch, seq_len, embed_dim)
        """
        embeds = []
        
        for nivel in self.niveles:
            if nivel in tokens_por_nivel:
                tokens = tokens_por_nivel[nivel]
                # Asegurar que no exceda vocab size
                tokens = tokens.clamp(0, self.embeddings[str(nivel)].num_embeddings - 1)
                embed = self.embeddings[str(nivel)](tokens)
                embeds.append(embed)
        
        if not embeds:
            raise ValueError("No hay tokens para ningún nivel")
        
        # Si diferentes longitudes, usar la más larga y repetir/truncar
        max_len = max(e.size(1) for e in embeds)
        
        aligned_embeds = []
        for embed in embeds:
            if embed.size(1) < max_len:
                # Repetir último token
                padding = embed[:, -1:, :].repeat(1, max_len - embed.size(1), 1)
                embed = torch.cat([embed, padding], dim=1)
            elif embed.size(1) > max_len:
                embed = embed[:, :max_len, :]
            aligned_embeds.append(embed)
        
        # Concatenar y combinar
        combined = torch.cat(aligned_embeds, dim=-1)
        output = self.combinar(combined)
        return self.norm(output)


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
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)
