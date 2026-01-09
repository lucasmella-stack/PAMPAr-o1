# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Segunda Cabeza
"""
Hipocampo - Memoria del Sistema

En el cerebro real:
- Consolida memorias de corto a largo plazo
- Recupera memorias relevantes según contexto
- Asocia experiencias similares

En LLARRI v7:
- Almacena representaciones de experiencias
- Recupera memorias similares al input actual
- Usa hashing para búsqueda eficiente O(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Hipocampo(nn.Module):
    """
    Memoria episódica con recuperación eficiente.
    
    Usa Locality Sensitive Hashing (LSH) para
    encontrar memorias similares en O(1) promedio
    en vez de O(n) con búsqueda exhaustiva.
    """
    
    def __init__(
        self,
        dim: int,
        capacidad: int = 10000,
        n_hashes: int = 4,
        n_buckets: int = 64,
        k_memorias: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.capacidad = capacidad
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.k_memorias = k_memorias
        
        # Memorias almacenadas (no son parámetros, son buffer)
        self.register_buffer('memorias', torch.zeros(capacidad, dim))
        self.register_buffer('memorias_validas', torch.zeros(capacidad, dtype=torch.bool))
        self.register_buffer('indice_escritura', torch.tensor(0))
        
        # Proyecciones para LSH (fijas, no aprendibles)
        self.register_buffer(
            'hash_proyecciones',
            torch.randn(n_hashes, dim) / (dim ** 0.5)
        )
        
        # Proyector para queries (aprendible)
        self.proyector_query = nn.Linear(dim, dim)
        
        # Proyector para integrar memorias recuperadas
        self.integrador = nn.Linear(dim * 2, dim)
        
    def _hash(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula hash LSH para un batch de vectores."""
        # x: (batch, dim) o (batch, seq, dim)
        shape_original = x.shape
        if len(shape_original) == 3:
            x = x.reshape(-1, self.dim)
            
        # Proyectar
        proj = x @ self.hash_proyecciones.T  # (batch, n_hashes)
        
        # Convertir a índice de bucket
        # Cuantizar a n_buckets valores
        bucket_idx = (proj > 0).int()
        # Convertir binario a decimal
        powers = 2 ** torch.arange(self.n_hashes, device=x.device)
        hash_val = (bucket_idx * powers).sum(dim=-1)  # (batch,)
        
        if len(shape_original) == 3:
            hash_val = hash_val.reshape(shape_original[0], shape_original[1])
            
        return hash_val % self.n_buckets
    
    def memorizar(self, x: torch.Tensor):
        """
        Guarda nuevas experiencias en la memoria.
        Usa escritura circular cuando se llena.
        
        Args:
            x: (batch, dim) experiencias a memorizar
        """
        if not self.training:
            return
            
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            idx = self.indice_escritura.item()
            self.memorias[idx] = x[i].detach()
            self.memorias_validas[idx] = True
            self.indice_escritura = (self.indice_escritura + 1) % self.capacidad
    
    def recordar(self, query: torch.Tensor) -> torch.Tensor:
        """
        Recupera memorias relevantes para el query.
        
        Args:
            query: (batch, dim) o (batch, seq, dim)
            
        Returns:
            memorias_relevantes: mismo shape que query
        """
        shape_original = query.shape
        query_flat = query.reshape(-1, self.dim)
        
        # Si no hay memorias válidas, retornar zeros
        if not self.memorias_validas.any():
            return torch.zeros_like(query)
        
        # Proyectar query
        query_proj = self.proyector_query(query_flat)
        
        # Obtener memorias válidas
        memorias_activas = self.memorias[self.memorias_validas]
        
        if len(memorias_activas) == 0:
            return torch.zeros_like(query)
        
        # Similitud (más eficiente que LSH completo para memorias pequeñas)
        # Para memorias grandes, usar LSH
        if len(memorias_activas) < 1000:
            # Búsqueda exhaustiva (suficientemente rápida)
            similitud = query_proj @ memorias_activas.T  # (batch*seq, n_memorias)
            
            # Top-k memorias
            k = min(self.k_memorias, len(memorias_activas))
            top_k = similitud.topk(k, dim=-1)
            
            # Recuperar y promediar
            memorias_recuperadas = memorias_activas[top_k.indices]  # (batch*seq, k, dim)
            pesos = F.softmax(top_k.values, dim=-1)
            resultado = (memorias_recuperadas * pesos.unsqueeze(-1)).sum(dim=1)
            
        else:
            # LSH para memoria grande
            resultado = self._recordar_lsh(query_proj, memorias_activas)
        
        return resultado.reshape(shape_original)
    
    def _recordar_lsh(
        self, 
        query: torch.Tensor, 
        memorias: torch.Tensor
    ) -> torch.Tensor:
        """Recuperación con LSH para memorias grandes."""
        # Hash del query
        hash_query = self._hash(query)  # (batch*seq,)
        
        # Hash de memorias
        hash_memorias = self._hash(memorias)  # (n_memorias,)
        
        # Para cada query, encontrar memorias en el mismo bucket
        resultados = []
        for i in range(len(query)):
            bucket = hash_query[i].item()
            en_bucket = (hash_memorias == bucket)
            
            if en_bucket.any():
                candidatos = memorias[en_bucket]
                # Similitud solo con candidatos
                sim = query[i] @ candidatos.T
                k = min(self.k_memorias, len(candidatos))
                top_k = sim.topk(k)
                
                mem_k = candidatos[top_k.indices]
                pesos = F.softmax(top_k.values, dim=-1)
                resultado = (mem_k * pesos.unsqueeze(-1)).sum(dim=0)
            else:
                # Si bucket vacío, usar vecino más cercano global
                sim = query[i] @ memorias.T
                idx = sim.argmax()
                resultado = memorias[idx]
                
            resultados.append(resultado)
            
        return torch.stack(resultados)
    
    def forward(
        self, 
        x: torch.Tensor,
        memorizar: bool = True,
    ) -> torch.Tensor:
        """
        Procesa input enriqueciendo con memorias relevantes.
        
        Args:
            x: (batch, seq, dim) input
            memorizar: si guardar esta experiencia
            
        Returns:
            x enriquecido con memorias
        """
        # Recuperar memorias relevantes
        memorias = self.recordar(x)
        
        # Integrar memorias con input
        combinado = torch.cat([x, memorias], dim=-1)
        x_enriquecido = self.integrador(combinado)
        
        # Guardar experiencia (solo el resumen)
        if memorizar and self.training:
            resumen = x.mean(dim=1)  # (batch, dim)
            self.memorizar(resumen)
        
        return x_enriquecido
    
    def reset_memorias(self):
        """Borra todas las memorias."""
        self.memorias.zero_()
        self.memorias_validas.zero_()
        self.indice_escritura.zero_()


class HipocampoExterno(nn.Module):
    """
    Versión que usa almacenamiento externo (FAISS).
    Para memorias muy grandes que no caben en GPU.
    
    Requiere: pip install faiss-cpu (o faiss-gpu)
    """
    
    def __init__(
        self,
        dim: int,
        k_memorias: int = 5,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.k_memorias = k_memorias
        
        # Proyector para queries
        self.proyector_query = nn.Linear(dim, dim)
        self.integrador = nn.Linear(dim * 2, dim)
        
        # Index FAISS (lazy initialization)
        self._index = None
        self._use_gpu = use_gpu
        
    def _init_index(self):
        """Inicializa el índice FAISS."""
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product
            if self._use_gpu:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
        except ImportError:
            print("FAISS no instalado, usando Hipocampo regular")
            self._index = None
    
    def memorizar(self, x: torch.Tensor):
        """Añade memorias al índice."""
        if self._index is None:
            self._init_index()
        if self._index is None:
            return
            
        x_np = x.detach().cpu().numpy()
        self._index.add(x_np)
    
    def recordar(self, query: torch.Tensor) -> torch.Tensor:
        """Recupera memorias del índice."""
        if self._index is None or self._index.ntotal == 0:
            return torch.zeros_like(query)
            
        q_np = query.detach().cpu().numpy()
        D, I = self._index.search(q_np, self.k_memorias)
        
        # Reconstruir memorias
        memorias = []
        for indices in I:
            mems = [self._index.reconstruct(int(i)) for i in indices if i >= 0]
            if mems:
                memorias.append(torch.tensor(mems).mean(dim=0))
            else:
                memorias.append(torch.zeros(self.dim))
                
        return torch.stack(memorias).to(query.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.proyector_query(x.mean(dim=1))
        memorias = self.recordar(query).unsqueeze(1).expand(-1, x.shape[1], -1)
        combinado = torch.cat([x, memorias], dim=-1)
        return self.integrador(combinado)
