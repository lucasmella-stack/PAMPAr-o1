"""
language_model.py - Modelo de Lenguaje para Post-procesamiento OCR

Implementa múltiples técnicas para mejorar la precisión del OCR:

1. Cadenas de Markov (Character-level N-grams)
   - Calcula probabilidad de secuencias de caracteres
   - Detecta combinaciones imposibles en español

2. Reglas Fonéticas del Español
   - Q siempre seguida de U
   - Ñ casi siempre seguida de vocal
   - Combinaciones imposibles (qo, ñr, etc.)

3. Corrector Ortográfico
   - Diccionario español
   - Distancia de Levenshtein para sugerencias

4. Re-ranking de Candidatos
   - Combina scores de OCR + LM para elegir mejor opción

Uso:
    from llarri.inference.language_model import SpanishLanguageModel
    
    lm = SpanishLanguageModel()
    
    # Corregir texto
    corrected = lm.correct("Holla mundo")  # → "Hola mundo"
    
    # Re-rankear candidatos de beam search
    best = lm.rerank_candidates(["Holla", "Hola", "Holia"])  # → "Hola"
    
    # Score de una secuencia
    score = lm.score_text("Hola mundo")  # → 0.95

Referencias:
- Jurafsky & Martin - Speech and Language Processing (Cap. 3: N-grams)
- Kneser-Ney Smoothing para n-gramas
- Peter Norvig - How to Write a Spelling Corrector
"""

from __future__ import annotations

import re
import math
from collections import defaultdict
from typing import Optional, Union, List, Tuple
from functools import lru_cache
from pathlib import Path

# Intentar importar dependencias opcionales
try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False
    SpellChecker = None

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    Levenshtein = None

# Wordfreq - corpus de frecuencias de palabras (Wikipedia, Twitter, etc.)
try:
    from wordfreq import zipf_frequency, word_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False
    zipf_frequency = None
    word_frequency = None

# SymSpellPy - corrector ortográfico muy rápido
try:
    from symspellpy import SymSpell, Verbosity
    HAS_SYMSPELL = True
except ImportError:
    HAS_SYMSPELL = False
    SymSpell = None


class SpanishCharNgrams:
    """
    Modelo de N-gramas a nivel de caracteres para español.
    
    Implementa Cadenas de Markov donde cada estado es una secuencia
    de N-1 caracteres y la transición es el siguiente caracter.
    
    P(c_n | c_1, c_2, ..., c_{n-1}) ≈ P(c_n | c_{n-N+1}, ..., c_{n-1})
    """
    
    # N-gramas pre-calculados de español (frecuencias relativas)
    # Basado en análisis de corpus de Wikipedia en español
    COMMON_BIGRAMS = {
        # Vocales después de consonantes comunes
        'qu': 0.99, 'gu': 0.7, 'cu': 0.3, 'bu': 0.3,
        'ca': 0.4, 'co': 0.4, 'ce': 0.3, 'ci': 0.3,
        'pa': 0.35, 'pe': 0.25, 'pi': 0.15, 'po': 0.2, 'pu': 0.05,
        'ta': 0.3, 'te': 0.3, 'ti': 0.2, 'to': 0.15, 'tu': 0.05,
        'ma': 0.3, 'me': 0.25, 'mi': 0.2, 'mo': 0.15, 'mu': 0.1,
        'na': 0.25, 'ne': 0.2, 'ni': 0.2, 'no': 0.25, 'nu': 0.1,
        'la': 0.3, 'le': 0.25, 'li': 0.15, 'lo': 0.2, 'lu': 0.1,
        'ra': 0.25, 're': 0.3, 'ri': 0.2, 'ro': 0.15, 'ru': 0.1,
        'sa': 0.25, 'se': 0.3, 'si': 0.2, 'so': 0.15, 'su': 0.1,
        'da': 0.25, 'de': 0.35, 'di': 0.2, 'do': 0.15, 'du': 0.05,
        
        # Dígrafos españoles
        'll': 0.15, 'rr': 0.1, 'ch': 0.1, 'ñ ': 0.05,
        'lla': 0.3, 'lle': 0.25, 'lli': 0.15, 'llo': 0.25, 'llu': 0.05,
        
        # Combinaciones vocálicas comunes (diptongos)
        'ia': 0.2, 'ie': 0.25, 'io': 0.2, 'iu': 0.05,
        'ua': 0.2, 'ue': 0.3, 'ui': 0.15, 'uo': 0.05,
        'ai': 0.15, 'ei': 0.2, 'oi': 0.1, 'au': 0.1, 'eu': 0.1,
        
        # Finales de palabra comunes
        'ar': 0.3, 'er': 0.25, 'ir': 0.2, 'or': 0.15,
        'as': 0.25, 'es': 0.35, 'is': 0.1, 'os': 0.2, 'us': 0.1,
        'an': 0.2, 'en': 0.3, 'in': 0.15, 'on': 0.2, 'un': 0.15,
        'al': 0.2, 'el': 0.25, 'il': 0.1, 'ol': 0.1, 'ul': 0.05,
        
        # Prefijos comunes
        'de': 0.4, 'en': 0.35, 'es': 0.3, 'un': 0.25, 'el': 0.3,
        'la': 0.35, 'lo': 0.25, 'le': 0.2, 'se': 0.3, 'no': 0.25,
        'co': 0.25, 'pre': 0.2, 'pro': 0.2, 'con': 0.25,
    }
    
    COMMON_TRIGRAMS = {
        # Trigramas más frecuentes en español
        'que': 0.4, 'ent': 0.3, 'con': 0.3, 'ion': 0.25,
        'los': 0.3, 'las': 0.3, 'del': 0.25, 'est': 0.25,
        'aci': 0.2, 'ado': 0.25, 'ido': 0.2, 'ara': 0.2,
        'era': 0.2, 'ero': 0.15, 'ien': 0.2, 'nte': 0.25,
        'par': 0.2, 'por': 0.25, 'tra': 0.15, 'pre': 0.15,
        'pro': 0.15, 'com': 0.2, 'men': 0.15, 'cia': 0.2,
        'ter': 0.15, 'per': 0.15, 'der': 0.1, 'ver': 0.1,
    }
    
    # Combinaciones imposibles o muy raras en español
    IMPOSSIBLE_SEQUENCES = frozenset([
        # Q sin U
        'qa', 'qe', 'qi', 'qo', 'qr', 'qs', 'qt', 'qn', 'qm', 'ql',
        # Consonantes dobles que no existen
        'bb', 'dd', 'ff', 'gg', 'jj', 'kk', 'mm', 'nn', 'pp', 'ss', 'tt', 'vv', 'ww', 'xx', 'yy', 'zz',
        # Excepto: cc (acceso), ll (llamar), rr (carro)
        # LL seguida de consonantes (excepto y: lly no es común pero existe)
        'llb', 'llc', 'lld', 'llf', 'llg', 'llh', 'llj', 'llk', 'lll', 'llm', 
        'lln', 'llp', 'llq', 'llr', 'lls', 'llt', 'llv', 'llw', 'llx', 'llz',
        # Ñ seguida de consonantes duras
        'ñb', 'ñc', 'ñd', 'ñf', 'ñg', 'ñj', 'ñk', 'ñl', 'ñm', 'ñn', 'ñp', 'ñr', 'ñs', 'ñt', 'ñv', 'ñx', 'ñz',
        # H en posiciones raras
        'hh', 'hn', 'hm', 'hl', 'hr', 'hs', 'ht',
        # Otras combinaciones raras
        'xq', 'zq', 'zx', 'xz', 'vq', 'wq', 'kq',
        'aaa', 'eee', 'iii', 'ooo', 'uuu',  # Triples vocales
    ])
    
    # Probabilidad por defecto para n-gramas no vistos (smoothing)
    DEFAULT_PROBABILITY = 0.001
    
    def __init__(self, n: int = 3):
        """
        Inicializa el modelo de n-gramas.
        
        Args:
            n: Orden del modelo (2=bigramas, 3=trigramas)
        """
        self.n = n
        self._cache = {}
    
    @lru_cache(maxsize=10000)
    def get_ngram_probability(self, ngram: str) -> float:
        """
        Obtiene la probabilidad de un n-grama.
        
        Usa las tablas pre-calculadas y aplica smoothing para
        n-gramas no vistos.
        """
        ngram = ngram.lower()
        
        # Verificar si es secuencia imposible
        if ngram in self.IMPOSSIBLE_SEQUENCES:
            return 0.0001  # Muy baja pero no cero
        
        # Buscar en trigramas
        if len(ngram) >= 3 and ngram[:3] in self.COMMON_TRIGRAMS:
            return self.COMMON_TRIGRAMS[ngram[:3]]
        
        # Buscar en bigramas
        if len(ngram) >= 2 and ngram[:2] in self.COMMON_BIGRAMS:
            return self.COMMON_BIGRAMS[ngram[:2]]
        
        # Smoothing: probabilidad basada en caracteres individuales
        return self.DEFAULT_PROBABILITY
    
    def score_sequence(self, text: str) -> float:
        """
        Calcula el log-probability de una secuencia de texto.
        
        Usa el modelo de Markov:
        P(text) = Π P(c_i | c_{i-n+1}...c_{i-1})
        
        Retorna log-probability para evitar underflow.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Log-probability (más alto = más probable)
        """
        if not text:
            return 0.0
        
        text = text.lower()
        log_prob = 0.0
        
        # Evaluar cada n-grama
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            prob = self.get_ngram_probability(ngram)
            log_prob += math.log(prob + 1e-10)  # Evitar log(0)
        
        # Normalizar por longitud
        num_ngrams = max(1, len(text) - self.n + 1)
        return log_prob / num_ngrams
    
    def has_impossible_sequence(self, text: str) -> bool:
        """Verifica si el texto contiene secuencias imposibles."""
        text = text.lower()
        for seq in self.IMPOSSIBLE_SEQUENCES:
            if seq in text:
                return True
        return False
    
    def find_impossible_sequences(self, text: str) -> list[tuple[int, str]]:
        """
        Encuentra todas las secuencias imposibles en el texto.
        
        Returns:
            Lista de (posición, secuencia)
        """
        text = text.lower()
        found = []
        for seq in self.IMPOSSIBLE_SEQUENCES:
            start = 0
            while True:
                pos = text.find(seq, start)
                if pos == -1:
                    break
                found.append((pos, seq))
                start = pos + 1
        return sorted(found, key=lambda x: x[0])


class SpanishPhoneticRules:
    """
    Reglas fonéticas específicas del español.
    
    Implementa restricciones fonológicas que pueden usarse para
    validar y corregir predicciones OCR.
    """
    
    VOWELS = set('aeiouáéíóúü')
    CONSONANTS = set('bcdfghjklmnñpqrstvwxyz')
    
    # Reglas de transición: (contexto) -> {siguiente: probabilidad}
    TRANSITION_RULES = {
        # Q siempre seguida de U
        'q': {'u': 0.99},
        
        # GU antes de E/I (güe, güi con diéresis)
        'gu': {'e': 0.4, 'i': 0.3, 'a': 0.2, 'o': 0.1},
        
        # Ñ seguida de vocal
        'ñ': {'a': 0.25, 'e': 0.25, 'i': 0.2, 'o': 0.25, 'u': 0.05},
        
        # LL seguida de vocal
        'll': {'a': 0.3, 'e': 0.25, 'i': 0.15, 'o': 0.25, 'u': 0.05},
        
        # RR solo entre vocales (no al inicio)
        'rr': {'a': 0.25, 'e': 0.25, 'i': 0.2, 'o': 0.25, 'u': 0.05},
        
        # CH seguida de vocal
        'ch': {'a': 0.3, 'e': 0.25, 'i': 0.2, 'o': 0.2, 'u': 0.05},
    }
    
    @classmethod
    def validate_word(cls, word: str) -> tuple[bool, list[str]]:
        """
        Valida una palabra según reglas fonéticas.
        
        Returns:
            (es_válida, lista_de_errores)
        """
        word = word.lower()
        errors = []
        
        # Regla 1: Q debe ir seguida de U
        for i, char in enumerate(word):
            if char == 'q':
                if i + 1 >= len(word) or word[i + 1] != 'u':
                    errors.append(f"Q sin U en posición {i}")
        
        # Regla 2: No puede haber más de 2 vocales seguidas sin ser diptongo válido
        vowel_count = 0
        for i, char in enumerate(word):
            if char in cls.VOWELS:
                vowel_count += 1
                if vowel_count > 2:
                    errors.append(f"Más de 2 vocales seguidas en posición {i-2}")
            else:
                vowel_count = 0
        
        # Regla 3: RR no puede estar al inicio ni al final
        if word.startswith('rr'):
            errors.append("RR al inicio de palabra")
        if word.endswith('rr'):
            errors.append("RR al final de palabra")
        
        # Regla 4: No puede haber más de 2 consonantes seguidas al inicio
        consonant_count = 0
        for char in word:
            if char in cls.CONSONANTS:
                consonant_count += 1
            else:
                break
        if consonant_count > 2:
            # Excepciones: str, spr, etc.
            if not word[:3] in ['str', 'spr', 'spl']:
                errors.append(f"Más de 2 consonantes al inicio: {word[:consonant_count]}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def suggest_correction(cls, word: str) -> Optional[str]:
        """
        Sugiere corrección basada en reglas fonéticas.
        
        Solo corrige errores obvios basados en reglas.
        """
        word_lower = word.lower()
        corrected = list(word_lower)
        made_change = False
        
        # Corregir Q sin U -> insertar U
        i = 0
        while i < len(corrected):
            if corrected[i] == 'q' and i + 1 < len(corrected):
                next_char = corrected[i + 1]
                if next_char != 'u':
                    # qi, qe -> qui, que (insertar u)
                    if next_char in 'aeiou':
                        corrected.insert(i + 1, 'u')
                        made_change = True
                        i += 1  # Saltar la u insertada
            i += 1
        
        result = ''.join(corrected)
        
        # Preservar capitalización original
        if word and word[0].isupper():
            result = result[0].upper() + result[1:]
        
        return result if made_change else None


class SpanishWordFreq:
    """
    Modelo de frecuencia de palabras usando wordfreq.
    
    Wordfreq proporciona frecuencias de palabras basadas en:
    - Wikipedia en español
    - Twitter
    - Subtítulos de películas
    - OpenSubtitles
    - Google Books
    
    Esto da un corpus de millones de palabras con sus frecuencias reales.
    """
    
    LANGUAGE = 'es'  # Español
    
    # Cache para evitar llamadas repetidas
    _word_cache: dict = {}
    
    @classmethod
    def is_available(cls) -> bool:
        """Verifica si wordfreq está disponible."""
        return HAS_WORDFREQ
    
    @classmethod
    @lru_cache(maxsize=50000)
    def get_frequency(cls, word: str) -> float:
        """
        Obtiene la frecuencia de una palabra en escala Zipf.
        
        Escala Zipf:
        - 7-8: Palabras extremadamente comunes (de, la, que, el)
        - 5-6: Palabras muy comunes (casa, mundo, hacer)
        - 4-5: Palabras comunes (teléfono, dirección)
        - 3-4: Palabras menos comunes (quirófano, efervescente)
        - 1-2: Palabras raras
        - 0: Palabra no encontrada
        
        Args:
            word: Palabra a buscar
            
        Returns:
            Frecuencia en escala Zipf (0-8)
        """
        if not HAS_WORDFREQ:
            return 0.0
        
        return zipf_frequency(word.lower(), cls.LANGUAGE)
    
    @classmethod
    def score_word(cls, word: str) -> float:
        """
        Convierte frecuencia Zipf a score 0-1.
        
        Args:
            word: Palabra a evaluar
            
        Returns:
            Score normalizado (0 = desconocida, 1 = muy común)
        """
        freq = cls.get_frequency(word)
        # Normalizar: Zipf 0-8 → Score 0-1
        return min(1.0, freq / 8.0)
    
    @classmethod
    def score_text(cls, text: str) -> float:
        """
        Calcula score promedio de todas las palabras.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Score promedio (0-1)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        scores = [cls.score_word(w) for w in words]
        return sum(scores) / len(scores)
    
    @classmethod
    def is_valid_word(cls, word: str, min_frequency: float = 2.0) -> bool:
        """
        Verifica si una palabra existe en el corpus español.
        
        Args:
            word: Palabra a verificar
            min_frequency: Frecuencia Zipf mínima para considerar válida
            
        Returns:
            True si la palabra es válida
        """
        return cls.get_frequency(word) >= min_frequency
    
    @classmethod
    def compare_candidates(cls, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Compara múltiples candidatos y los ordena por frecuencia.
        
        Args:
            candidates: Lista de palabras candidatas
            
        Returns:
            Lista de tuplas (palabra, frecuencia) ordenada
        """
        scored = [(c, cls.get_frequency(c)) for c in candidates]
        return sorted(scored, key=lambda x: -x[1])
    
    @classmethod
    def suggest_correction(cls, word: str, max_distance: int = 2) -> Optional[str]:
        """
        Sugiere la corrección más probable basada en frecuencia.
        
        Genera candidatos con distancia de Levenshtein <= max_distance
        y retorna el más frecuente.
        
        Args:
            word: Palabra a corregir
            max_distance: Distancia máxima de edición
            
        Returns:
            Palabra corregida o None si no hay sugerencia
        """
        if cls.is_valid_word(word):
            return word  # Ya es válida
        
        word_lower = word.lower()
        best_candidate = None
        best_freq = 0.0
        
        # Generar candidatos con edits simples
        candidates = cls._generate_candidates(word_lower, max_distance)
        
        for candidate in candidates:
            freq = cls.get_frequency(candidate)
            if freq > best_freq:
                best_freq = freq
                best_candidate = candidate
        
        if best_candidate and best_freq > 0:
            # Preservar capitalización
            if word and word[0].isupper():
                return best_candidate[0].upper() + best_candidate[1:]
            return best_candidate
        
        return None
    
    @classmethod
    def _generate_candidates(cls, word: str, max_distance: int = 1) -> set:
        """
        Genera candidatos con edits simples (Norvig's algorithm).
        
        Operaciones:
        - Eliminación de un caracter
        - Transposición de caracteres adyacentes
        - Reemplazo de un caracter
        - Inserción de un caracter
        """
        letters = 'abcdefghijklmnñopqrstuvwxyzáéíóúü'
        
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        # Distancia 1
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        candidates = set(deletes + transposes + replaces + inserts)
        
        # Distancia 2 (si se pide)
        if max_distance >= 2:
            for c1 in list(candidates)[:500]:  # Limitar para performance
                splits2 = [(c1[:i], c1[i:]) for i in range(len(c1) + 1)]
                deletes2 = [L + R[1:] for L, R in splits2 if R]
                transposes2 = [L + R[1] + R[0] + R[2:] for L, R in splits2 if len(R) > 1]
                replaces2 = [L + c + R[1:] for L, R in splits2 if R for c in letters]
                inserts2 = [L + c + R for L, R in splits2 for c in letters]
                candidates.update(deletes2 + transposes2 + replaces2 + inserts2)
        
        return candidates


class SpanishSymSpell:
    """
    Corrector ortográfico usando SymSpellPy.
    
    SymSpell es 1000x más rápido que otros correctores porque
    pre-calcula todas las eliminaciones posibles.
    
    Combina:
    - Diccionario de frecuencias español
    - Algoritmo de edición muy eficiente
    """
    
    _instance: Optional['SpanishSymSpell'] = None
    _symspell: Optional[SymSpell] = None
    
    def __new__(cls):
        """Singleton para no recargar el diccionario."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inicializa SymSpell con diccionario español."""
        if not HAS_SYMSPELL:
            self._symspell = None
            return
        
        self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Cargar diccionario interno de palabras comunes españolas
        self._load_spanish_dictionary()
    
    def _load_spanish_dictionary(self):
        """Carga diccionario de palabras españolas con frecuencias."""
        if not self._symspell:
            return
        
        # Palabras muy comunes con frecuencias aproximadas
        # Frecuencia: veces por millón de palabras
        spanish_words = {
            # Artículos y determinantes (frecuencia muy alta)
            'de': 50000, 'la': 40000, 'que': 38000, 'el': 35000, 'en': 32000,
            'y': 30000, 'a': 28000, 'los': 25000, 'del': 20000, 'se': 18000,
            'las': 17000, 'por': 16000, 'un': 15000, 'para': 14000, 'con': 13000,
            'no': 12000, 'una': 11000, 'su': 10000, 'al': 9500, 'lo': 9000,
            'como': 8500, 'más': 8000, 'pero': 7500, 'sus': 7000, 'le': 6500,
            'ya': 6000, 'o': 5500, 'este': 5000, 'sí': 4800, 'porque': 4500,
            'esta': 4300, 'entre': 4100, 'cuando': 4000, 'muy': 3800, 'sin': 3600,
            'sobre': 3400, 'también': 3200, 'me': 3000, 'hasta': 2900, 'hay': 2800,
            'donde': 2700, 'quien': 2600, 'desde': 2500, 'todo': 2400, 'nos': 2300,
            'durante': 2200, 'todos': 2100, 'uno': 2000, 'les': 1900, 'ni': 1800,
            'contra': 1700, 'otros': 1600, 'ese': 1500, 'eso': 1450, 'ante': 1400,
            'ellos': 1350, 'e': 1300, 'esto': 1250, 'mí': 1200, 'antes': 1150,
            'algunos': 1100, 'qué': 1050, 'unos': 1000, 'yo': 980, 'otro': 960,
            'otras': 940, 'otra': 920, 'él': 900, 'tanto': 880, 'esa': 860,
            'estos': 840, 'mucho': 820, 'quienes': 800, 'nada': 780, 'muchos': 760,
            'cual': 740, 'poco': 720, 'ella': 700, 'estar': 680, 'estas': 660,
            'algunas': 640, 'algo': 620, 'nosotros': 600, 'mi': 580, 'mis': 560,
            'tú': 540, 'te': 520, 'ti': 500, 'tu': 480, 'tus': 460,
            
            # Verbos comunes conjugados
            'es': 25000, 'ha': 12000, 'son': 8000, 'tiene': 7000, 'fue': 6500,
            'ser': 6000, 'hace': 5500, 'era': 5000, 'sido': 4800, 'puede': 4600,
            'está': 4400, 'hecho': 4200, 'han': 4000, 'tiene': 3800, 'había': 3600,
            'hacer': 3400, 'van': 3200, 'estar': 3000, 'tienen': 2900, 'sea': 2800,
            'haber': 2700, 'siendo': 2600, 'fueron': 2500, 'tener': 2400, 'poder': 2300,
            'decir': 2200, 'ir': 2100, 'ver': 2000, 'dar': 1900, 'saber': 1800,
            'querer': 1700, 'llegar': 1600, 'pasar': 1500, 'deber': 1400, 'poner': 1300,
            'venir': 1200, 'seguir': 1100, 'encontrar': 1000, 'llamar': 950, 'llevar': 900,
            'dejar': 850, 'sentir': 800, 'parecer': 750, 'quedar': 700, 'creer': 650,
            'hablar': 600, 'pensar': 550, 'salir': 500, 'conocer': 480, 'comer': 460,
            'vivir': 440, 'escribir': 420, 'leer': 400, 'abrir': 380, 'cerrar': 360,
            'trabajar': 340, 'estudiar': 320, 'jugar': 300, 'comprar': 280, 'vender': 260,
            'soy': 5000, 'eres': 2000, 'somos': 1500, 'estoy': 3000, 'estás': 1500,
            'estamos': 1200, 'están': 2500, 'tengo': 3500, 'tienes': 2000, 'tenemos': 1500,
            'hago': 2000, 'haces': 800, 'hacemos': 600, 'hacen': 1200, 'puedo': 2500,
            'puedes': 1500, 'podemos': 1200, 'pueden': 1800, 'digo': 1500, 'dices': 600,
            'dice': 3500, 'decimos': 400, 'dicen': 1200, 'voy': 2500, 'vas': 1200,
            'va': 4000, 'vamos': 2000, 'quiero': 2500, 'quieres': 1200, 'quiere': 2200,
            'queremos': 800, 'quieren': 1000,
            
            # Sustantivos comunes
            'año': 8000, 'años': 7500, 'tiempo': 6000, 'vez': 5500, 'vida': 5000,
            'parte': 4500, 'mundo': 4200, 'caso': 4000, 'día': 3800, 'país': 3600,
            'lugar': 3400, 'casa': 3200, 'hombre': 3000, 'cosa': 2900, 'trabajo': 2800,
            'momento': 2700, 'forma': 2600, 'gobierno': 2500, 'nombre': 2400, 'punto': 2300,
            'estado': 2200, 'persona': 2100, 'tipo': 2000, 'fin': 1900, 'grupo': 1800,
            'problema': 1700, 'mujer': 1650, 'mano': 1600, 'hecho': 1550, 'ciudad': 1500,
            'número': 1450, 'historia': 1400, 'lado': 1350, 'agua': 1300, 'ejemplo': 1250,
            'familia': 1200, 'cuerpo': 1150, 'derecho': 1100, 'razón': 1050, 'palabra': 1000,
            'hijo': 950, 'cuenta': 900, 'padre': 850, 'madre': 800, 'tierra': 750,
            'empresa': 700, 'dinero': 680, 'proceso': 660, 'proyecto': 640, 'clase': 620,
            'sistema': 600, 'sociedad': 580, 'gente': 560, 'idea': 540, 'ley': 520,
            'fuerza': 500, 'nivel': 480, 'desarrollo': 460, 'poder': 440, 'cambio': 420,
            'muerte': 400, 'noche': 380, 'manera': 360, 'orden': 340, 'sentido': 320,
            'centro': 300, 'guerra': 290, 'medio': 280, 'libro': 270, 'calle': 260,
            'hora': 250, 'relación': 240, 'información': 230, 'situación': 220, 'decisión': 210,
            
            # Adjetivos comunes
            'primer': 5000, 'primera': 4800, 'primero': 4600, 'nuevo': 4000, 'nueva': 3800,
            'gran': 3600, 'grande': 3400, 'mayor': 3200, 'mejor': 3000, 'mismo': 2800,
            'misma': 2600, 'último': 2400, 'última': 2200, 'bueno': 2000, 'buena': 1900,
            'cierto': 1800, 'alto': 1700, 'alta': 1600, 'largo': 1500, 'pequeño': 1400,
            'pequeña': 1300, 'antiguo': 1200, 'joven': 1100, 'viejo': 1000, 'malo': 950,
            'mala': 900, 'diferente': 850, 'importante': 800, 'posible': 750, 'único': 700,
            
            # Adverbios comunes
            'más': 15000, 'ya': 8000, 'también': 5000, 'muy': 4500, 'bien': 4000,
            'así': 3500, 'solo': 3200, 'siempre': 3000, 'después': 2800, 'ahora': 2600,
            'entonces': 2400, 'aquí': 2200, 'nunca': 2000, 'menos': 1800, 'antes': 1600,
            'todavía': 1400, 'hoy': 1300, 'casi': 1200, 'luego': 1100, 'aún': 1000,
            
            # Números
            'uno': 3000, 'dos': 2800, 'tres': 2500, 'cuatro': 2000, 'cinco': 1800,
            'seis': 1500, 'siete': 1300, 'ocho': 1100, 'nueve': 900, 'diez': 1200,
            'cien': 800, 'mil': 1500, 'millón': 700, 'millones': 800,
            
            # Palabras de documentos/formularios (importantes para OCR)
            'nombre': 2500, 'apellido': 800, 'dirección': 1200, 'teléfono': 1000,
            'fecha': 1500, 'firma': 600, 'documento': 800, 'número': 1500,
            'cédula': 400, 'identificación': 500, 'nacimiento': 600, 'edad': 700,
            'sexo': 400, 'estado': 2000, 'civil': 500, 'casado': 300, 'soltero': 250,
            'profesión': 400, 'ocupación': 350, 'domicilio': 300, 'ciudad': 1500,
            'país': 3600, 'código': 600, 'postal': 400, 'correo': 500, 'email': 300,
            'total': 1200, 'subtotal': 200, 'precio': 800, 'cantidad': 700,
            'producto': 600, 'servicio': 700, 'factura': 400, 'recibo': 300,
            'pago': 500, 'efectivo': 300, 'tarjeta': 400, 'crédito': 350,
            'banco': 500, 'cuenta': 900, 'monto': 400, 'importe': 300,
            
            # Apellidos comunes (importantes para OCR de documentos)
            'garcía': 500, 'rodríguez': 450, 'martínez': 420, 'lópez': 400,
            'gonzález': 380, 'hernández': 360, 'pérez': 340, 'sánchez': 320,
            'ramírez': 300, 'torres': 280, 'flores': 260, 'rivera': 240,
            'gómez': 220, 'díaz': 200, 'cruz': 190, 'morales': 180,
            'reyes': 170, 'gutiérrez': 160, 'ortiz': 150, 'ramos': 140,
            'jiménez': 130, 'ruiz': 120, 'moreno': 110, 'álvarez': 100,
            'romero': 95, 'alonso': 90, 'navarro': 85, 'domínguez': 80,
            
            # Meses
            'enero': 600, 'febrero': 500, 'marzo': 550, 'abril': 480,
            'mayo': 520, 'junio': 450, 'julio': 500, 'agosto': 420,
            'septiembre': 400, 'octubre': 450, 'noviembre': 380, 'diciembre': 500,
            
            # Días
            'lunes': 400, 'martes': 350, 'miércoles': 300, 'jueves': 320,
            'viernes': 380, 'sábado': 400, 'domingo': 450,
        }
        
        # Cargar cada palabra en SymSpell
        for word, freq in spanish_words.items():
            self._symspell.create_dictionary_entry(word, freq)
        
        # Si wordfreq está disponible, agregar más palabras
        if HAS_WORDFREQ:
            self._load_from_wordfreq()
    
    def _load_from_wordfreq(self):
        """Carga palabras adicionales desde wordfreq."""
        if not HAS_WORDFREQ or not self._symspell:
            return
        
        from wordfreq import top_n_list, zipf_frequency
        
        # Obtener top 10000 palabras del español
        try:
            top_words = top_n_list('es', 10000)
            for word in top_words:
                # Convertir Zipf a frecuencia aproximada
                freq = int(10 ** zipf_frequency(word, 'es'))
                if freq > 0:
                    self._symspell.create_dictionary_entry(word, freq)
        except Exception:
            pass  # Silenciar errores de carga
    
    def correct(self, word: str) -> str:
        """
        Corrige una palabra usando SymSpell.
        
        Args:
            word: Palabra a corregir
            
        Returns:
            Palabra corregida
        """
        if not self._symspell:
            return word
        
        # SymSpell funciona mejor en minúsculas
        is_capitalized = word and word[0].isupper()
        word_lower = word.lower()
        
        # Buscar sugerencias
        suggestions = self._symspell.lookup(
            word_lower,
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True
        )
        
        if suggestions:
            result = suggestions[0].term
            # Preservar capitalización
            if is_capitalized:
                result = result[0].upper() + result[1:]
            return result
        
        return word
    
    def correct_text(self, text: str) -> str:
        """
        Corrige un texto completo.
        
        Args:
            text: Texto a corregir
            
        Returns:
            Texto corregido
        """
        if not self._symspell:
            return text
        
        # Usar lookup_compound para frases
        suggestions = self._symspell.lookup_compound(
            text.lower(),
            max_edit_distance=2
        )
        
        if suggestions:
            return suggestions[0].term
        
        return text


class SpanishSpellChecker:
    """
    Corrector ortográfico para español.
    
    Combina:
    - Diccionario español (via pyspellchecker)
    - Distancia de Levenshtein para sugerencias
    - Heurísticas específicas del español
    """
    
    # Palabras muy comunes en español que deben estar en el diccionario
    COMMON_SPANISH_WORDS = {
        # Artículos y determinantes
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
        'aquel', 'aquella', 'aquellos', 'aquellas', 'del', 'al',
        # Pronombres
        'yo', 'tú', 'tu', 'él', 'ella', 'nosotros', 'ustedes', 'ellos', 'ellas',
        'me', 'te', 'se', 'nos', 'lo', 'le', 'les', 'mi', 'mis', 'su', 'sus',
        'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra',
        # Preposiciones
        'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'en', 'entre',
        'hacia', 'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras',
        # Conjunciones
        'y', 'e', 'ni', 'o', 'u', 'pero', 'sino', 'que', 'si', 'como', 'cuando',
        'porque', 'aunque', 'mientras', 'donde', 'cual', 'quien', 'cuyo',
        # Verbos muy comunes - infinitivos
        'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'decir', 'ir',
        'ver', 'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner',
        'venir', 'seguir', 'encontrar', 'llamar', 'llevar', 'dejar', 'sentir',
        'parecer', 'quedar', 'creer', 'hablar', 'pensar', 'salir', 'conocer',
        'comer', 'vivir', 'escribir', 'leer', 'abrir', 'cerrar', 'trabajar',
        # Verbos - presente indicativo
        'soy', 'eres', 'es', 'somos', 'son',
        'estoy', 'estás', 'está', 'estamos', 'están',
        'he', 'has', 'ha', 'hemos', 'han', 'hay',
        'tengo', 'tienes', 'tiene', 'tenemos', 'tienen',
        'hago', 'haces', 'hace', 'hacemos', 'hacen',
        'puedo', 'puedes', 'puede', 'podemos', 'pueden',
        'digo', 'dices', 'dice', 'decimos', 'dicen',
        'voy', 'vas', 'va', 'vamos', 'van',
        'veo', 'ves', 've', 'vemos', 'ven',
        'doy', 'das', 'da', 'damos', 'dan',
        'sé', 'sabes', 'sabe', 'sabemos', 'saben',
        'quiero', 'quieres', 'quiere', 'queremos', 'quieren',
        'vengo', 'vienes', 'viene', 'venimos', 'vienen',
        'como', 'comes', 'come', 'comemos', 'comen',
        'vivo', 'vives', 'vive', 'vivimos', 'viven',
        # Verbos - pasado
        'fue', 'fueron', 'era', 'eran', 'sido', 'siendo',
        'estuvo', 'estuvieron', 'estaba', 'estaban',
        'tuvo', 'tuvieron', 'tenía', 'tenían',
        'hizo', 'hicieron', 'hacía', 'hacían',
        'pudo', 'pudieron', 'podía', 'podían',
        'dijo', 'dijeron', 'decía', 'decían',
        'vino', 'vinieron', 'venía', 'venían',
        'dio', 'dieron', 'daba', 'daban',
        # Verbos - otros tiempos
        'haré', 'harás', 'hará', 'haremos', 'harán',
        'seré', 'serás', 'será', 'seremos', 'serán',
        'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendrán',
        'hecho', 'dicho', 'visto', 'puesto', 'escrito', 'abierto',
        # Adjetivos comunes
        'bueno', 'buena', 'buenos', 'buenas', 'malo', 'mala', 'malos', 'malas',
        'grande', 'grandes', 'pequeño', 'pequeña', 'pequeños', 'pequeñas',
        'nuevo', 'nueva', 'nuevos', 'nuevas', 'viejo', 'vieja', 'viejos', 'viejas',
        'primero', 'primera', 'último', 'última', 'segundo', 'segunda',
        'mismo', 'misma', 'mismos', 'mismas', 'otro', 'otra', 'otros', 'otras',
        'todo', 'toda', 'todos', 'todas', 'cada', 'alguno', 'alguna', 'ninguno',
        'largo', 'larga', 'corto', 'corta', 'alto', 'alta', 'bajo', 'baja',
        # Adverbios comunes
        'bien', 'mal', 'muy', 'mucho', 'mucha', 'muchos', 'muchas',
        'poco', 'poca', 'pocos', 'pocas', 'más', 'menos', 'tan', 'tanto',
        'aquí', 'ahí', 'allí', 'acá', 'allá', 'ahora', 'hoy', 'ayer', 'mañana',
        'siempre', 'nunca', 'también', 'tampoco', 'ya', 'todavía', 'aún',
        'solo', 'sólo', 'solamente', 'casi', 'apenas', 'incluso', 'además',
        'luego', 'después', 'antes', 'pronto', 'tarde', 'temprano',
        'cerca', 'lejos', 'dentro', 'fuera', 'arriba', 'abajo', 'adelante',
        # Sustantivos comunes
        'hombre', 'mujer', 'niño', 'niña', 'persona', 'gente', 'hijo', 'hija',
        'padre', 'madre', 'hermano', 'hermana', 'familia', 'amigo', 'amiga',
        'cosa', 'tiempo', 'año', 'mes', 'semana', 'día', 'hora', 'minuto',
        'vez', 'parte', 'mundo', 'vida', 'forma', 'manera', 'modo',
        'casa', 'país', 'ciudad', 'calle', 'lugar', 'sitio', 'punto',
        'trabajo', 'empresa', 'gobierno', 'estado', 'ley', 'derecho',
        'nombre', 'número', 'cuenta', 'caso', 'ejemplo', 'idea', 'problema',
        'monto', 'total', 'precio', 'pago', 'valor', 'cantidad', 'dinero',
        'fecha', 'firma', 'dirección', 'teléfono', 'correo', 'documento',
        'agua', 'tierra', 'aire', 'fuego', 'luz', 'sol', 'luna',
        'mano', 'ojo', 'ojos', 'cabeza', 'cuerpo', 'pie', 'pies',
        # Palabras de documentos/formularios
        'factura', 'recibo', 'contrato', 'certificado', 'formulario',
        'cliente', 'proveedor', 'producto', 'servicio', 'pedido',
        'rut', 'dni', 'cédula', 'pasaporte', 'licencia', 'carnet',
        'banco', 'cuenta', 'transferencia', 'depósito', 'cheque',
        'empresa', 'compañía', 'sociedad', 'limitada', 'anónima',
        # Números escritos
        'cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete',
        'ocho', 'nueve', 'diez', 'once', 'doce', 'trece', 'catorce', 'quince',
        'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve',
        'veinte', 'veintiuno', 'treinta', 'cuarenta', 'cincuenta',
        'sesenta', 'setenta', 'ochenta', 'noventa', 'cien', 'ciento',
        'doscientos', 'trescientos', 'quinientos', 'mil', 'millón', 'millones',
        # Saludos y formalidades
        'hola', 'buenos', 'buenas', 'días', 'tardes', 'noches', 'gracias',
        'señor', 'señora', 'señorita', 'don', 'doña', 'doctor', 'doctora',
        'estimado', 'estimada', 'querido', 'querida', 'respetado', 'respetada',
        'atentamente', 'cordialmente', 'saludos', 'despedida', 'afectuosamente',
        # Apellidos comunes hispanos
        'garcía', 'rodríguez', 'martínez', 'lópez', 'gonzález', 'hernández',
        'pérez', 'sánchez', 'ramírez', 'torres', 'flores', 'rivera', 'gómez',
        'díaz', 'reyes', 'morales', 'jiménez', 'ruiz', 'álvarez', 'mendoza',
        'castro', 'ortiz', 'romero', 'ramos', 'silva', 'vargas', 'medina',
        # Meses y días
        'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
        'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
        'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo',
    }
    
    def __init__(self):
        """Inicializa el corrector."""
        self._spell = None
        self._custom_words = set(self.COMMON_SPANISH_WORDS)
        
        if HAS_SPELLCHECKER:
            try:
                self._spell = SpellChecker(language='es')
                # Agregar palabras comunes al corrector
                self._spell.word_frequency.load_words(list(self.COMMON_SPANISH_WORDS))
            except Exception:
                # Si no hay diccionario español, usar inglés como fallback
                self._spell = SpellChecker(language='en')
    
    def add_custom_words(self, words: list[str]):
        """Agrega palabras personalizadas al diccionario."""
        self._custom_words.update(w.lower() for w in words)
        if self._spell:
            self._spell.word_frequency.load_words(words)
    
    def is_valid_word(self, word: str) -> bool:
        """Verifica si una palabra está en el diccionario."""
        word_lower = word.lower()
        
        # Verificar diccionario personalizado
        if word_lower in self._custom_words:
            return True
        
        # Verificar con spellchecker
        if self._spell:
            return word_lower in self._spell
        
        return True  # Si no hay spellchecker, asumir válida
    
    def get_suggestions(self, word: str, max_suggestions: int = 5) -> list[str]:
        """
        Obtiene sugerencias para una palabra mal escrita.
        
        Args:
            word: Palabra a corregir
            max_suggestions: Máximo de sugerencias
            
        Returns:
            Lista de sugerencias ordenadas por probabilidad
        """
        if self._spell:
            candidates = self._spell.candidates(word)
            if candidates:
                return list(candidates)[:max_suggestions]
        
        return []
    
    def _find_best_match_in_dictionary(self, word: str, max_distance: int = 2) -> Optional[str]:
        """
        Encuentra la mejor coincidencia en el diccionario personalizado.
        
        Prioriza palabras con menor distancia de Levenshtein.
        """
        word_lower = word.lower()
        best_match = None
        best_distance = max_distance + 1
        
        for dict_word in self._custom_words:
            # Filtro rápido: diferencia de longitud no puede ser mayor que max_distance
            if abs(len(dict_word) - len(word_lower)) > max_distance:
                continue
            
            distance = self.levenshtein_distance(word_lower, dict_word)
            if distance < best_distance:
                best_distance = distance
                best_match = dict_word
        
        return best_match if best_distance <= max_distance else None
    
    def correct(self, word: str) -> str:
        """
        Corrige una palabra usando múltiples estrategias.
        
        Orden de prioridad:
        1. Si está en diccionario, retornar original
        2. Buscar en diccionario personalizado (distancia <= 2)
        3. Usar spellchecker externo
        
        Args:
            word: Palabra a corregir
            
        Returns:
            Palabra corregida (o la original si no hay corrección)
        """
        if not word or len(word) < 2:
            return word
            
        if self.is_valid_word(word):
            return word
        
        original_case = word[0].isupper() if word else False
        
        # Estrategia 1: Buscar en diccionario personalizado
        custom_match = self._find_best_match_in_dictionary(word)
        if custom_match:
            if original_case:
                return custom_match[0].upper() + custom_match[1:]
            return custom_match
        
        # Estrategia 2: Usar spellchecker externo
        if self._spell:
            correction = self._spell.correction(word)
            if correction and correction != word.lower():
                # Verificar que la corrección no sea peor
                orig_dist = len(word)  # Distancia máxima posible
                corr_dist = self.levenshtein_distance(word.lower(), correction)
                
                if corr_dist <= 2:  # Solo aceptar correcciones cercanas
                    if original_case:
                        return correction[0].upper() + correction[1:]
                    return correction
        
        return word
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calcula la distancia de Levenshtein entre dos strings.
        
        Usa la librería optimizada si está disponible.
        """
        if HAS_LEVENSHTEIN:
            return Levenshtein.distance(s1, s2)
        
        # Implementación manual si no hay librería
        if len(s1) < len(s2):
            return SpanishSpellChecker.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class SpanishLanguageModel:
    """
    Modelo de Lenguaje completo para post-procesamiento OCR en español.
    
    Combina:
    - Cadenas de Markov (n-gramas de caracteres)
    - Reglas fonéticas del español
    - Corrector ortográfico (SymSpell + spellchecker)
    - Frecuencias de palabras (wordfreq)
    - Re-ranking de candidatos OCR
    
    Uso:
        lm = SpanishLanguageModel()
        
        # Corregir texto completo
        corrected = lm.correct_text("Holla mundo")
        
        # Re-rankear candidatos de beam search
        best = lm.rerank_candidates(["Holla", "Hola", "Holia"])
        
        # Obtener score de probabilidad
        score = lm.score_text("Hola mundo")
    """
    
    def __init__(
        self,
        use_ngrams: bool = True,
        use_phonetic_rules: bool = True,
        use_spellchecker: bool = True,
        use_wordfreq: bool = True,
        use_symspell: bool = True,
        ngram_weight: float = 0.2,
        phonetic_weight: float = 0.1,
        spell_weight: float = 0.2,
        wordfreq_weight: float = 0.5,  # Más peso a wordfreq porque es muy bueno
    ):
        """
        Inicializa el modelo de lenguaje.
        
        Args:
            use_ngrams: Usar modelo de n-gramas de caracteres
            use_phonetic_rules: Usar reglas fonéticas del español
            use_spellchecker: Usar corrector ortográfico básico
            use_wordfreq: Usar frecuencias de palabras (corpus español)
            use_symspell: Usar SymSpell (corrector rápido)
            ngram_weight: Peso del score de n-gramas
            phonetic_weight: Peso de reglas fonéticas
            spell_weight: Peso del corrector básico
            wordfreq_weight: Peso de frecuencias de palabras
        """
        self.use_ngrams = use_ngrams
        self.use_phonetic_rules = use_phonetic_rules
        self.use_spellchecker = use_spellchecker
        self.use_wordfreq = use_wordfreq and HAS_WORDFREQ
        self.use_symspell = use_symspell and HAS_SYMSPELL
        
        # Pesos normalizados
        weights = []
        if use_ngrams:
            weights.append(ngram_weight)
        if use_phonetic_rules:
            weights.append(phonetic_weight)
        if use_spellchecker:
            weights.append(spell_weight)
        if self.use_wordfreq:
            weights.append(wordfreq_weight)
        
        total_weight = sum(weights) if weights else 1.0
        self.ngram_weight = ngram_weight / total_weight if use_ngrams else 0
        self.phonetic_weight = phonetic_weight / total_weight if use_phonetic_rules else 0
        self.spell_weight = spell_weight / total_weight if use_spellchecker else 0
        self.wordfreq_weight = wordfreq_weight / total_weight if self.use_wordfreq else 0
        
        # Inicializar componentes
        self.ngrams = SpanishCharNgrams(n=3) if use_ngrams else None
        self.phonetic = SpanishPhoneticRules if use_phonetic_rules else None
        self.spell = SpanishSpellChecker() if use_spellchecker else None
        self.symspell = SpanishSymSpell() if self.use_symspell else None
        
        # Mostrar qué componentes están activos
        self._log_components()
    
    def _log_components(self):
        """Imprime qué componentes están activos (solo en debug)."""
        components = []
        if self.ngrams:
            components.append("N-gramas")
        if self.phonetic:
            components.append("Reglas fonéticas")
        if self.spell:
            components.append("SpellChecker")
        if self.use_wordfreq:
            components.append("WordFreq (corpus español)")
        if self.symspell:
            components.append("SymSpell")
        # print(f"[LM] Componentes activos: {', '.join(components)}")
    
    def score_text(self, text: str) -> float:
        """
        Calcula un score de probabilidad para el texto.
        
        Score más alto = texto más probable en español.
        Combina múltiples señales: n-gramas, fonética, diccionario, wordfreq.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Score normalizado entre 0 y 1
        """
        if not text:
            return 0.0
        
        scores = []
        
        # Score de n-gramas (probabilidad de secuencias de caracteres)
        if self.ngrams:
            ngram_score = self.ngrams.score_sequence(text)
            # Convertir log-prob a score 0-1
            ngram_score = math.exp(ngram_score)
            ngram_score = min(1.0, ngram_score * 10)  # Escalar
            scores.append(('ngram', ngram_score, self.ngram_weight))
        
        # Score de reglas fonéticas (validez fonética del español)
        if self.phonetic:
            words = text.split()
            valid_count = sum(1 for w in words if self.phonetic.validate_word(w)[0])
            phonetic_score = valid_count / max(1, len(words))
            scores.append(('phonetic', phonetic_score, self.phonetic_weight))
        
        # Score de corrector ortográfico básico
        if self.spell:
            words = re.findall(r'\w+', text)
            valid_count = sum(1 for w in words if self.spell.is_valid_word(w))
            spell_score = valid_count / max(1, len(words))
            scores.append(('spell', spell_score, self.spell_weight))
        
        # Score de wordfreq (frecuencia real en corpus español)
        # Este es el más confiable porque usa datos de Wikipedia, Twitter, etc.
        if self.use_wordfreq:
            wordfreq_score = SpanishWordFreq.score_text(text)
            scores.append(('wordfreq', wordfreq_score, self.wordfreq_weight))
        
        # Promedio ponderado
        if not scores:
            return 0.5
        
        total_score = sum(score * weight for _, score, weight in scores)
        return total_score
    
    def correct_text(self, text: str) -> str:
        """
        Corrige el texto usando todas las técnicas disponibles.
        
        Pipeline de corrección:
        1. WordFreq - Para validar que las palabras existen
        2. Reglas fonéticas - Para errores específicos del español (q→qu)
        3. SymSpell/SpellChecker - Para corrección ortográfica
        
        Args:
            text: Texto a corregir
            
        Returns:
            Texto corregido
        """
        if not text:
            return text
        
        # Corrección palabra por palabra (más controlado)
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        corrected_tokens = []
        
        # Track si es primera palabra de oración
        is_first_word = True
        
        for token in tokens:
            # Si es palabra
            if re.match(r'\w+', token):
                # No corregir números ni palabras muy cortas
                if token.isdigit() or len(token) < 2:
                    corrected_tokens.append(token)
                else:
                    corrected = self._correct_word(token, is_first_word=is_first_word)
                    corrected_tokens.append(corrected)
                is_first_word = False
            else:
                # Preservar espacios y puntuación
                corrected_tokens.append(token)
                # Reset is_first_word después de puntuación final
                if token in '.!?':
                    is_first_word = True
        
        return ''.join(corrected_tokens)
    
    def _correct_word(self, word: str, is_first_word: bool = False) -> str:
        """
        Corrige una palabra individual usando múltiples estrategias.
        
        Estrategia: buscar alternativas y elegir la de mayor frecuencia
        si es significativamente mejor que la original.
        
        Args:
            word: Palabra a corregir
            is_first_word: Si es la primera palabra de la oración
            
        Returns:
            Palabra corregida
        """
        if not word or len(word) < 2:
            return word
        
        # Guardar info del original
        original_case = word[0].isupper() if word else False
        word_lower = word.lower()
        
        # Determinar si es nombre propio (capitalizada pero NO primera palabra)
        is_proper_noun = original_case and not is_first_word and word[1:].islower() if len(word) > 1 else False
        
        # Frecuencia original
        orig_freq = SpanishWordFreq.get_frequency(word_lower) if self.use_wordfreq else 0
        
        # Si es muy frecuente (>5), probablemente es correcta
        if orig_freq >= 5.0:
            return word
        
        # Coleccionar candidatos de corrección
        candidates = {word_lower: orig_freq}
        
        # 1. Corrección fonética (q→qu, etc.)
        if self.phonetic:
            phonetic_correction = self.phonetic.suggest_correction(word_lower)
            if phonetic_correction and phonetic_correction != word_lower:
                freq = SpanishWordFreq.get_frequency(phonetic_correction) if self.use_wordfreq else 0
                candidates[phonetic_correction] = freq
        
        # 2. Correcciones comunes de OCR (tildes, ñ, ll, etc.)
        ocr_corrections = self._get_ocr_corrections(word_lower)
        for corr in ocr_corrections:
            if corr not in candidates:
                freq = SpanishWordFreq.get_frequency(corr) if self.use_wordfreq else 0
                candidates[corr] = freq
        
        # 3. SymSpell suggestion (solo si no es nombre propio)
        if self.symspell and not is_proper_noun:
            sym_correction = self.symspell.correct(word_lower)
            if sym_correction and sym_correction not in candidates:
                freq = SpanishWordFreq.get_frequency(sym_correction) if self.use_wordfreq else 0
                candidates[sym_correction] = freq
        
        # 4. WordFreq suggestion (si la palabra tiene baja frecuencia)
        if self.use_wordfreq and orig_freq < 4.0 and not is_proper_noun:
            wf_suggestion = SpanishWordFreq.suggest_correction(word_lower, max_distance=1)
            if wf_suggestion and wf_suggestion not in candidates:
                freq = SpanishWordFreq.get_frequency(wf_suggestion)
                candidates[wf_suggestion] = freq
        
        # Elegir el mejor candidato
        best_word = word_lower
        best_freq = orig_freq
        
        for cand, freq in candidates.items():
            # Para nombres propios, solo aceptar correcciones de tildes (misma longitud)
            if is_proper_noun:
                if len(cand) == len(word_lower) and freq > best_freq:
                    best_word = cand
                    best_freq = freq
            else:
                # Para palabras normales: candidato debe ser mejor (>0.5 punto Zipf)
                if freq > best_freq + 0.5:
                    best_word = cand
                    best_freq = freq
        
        # Preservar capitalización
        if original_case and best_word:
            best_word = best_word[0].upper() + best_word[1:]
        
        return best_word
    
    def _get_ocr_corrections(self, word: str) -> list:
        """
        Genera correcciones comunes para errores de OCR.
        
        Errores típicos:
        - Falta de tildes: a→á, e→é, i→í, o→ó, u→ú
        - n→ñ
        - b↔v
        - ll↔l (doble l)
        """
        corrections = []
        
        # Mapa de correcciones de tildes
        tilde_map = {
            'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú',
            'n': 'ñ'
        }
        
        # Generar variantes con tildes
        for i, char in enumerate(word):
            if char in tilde_map:
                variant = word[:i] + tilde_map[char] + word[i+1:]
                corrections.append(variant)
        
        # b↔v
        if 'b' in word:
            corrections.append(word.replace('b', 'v'))
        if 'v' in word:
            corrections.append(word.replace('v', 'b'))
        
        # ll↔l (solo si tiene sentido)
        if 'll' in word:
            corrections.append(word.replace('ll', 'l'))
        # No agregar ll donde hay l simple porque genera demasiados candidatos
        
        return corrections
    
    def correct_word(self, word: str, is_first_word: bool = True) -> str:
        """Corrige una sola palabra (alias público)."""
        return self._correct_word(word, is_first_word=is_first_word)
    
    def rerank_candidates(
        self,
        candidates: list[str],
        ocr_scores: Optional[list[float]] = None,
        ocr_weight: float = 0.6,
    ) -> list[str]:
        """
        Re-ordena candidatos OCR según probabilidad lingüística.
        
        Combina el score del OCR con el score del modelo de lenguaje.
        
        Args:
            candidates: Lista de candidatos del beam search
            ocr_scores: Scores del modelo OCR (opcional)
            ocr_weight: Peso del score OCR vs LM
            
        Returns:
            Candidatos re-ordenados (mejor primero)
        """
        if not candidates:
            return []
        
        if ocr_scores is None:
            ocr_scores = [1.0] * len(candidates)
        
        # Calcular score combinado
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            lm_score = self.score_text(candidate)
            ocr_score = ocr_scores[i] if i < len(ocr_scores) else 1.0
            
            # Score combinado
            combined_score = (ocr_weight * ocr_score) + ((1 - ocr_weight) * lm_score)
            scored_candidates.append((candidate, combined_score, lm_score, ocr_score))
        
        # Ordenar por score combinado
        scored_candidates.sort(key=lambda x: -x[1])
        
        return [c[0] for c in scored_candidates]
    
    def analyze_text(self, text: str) -> dict:
        """
        Analiza un texto y devuelve métricas detalladas.
        
        Útil para debugging y entender por qué el modelo
        prefiere ciertas correcciones.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con métricas detalladas
        """
        analysis = {
            'text': text,
            'overall_score': self.score_text(text),
            'word_count': len(text.split()),
            'char_count': len(text),
        }
        
        # Análisis de n-gramas
        if self.ngrams:
            analysis['ngrams'] = {
                'score': math.exp(self.ngrams.score_sequence(text)),
                'has_impossible': self.ngrams.has_impossible_sequence(text),
                'impossible_sequences': self.ngrams.find_impossible_sequences(text),
            }
        
        # Análisis fonético
        if self.phonetic:
            words = text.split()
            word_analysis = []
            for word in words:
                is_valid, errors = self.phonetic.validate_word(word)
                word_analysis.append({
                    'word': word,
                    'is_valid': is_valid,
                    'errors': errors,
                })
            analysis['phonetic'] = {
                'valid_words': sum(1 for w in word_analysis if w['is_valid']),
                'total_words': len(words),
                'word_details': word_analysis,
            }
        
        # Análisis ortográfico
        if self.spell:
            words = re.findall(r'\w+', text)
            spell_analysis = []
            for word in words:
                is_valid = self.spell.is_valid_word(word)
                suggestions = [] if is_valid else self.spell.get_suggestions(word, 3)
                spell_analysis.append({
                    'word': word,
                    'is_valid': is_valid,
                    'suggestions': suggestions,
                })
            analysis['spelling'] = {
                'valid_words': sum(1 for w in spell_analysis if w['is_valid']),
                'total_words': len(words),
                'word_details': spell_analysis,
            }
        
        return analysis
    
    def get_correction_metrics(
        self,
        prediction: str,
        correction: str,
    ) -> dict:
        """
        Calcula métricas entre predicción y corrección.
        
        Útil para el sistema de feedback del dashboard.
        
        Args:
            prediction: Texto predicho por OCR
            correction: Texto corregido por usuario
            
        Returns:
            Diccionario con métricas
        """
        # Distancia de Levenshtein
        if HAS_LEVENSHTEIN:
            distance = Levenshtein.distance(prediction, correction)
            ratio = Levenshtein.ratio(prediction, correction)
        else:
            distance = SpanishSpellChecker.levenshtein_distance(prediction, correction)
            max_len = max(len(prediction), len(correction))
            ratio = 1 - (distance / max_len) if max_len > 0 else 1.0
        
        # Categorizar error
        if prediction == correction:
            category = 'correct'
        elif distance <= 2:
            category = 'almost_correct'
        else:
            category = 'incorrect'
        
        # Encontrar diferencias caracter por caracter
        char_errors = []
        for i, (p, c) in enumerate(zip(prediction, correction)):
            if p != c:
                char_errors.append({
                    'position': i,
                    'predicted': p,
                    'actual': c,
                })
        
        # Peso para training
        if category == 'correct':
            training_weight = 1.0
        elif category == 'almost_correct':
            training_weight = 2.0  # Más valioso para aprender
        else:
            training_weight = 1.0
        
        return {
            'prediction': prediction,
            'correction': correction,
            'is_correct': prediction == correction,
            'similarity_score': ratio,
            'levenshtein_distance': distance,
            'error_category': category,
            'char_errors': char_errors,
            'training_weight': training_weight,
        }


# === Funciones de conveniencia ===

_default_lm: Optional[SpanishLanguageModel] = None


def get_language_model() -> SpanishLanguageModel:
    """Obtiene la instancia singleton del modelo de lenguaje."""
    global _default_lm
    if _default_lm is None:
        _default_lm = SpanishLanguageModel()
    return _default_lm


def correct_text(text: str) -> str:
    """Corrige texto usando el modelo de lenguaje por defecto."""
    return get_language_model().correct_text(text)


def score_text(text: str) -> float:
    """Calcula score de un texto."""
    return get_language_model().score_text(text)


def rerank_candidates(candidates: list[str]) -> list[str]:
    """Re-ordena candidatos OCR."""
    return get_language_model().rerank_candidates(candidates)


def get_correction_metrics(prediction: str, correction: str) -> dict:
    """Calcula métricas entre predicción y corrección."""
    return get_language_model().get_correction_metrics(prediction, correction)
