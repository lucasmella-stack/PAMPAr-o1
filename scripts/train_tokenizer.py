# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Entrenar tokenizer BPE pequeño para LLARRI v7.2

Vocab size: 8,000 tokens (cabe en memoria)
"""

import sentencepiece as spm
from pathlib import Path
import os


def main():
    print('=' * 60)
    print('   ENTRENANDO TOKENIZER BPE PARA LLARRI')
    print('=' * 60)
    
    data_dir = Path('data/wikitext-103/wikitext-103-raw')
    train_file = data_dir / 'wiki.train.raw'
    
    # Directorio para el tokenizer
    os.makedirs('data/tokenizer', exist_ok=True)
    
    print(f'\n📖 Entrenando tokenizer en {train_file}...')
    print('   Vocab size: 8,000')
    print('   Modelo: BPE')
    
    # Entrenar SentencePiece BPE
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix='data/tokenizer/llarri_bpe',
        vocab_size=8000,
        model_type='bpe',
        character_coverage=0.9995,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        max_sentence_length=16384,
        # Tokens especiales
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    
    print('\n✅ Tokenizer entrenado!')
    print('   Guardado en: data/tokenizer/llarri_bpe.model')
    
    # Probar el tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load('data/tokenizer/llarri_bpe.model')
    
    print(f'\n📊 Vocab size real: {sp.get_piece_size()}')
    
    # Ejemplos
    ejemplos = [
        'The quick brown fox jumps over the lazy dog.',
        'Once upon a time in a land far away.',
        'Machine learning is transforming the world.',
    ]
    
    print('\n📝 Ejemplos de tokenización:')
    for texto in ejemplos:
        tokens = sp.encode_as_pieces(texto)
        ids = sp.encode_as_ids(texto)
        print(f'\n   "{texto}"')
        print(f'   Tokens: {tokens[:15]}...' if len(tokens) > 15 else f'   Tokens: {tokens}')
        print(f'   IDs: {ids[:15]}...' if len(ids) > 15 else f'   IDs: {ids}')
        print(f'   Num tokens: {len(tokens)}')


if __name__ == '__main__':
    main()
