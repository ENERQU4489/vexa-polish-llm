"""
VexaTokenizer - Custom char-level tokenizer for Polish language
Supports Polish diacritical marks and special characters
"""

import json
import os
from typing import List, Dict, Optional


class VexaTokenizer:
    """
    Character-level tokenizer optimized for Polish language.
    Supports Polish diacritical marks: ą, ć, ę, ł, ń, ó, ś, ź, ż
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_path: Path to vocabulary file (vocab.json)
        """
        self.vocab_path = vocab_path
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build character vocabulary from list of texts.
        
        Args:
            texts: List of texts to analyze
            min_freq: Minimum character frequency
        """
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.char_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        
        current_id = len(special_tokens)
        for char, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq:
                self.char_to_id[char] = current_id
                current_id += 1
        
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
        print(f"✓ Vocabulary built: {self.vocab_size} unique characters")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to list of IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of character IDs
        """
        ids = []
        
        if add_special_tokens:
            ids.append(self.char_to_id[self.BOS_TOKEN])
        
        for char in text:
            char_id = self.char_to_id.get(char, self.char_to_id[self.UNK_TOKEN])
            ids.append(char_id)
        
        if add_special_tokens:
            ids.append(self.char_to_id[self.EOS_TOKEN])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode list of IDs to text.
        
        Args:
            ids: List of IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        special_ids = {
            self.char_to_id[self.PAD_TOKEN],
            self.char_to_id[self.UNK_TOKEN],
            self.char_to_id[self.BOS_TOKEN],
            self.char_to_id[self.EOS_TOKEN]
        }
        
        chars = []
        for char_id in ids:
            if skip_special_tokens and char_id in special_ids:
                continue
            char = self.id_to_char.get(char_id, self.UNK_TOKEN)
            chars.append(char)
        
        return ''.join(chars)
    
    def save_vocab(self, path: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            path: File path
        """
        vocab_data = {
            'char_to_id': self.char_to_id,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'PAD': self.PAD_TOKEN,
                'UNK': self.UNK_TOKEN,
                'BOS': self.BOS_TOKEN,
                'EOS': self.EOS_TOKEN
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Vocabulary saved: {path}")
    
    def load_vocab(self, path: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            path: File path
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_id = {k: int(v) for k, v in vocab_data['char_to_id'].items()}
        self.id_to_char = {int(idx): char for char, idx in self.char_to_id.items()}
        self.vocab_size = vocab_data['vocab_size']
        
        special = vocab_data.get('special_tokens', {})
        self.PAD_TOKEN = special.get('PAD', '<PAD>')
        self.UNK_TOKEN = special.get('UNK', '<UNK>')
        self.BOS_TOKEN = special.get('BOS', '<BOS>')
        self.EOS_TOKEN = special.get('EOS', '<EOS>')
        
        print(f"✓ Vocabulary loaded: {self.vocab_size} characters from {path}")
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_pad_id(self) -> int:
        """Return PAD token ID."""
        return self.char_to_id[self.PAD_TOKEN]
    
    def get_unk_id(self) -> int:
        """Return UNK token ID."""
        return self.char_to_id[self.UNK_TOKEN]
    
    def get_bos_id(self) -> int:
        """Return BOS token ID."""
        return self.char_to_id[self.BOS_TOKEN]
    
    def get_eos_id(self) -> int:
        """Return EOS token ID."""
        return self.char_to_id[self.EOS_TOKEN]
