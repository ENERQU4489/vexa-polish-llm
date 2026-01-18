"""
VexaTokenizer - Custom tokenizer for Polish language
Supports character-level or word-level tokenization
Supports Polish diacritical marks and special characters
"""

import json
import os
from typing import List, Dict, Optional


class VexaTokenizer:
    """
    Tokenizer optimized for Polish language.
    Supports character-level or word-level tokenization.
    Supports Polish diacritical marks: ą, ć, ę, ł, ń, ó, ś, ź, ż
    """

    def __init__(self, vocab_path: Optional[str] = None, tokenization_level: str = 'char'):
        """
        Initialize tokenizer.

        Args:
            vocab_path: Path to vocabulary file (vocab.json)
            tokenization_level: 'char' for character-level, 'word' for word-level
        """
        self.vocab_path = vocab_path
        self.tokenization_level = tokenization_level
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size = 0

        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'

        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from list of texts.

        Args:
            texts: List of texts to analyze
            min_freq: Minimum token frequency
        """
        token_freq = {}
        for text in texts:
            if self.tokenization_level == 'char':
                tokens = list(text)
            elif self.tokenization_level == 'word':
                tokens = text.split()
            else:
                raise ValueError(f"Unsupported tokenization_level: {self.tokenization_level}")
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.token_to_id = {token: idx for idx, token in enumerate(special_tokens)}

        current_id = len(special_tokens)
        for token, freq in sorted(token_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq:
                self.token_to_id[token] = current_id
                current_id += 1

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

        token_type = 'characters' if self.tokenization_level == 'char' else 'words'
        print(f"✓ Vocabulary built: {self.vocab_size} unique {token_type}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to list of IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id[self.BOS_TOKEN])

        if self.tokenization_level == 'char':
            tokens = list(text)
        elif self.tokenization_level == 'word':
            tokens = text.split()
        else:
            raise ValueError(f"Unsupported tokenization_level: {self.tokenization_level}")

        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            ids.append(token_id)

        if add_special_tokens:
            ids.append(self.token_to_id[self.EOS_TOKEN])

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
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN]
        }

        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)

        if self.tokenization_level == 'char':
            return ''.join(tokens)
        elif self.tokenization_level == 'word':
            return ' '.join(tokens)
        else:
            raise ValueError(f"Unsupported tokenization_level: {self.tokenization_level}")
    
    def save_vocab(self, path: str) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: File path
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'tokenization_level': self.tokenization_level,
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

        self.token_to_id = {k: int(v) for k, v in vocab_data['token_to_id'].items()}
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
        self.vocab_size = vocab_data['vocab_size']
        self.tokenization_level = vocab_data.get('tokenization_level', 'char')

        special = vocab_data.get('special_tokens', {})
        self.PAD_TOKEN = special.get('PAD', '<PAD>')
        self.UNK_TOKEN = special.get('UNK', '<UNK>')
        self.BOS_TOKEN = special.get('BOS', '<BOS>')
        self.EOS_TOKEN = special.get('EOS', '<EOS>')

        token_type = 'characters' if self.tokenization_level == 'char' else 'words'
        print(f"✓ Vocabulary loaded: {self.vocab_size} {token_type} from {path}")
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_pad_id(self) -> int:
        """Return PAD token ID."""
        return self.token_to_id[self.PAD_TOKEN]

    def get_unk_id(self) -> int:
        """Return UNK token ID."""
        return self.token_to_id[self.UNK_TOKEN]

    def get_bos_id(self) -> int:
        """Return BOS token ID."""
        return self.token_to_id[self.BOS_TOKEN]

    def get_eos_id(self) -> int:
        """Return EOS token ID."""
        return self.token_to_id[self.EOS_TOKEN]
