"""
Unit tests for VexaTokenizer
"""

import unittest
import tempfile
import os
from src.utils.tokenizer import VexaTokenizer


class TestVexaTokenizer(unittest.TestCase):
    """Test cases for VexaTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "Hello world",
            "Test string",
            "Polish: ąśćźż"
        ]
        self.tokenizer = VexaTokenizer()

    def test_build_vocab(self):
        """Test vocabulary building."""
        self.tokenizer.build_vocab(self.sample_texts)

        # Check that vocab was built
        self.assertGreater(self.tokenizer.vocab_size, 0)

        # Check special tokens are present
        self.assertIn('<PAD>', self.tokenizer.token_to_id)
        self.assertIn('<UNK>', self.tokenizer.token_to_id)
        self.assertIn('<BOS>', self.tokenizer.token_to_id)
        self.assertIn('<EOS>', self.tokenizer.token_to_id)

        # Check Polish characters are included
        self.assertIn('ą', self.tokenizer.token_to_id)

    def test_encode_decode(self):
        """Test encoding and decoding."""
        self.tokenizer.build_vocab(self.sample_texts)

        test_text = "Hello"
        encoded = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(encoded)

        # Should include BOS and EOS tokens in encoded
        self.assertEqual(len(encoded), len(test_text) + 2)
        self.assertEqual(decoded, test_text)

    def test_encode_without_special_tokens(self):
        """Test encoding without special tokens."""
        self.tokenizer.build_vocab(self.sample_texts)

        test_text = "Hello"
        encoded = self.tokenizer.encode(test_text, add_special_tokens=False)

        # Should not include BOS and EOS tokens
        self.assertEqual(len(encoded), len(test_text))

    def test_decode_skip_special_tokens(self):
        """Test decoding with special tokens skipped."""
        self.tokenizer.build_vocab(self.sample_texts)

        test_text = "Hello"
        encoded = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)

        # Should include BOS and EOS in decoded text
        self.assertIn('<BOS>', decoded)
        self.assertIn('<EOS>', decoded)

    def test_unknown_character(self):
        """Test handling of unknown characters."""
        self.tokenizer.build_vocab(self.sample_texts)

        # Use a character not in training data
        test_text = "Hello👋"
        encoded = self.tokenizer.encode(test_text, add_special_tokens=False)

        # Unknown character should be mapped to UNK
        unk_id = self.tokenizer.get_unk_id()
        self.assertIn(unk_id, encoded)

    def test_save_load_vocab(self):
        """Test saving and loading vocabulary."""
        self.tokenizer.build_vocab(self.sample_texts)

        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_path = os.path.join(temp_dir, 'test_vocab.json')

            # Save vocab
            self.tokenizer.save_vocab(vocab_path)
            self.assertTrue(os.path.exists(vocab_path))

            # Load vocab in new tokenizer
            new_tokenizer = VexaTokenizer()
            new_tokenizer.load_vocab(vocab_path)

            # Check vocabularies match
            self.assertEqual(self.tokenizer.vocab_size, new_tokenizer.vocab_size)
            self.assertEqual(self.tokenizer.token_to_id, new_tokenizer.token_to_id)

    def test_special_token_ids(self):
        """Test special token ID getters."""
        self.tokenizer.build_vocab(self.sample_texts)

        self.assertEqual(self.tokenizer.get_pad_id(), self.tokenizer.token_to_id['<PAD>'])
        self.assertEqual(self.tokenizer.get_unk_id(), self.tokenizer.token_to_id['<UNK>'])
        self.assertEqual(self.tokenizer.get_bos_id(), self.tokenizer.token_to_id['<BOS>'])
        self.assertEqual(self.tokenizer.get_eos_id(), self.tokenizer.token_to_id['<EOS>'])

    def test_empty_text(self):
        """Test handling of empty text."""
        self.tokenizer.build_vocab(self.sample_texts)

        encoded = self.tokenizer.encode("")
        decoded = self.tokenizer.decode(encoded)

        # Should still have BOS and EOS
        self.assertEqual(len(encoded), 2)
        self.assertEqual(decoded, "")

    def test_word_level_tokenization(self):
        """Test word-level tokenization."""
        word_tokenizer = VexaTokenizer(tokenization_level='word')
        word_tokenizer.build_vocab(self.sample_texts)

        # Check that vocab was built
        self.assertGreater(word_tokenizer.vocab_size, 0)

        # Check special tokens are present
        self.assertIn('<PAD>', word_tokenizer.token_to_id)
        self.assertIn('<UNK>', word_tokenizer.token_to_id)
        self.assertIn('<BOS>', word_tokenizer.token_to_id)
        self.assertIn('<EOS>', word_tokenizer.token_to_id)

        # Check words are included
        self.assertIn('Hello', word_tokenizer.token_to_id)
        self.assertIn('world', word_tokenizer.token_to_id)

    def test_word_level_encode_decode(self):
        """Test encoding and decoding with word-level."""
        word_tokenizer = VexaTokenizer(tokenization_level='word')
        word_tokenizer.build_vocab(self.sample_texts)

        test_text = "Hello world"
        encoded = word_tokenizer.encode(test_text)
        decoded = word_tokenizer.decode(encoded)

        # Should include BOS and EOS tokens in encoded
        self.assertEqual(len(encoded), len(test_text.split()) + 2)
        self.assertEqual(decoded, test_text)

    def test_word_level_unknown_word(self):
        """Test handling of unknown words."""
        word_tokenizer = VexaTokenizer(tokenization_level='word')
        word_tokenizer.build_vocab(self.sample_texts)

        # Use a word not in training data
        test_text = "Hello unknownword"
        encoded = word_tokenizer.encode(test_text, add_special_tokens=False)

        # Unknown word should be mapped to UNK
        unk_id = word_tokenizer.get_unk_id()
        self.assertIn(unk_id, encoded)


if __name__ == '__main__':
    unittest.main()
