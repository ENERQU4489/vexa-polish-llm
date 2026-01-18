"""
DataSharder - Convert texts to binary training format
Creates compressed file with token sequences
"""

import os
import pickle
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class DataSharder:
    """
    Class for converting texts to binary training data.
    Creates token sequences ready for use by ants.
    """
    
    def __init__(self, tokenizer, max_sequence_length: int = 1000):
        """
        Initialize sharder.
        
        Args:
            tokenizer: VexaTokenizer instance
            max_sequence_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
    
    def load_text_files(self, directory: str) -> List[str]:
        """
        Load all text files from directory.
        
        Args:
            directory: Path to directory with .txt files
            
        Returns:
            List of texts
        """
        texts = []
        
        if not os.path.exists(directory):
            print(f"⚠ Directory does not exist: {directory}")
            return texts
        
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"⚠ No .txt files in directory: {directory}")
            return texts
        
        print(f"Loading {len(txt_files)} text files...")
        
        for filename in tqdm(txt_files, desc="Loading files"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():
                        texts.append(text)
            except Exception as e:
                print(f"\n⚠ Error loading {filename}: {e}")
        
        print(f"✓ Loaded {len(texts)} files")
        return texts
    
    def create_sequences(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences (input, target) from texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple (input_sequences, target_sequences) as numpy arrays
        """
        all_inputs = []
        all_targets = []
        
        print("Creating training sequences...")
        
        for text in tqdm(texts, desc="Processing texts"):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            for i in range(len(token_ids) - 1):
                end_idx = min(i + self.max_sequence_length, len(token_ids) - 1)
                
                input_seq = token_ids[i:end_idx]
                target_seq = token_ids[i+1:end_idx+1]
                
                if len(input_seq) > 0 and len(target_seq) > 0:
                    all_inputs.append(input_seq)
                    all_targets.append(target_seq)
        
        print(f"✓ Created {len(all_inputs)} training sequences")
        
        return all_inputs, all_targets
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None) -> np.ndarray:
        """
        Pad sequences to equal length.
        
        Args:
            sequences: List of sequences
            max_length: Maximum length (None = longest sequence)
            
        Returns:
            Numpy array with padded sequences
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        pad_id = self.tokenizer.get_pad_id()
        padded = np.full((len(sequences), max_length), pad_id, dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def save_training_data(self, 
                          input_sequences: List[List[int]], 
                          target_sequences: List[List[int]], 
                          output_path: str) -> None:
        """
        Save training data to binary file.
        
        Args:
            input_sequences: Input sequences
            target_sequences: Target sequences
            output_path: Output file path
        """
        print(f"Saving training data to {output_path}...")
        
        training_data = {
            'inputs': input_sequences,
            'targets': target_sequences,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_sequence_length': self.max_sequence_length,
            'num_sequences': len(input_sequences)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Data saved: {output_path} ({file_size_mb:.2f} MB)")
    
    def load_training_data(self, input_path: str) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Load training data from binary file.
        
        Args:
            input_path: File path
            
        Returns:
            Tuple (input_sequences, target_sequences)
        """
        print(f"Loading training data from {input_path}...")
        
        with open(input_path, 'rb') as f:
            training_data = pickle.load(f)
        
        print(f"✓ Loaded {training_data['num_sequences']} sequences")
        print(f"  Vocabulary size: {training_data['vocab_size']}")
        print(f"  Max sequence length: {training_data['max_sequence_length']}")
        
        return training_data['inputs'], training_data['targets']
    
    def process_directory(self, 
                         input_dir: str, 
                         output_path: str,
                         cleaner=None) -> None:
        """
        Process entire directory of texts and create training file (all-in-one).
        
        Args:
            input_dir: Directory with .txt files
            output_path: Output file path
            cleaner: Optional TextCleaner for cleaning texts
        """
        texts = self.load_text_files(input_dir)
        
        if not texts:
            print("⚠ No texts to process")
            return
        
        if cleaner:
            print("Cleaning texts...")
            texts = cleaner.clean_batch(texts)
            texts = [cleaner.remove_wiki_markup(t) for t in tqdm(texts, desc="Removing markup")]
            texts = [cleaner.normalize_polish_text(t) for t in tqdm(texts, desc="Normalizing")]
            texts = cleaner.remove_short_texts(texts, min_length=50)
            print(f"✓ After cleaning: {len(texts)} texts remain")
        
        input_seqs, target_seqs = self.create_sequences(texts)
        
        if input_seqs and target_seqs:
            self.save_training_data(input_seqs, target_seqs, output_path)
        else:
            print("⚠ No sequences created")
    
    def get_batch(self, 
                  input_sequences: List[List[int]], 
                  target_sequences: List[List[int]], 
                  batch_size: int,
                  shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator for training data batches.
        
        Args:
            input_sequences: Input sequences
            target_sequences: Target sequences
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Tuple (input_batch, target_batch)
        """
        num_sequences = len(input_sequences)
        indices = np.arange(num_sequences)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_sequences, batch_size):
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_indices = indices[start_idx:end_idx]
            
            batch_inputs = [input_sequences[i] for i in batch_indices]
            batch_targets = [target_sequences[i] for i in batch_indices]
            
            max_len = max(max(len(seq) for seq in batch_inputs),
                         max(len(seq) for seq in batch_targets))
            
            padded_inputs = self.pad_sequences(batch_inputs, max_len)
            padded_targets = self.pad_sequences(batch_targets, max_len)
            
            yield padded_inputs, padded_targets
