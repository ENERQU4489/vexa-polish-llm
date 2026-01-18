"""
Podstawowe testy jednostkowe dla Vexa Polish LLM
Uruchom: python -m pytest tests/
"""

import sys
import os
import pytest
import numpy as np

# Dodanie src do ścieżki
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import VexaTokenizer, TextCleaner
from src.core import AntGraph, TrainingAnt


class TestVexaTokenizer:
    """Testy dla tokenizera"""
    
    def test_build_vocab(self):
        """Test budowania słownika"""
        tokenizer = VexaTokenizer()
        texts = ["Witaj świecie!", "Polska to piękny kraj."]
        
        tokenizer.build_vocab(texts, min_freq=1)
        
        assert tokenizer.vocab_size > 0
        assert tokenizer.get_vocab_size() > 0
    
    def test_encode_decode(self):
        """Test kodowania i dekodowania"""
        tokenizer = VexaTokenizer()
        texts = ["Test"]
        tokenizer.build_vocab(texts)
        
        text = "Test"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        assert decoded == text
    
    def test_special_tokens(self):
        """Test specjalnych tokenów"""
        tokenizer = VexaTokenizer()
        texts = ["Test"]
        tokenizer.build_vocab(texts)
        
        assert tokenizer.get_pad_id() >= 0
        assert tokenizer.get_unk_id() >= 0
        assert tokenizer.get_bos_id() >= 0
        assert tokenizer.get_eos_id() >= 0
    
    def test_polish_characters(self):
        """Test polskich znaków"""
        tokenizer = VexaTokenizer()
        texts = ["ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ"]
        tokenizer.build_vocab(texts)
        
        text = "ąćę"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        assert decoded == text


class TestTextCleaner:
    """Testy dla cleanera"""
    
    def test_clean_basic(self):
        """Test podstawowego czyszczenia"""
        cleaner = TextCleaner()
        
        text = "Test   text   with   spaces"
        cleaned = cleaner.clean(text)
        
        assert "   " not in cleaned
    
    def test_remove_urls(self):
        """Test usuwania URLi"""
        cleaner = TextCleaner(remove_urls=True)
        
        text = "Visit http://example.com for more"
        cleaned = cleaner.clean(text)
        
        assert "http://" not in cleaned
    
    def test_remove_wiki_markup(self):
        """Test usuwania Wiki markup"""
        cleaner = TextCleaner()
        
        text = "[[Polska]] to [[Europa|kraj]]"
        cleaned = cleaner.remove_wiki_markup(text)
        
        assert "[[" not in cleaned
        assert "]]" not in cleaned
    
    def test_normalize_polish(self):
        """Test normalizacji polskiego tekstu"""
        cleaner = TextCleaner()
        
        text = "Polska ąćęłńóśźż 123"
        cleaned = cleaner.normalize_polish_text(text)
        
        assert "ą" in cleaned
        assert "ć" in cleaned


class TestAntGraph:
    """Testy dla grafu ACO"""
    
    def test_initialization(self):
        """Test inicjalizacji grafu"""
        graph = AntGraph(vocab_size=100, tau_init=0.1)
        
        assert graph.vocab_size == 100
        assert graph.pheromones.shape == (100, 100)
        assert graph.heuristics.shape == (100, 100)
    
    def test_transition_probabilities(self):
        """Test prawdopodobieństw przejścia"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        
        probs = graph.get_transition_probabilities(
            current_token=0,
            alpha=1.0,
            beta=2.0
        )
        
        assert len(probs) == 10
        assert np.isclose(probs.sum(), 1.0)
        assert all(p >= 0 for p in probs)
    
    def test_update_pheromones(self):
        """Test aktualizacji feromonów"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        
        initial_pheromones = graph.pheromones.copy()
        
        paths = [[0, 1, 2], [0, 2, 3]]
        rewards = [0.8, 0.9]
        
        graph.update_pheromones(paths, rewards)
        
        # Feromony powinny się zmienić
        assert not np.array_equal(graph.pheromones, initial_pheromones)
    
    def test_pheromone_bounds(self):
        """Test ograniczeń feromonów"""
        graph = AntGraph(
            vocab_size=10,
            tau_init=0.1,
            tau_min=0.01,
            tau_max=1.0
        )
        
        # Duża aktualizacja
        paths = [[0, 1]] * 100
        rewards = [10.0] * 100
        
        graph.update_pheromones(paths, rewards)
        
        # Sprawdzenie granic
        assert graph.pheromones.min() >= graph.tau_min
        assert graph.pheromones.max() <= graph.tau_max


class TestTrainingAnt:
    """Testy dla mrówki treningowej"""
    
    def test_initialization(self):
        """Test inicjalizacji mrówki"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        ant = TrainingAnt(ant_id=1, graph=graph, alpha=1.0, beta=2.0)
        
        assert ant.ant_id == 1
        assert ant.graph == graph
    
    def test_reset(self):
        """Test resetowania mrówki"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        ant = TrainingAnt(ant_id=1, graph=graph)
        
        ant.reset(start_token=5)
        
        assert ant.current_position == 5
        assert ant.path == [5]
    
    def test_generate_sequence(self):
        """Test generowania sekwencji"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        ant = TrainingAnt(ant_id=1, graph=graph)
        
        sequence = ant.generate_sequence(
            start_token=0,
            max_length=5,
            temperature=1.0
        )
        
        assert len(sequence) <= 5
        assert sequence[0] == 0
        assert all(0 <= token < 10 for token in sequence)
    
    def test_calculate_reward(self):
        """Test obliczania nagrody"""
        graph = AntGraph(vocab_size=10, tau_init=0.1)
        ant = TrainingAnt(ant_id=1, graph=graph)
        
        generated = [0, 1, 2, 3]
        target = [0, 1, 2, 3]
        
        reward = ant.calculate_reward(generated, target)
        
        assert 0.0 <= reward <= 1.0
        assert reward > 0.5  # Idealna zgodność


class TestIntegration:
    """Testy integracyjne"""
    
    def test_full_pipeline_small(self):
        """Test pełnego pipeline'u na małych danych"""
        # Tokenizer
        tokenizer = VexaTokenizer()
        texts = ["Test", "Polska", "ACO"]
        tokenizer.build_vocab(texts)
        
        # Graf
        graph = AntGraph(vocab_size=tokenizer.vocab_size, tau_init=0.1)
        
        # Dane treningowe
        sequences = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in texts
        ]
        
        graph.update_heuristics_from_data(sequences)
        
        # Mrówka
        ant = TrainingAnt(ant_id=1, graph=graph)
        
        # Trening
        for seq in sequences:
            if len(seq) > 1:
                reward = ant.train_on_sequence(seq)
                assert 0.0 <= reward <= 1.0
        
        # Aktualizacja feromonów
        paths = [ant.path]
        rewards = [0.5]
        graph.update_pheromones(paths, rewards)
        
        # Sprawdzenie że wszystko działa
        assert graph.total_updates > 0


def test_imports():
    """Test czy wszystkie moduły się importują"""
    from src.utils import VexaTokenizer, TextCleaner, WikiDownloader, DataSharder
    from src.core import AntGraph, TrainingAnt, VexaEngine
    from src.integration import VexaLLM
    
    assert VexaTokenizer is not None
    assert TextCleaner is not None
    assert WikiDownloader is not None
    assert DataSharder is not None
    assert AntGraph is not None
    assert TrainingAnt is not None
    assert VexaEngine is not None
    assert VexaLLM is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
