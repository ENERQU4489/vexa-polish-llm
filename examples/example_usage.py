"""
Przykłady użycia Vexa Polish LLM API
Demonstracja różnych funkcjonalności systemu
"""

import sys
import os

# Dodanie src do ścieżki
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import VexaTokenizer, TextCleaner, WikiDownloader, DataSharder
from src.core import AntGraph, TrainingAnt, VexaEngine
from src.integration import VexaLLM
import yaml


def example_1_tokenizer():
    """Przykład 1: Użycie tokenizera"""
    print("\n" + "="*60)
    print("PRZYKŁAD 1: Tokenizer")
    print("="*60 + "\n")
    
    # Inicjalizacja tokenizera
    tokenizer = VexaTokenizer()
    
    # Budowanie słownika
    texts = [
        "Witaj świecie!",
        "Polska to piękny kraj.",
        "Uczenie maszynowe jest fascynujące."
    ]
    
    tokenizer.build_vocab(texts, min_freq=1)
    
    # Kodowanie
    text = "Witaj świecie!"
    encoded = tokenizer.encode(text)
    print(f"Tekst: {text}")
    print(f"Zakodowany: {encoded}")
    
    # Dekodowanie
    decoded = tokenizer.decode(encoded)
    print(f"Zdekodowany: {decoded}")
    
    # Zapis i wczytanie
    tokenizer.save_vocab('examples/vocab_example.json')
    print("\n✓ Słownik zapisany")


def example_2_cleaner():
    """Przykład 2: Czyszczenie tekstu"""
    print("\n" + "="*60)
    print("PRZYKŁAD 2: Czyszczenie tekstu")
    print("="*60 + "\n")
    
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        lowercase=False
    )
    
    # Tekst z Wiki markup
    wiki_text = """
    [[Polska]] to państwo w [[Europa|Europie]].
    Zobacz też: http://example.com
    Email: test@example.com
    <ref>Źródło</ref>
    """
    
    print("Oryginalny tekst:")
    print(wiki_text)
    
    # Czyszczenie
    cleaned = cleaner.clean(wiki_text)
    cleaned = cleaner.remove_wiki_markup(cleaned)
    cleaned = cleaner.normalize_polish_text(cleaned)
    
    print("\nWyczyszczony tekst:")
    print(cleaned)


def example_3_graph():
    """Przykład 3: Graf ACO"""
    print("\n" + "="*60)
    print("PRZYKŁAD 3: Graf ACO")
    print("="*60 + "\n")
    
    # Inicjalizacja grafu
    graph = AntGraph(
        vocab_size=100,
        tau_init=0.1,
        tau_min=0.01,
        tau_max=10.0,
        rho=0.1
    )
    
    print(f"Rozmiar słownika: {graph.vocab_size}")
    print(f"Początkowe feromony: {graph.tau_init}")
    
    # Symulacja danych treningowych
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 4, 5, 6],
        [2, 3, 4, 5, 7]
    ]
    
    # Aktualizacja heurystyki
    graph.update_heuristics_from_data(sequences)
    
    # Pobranie prawdopodobieństw przejścia
    probs = graph.get_transition_probabilities(
        current_token=1,
        alpha=1.0,
        beta=2.0
    )
    
    print(f"\nPrawdopodobieństwa przejścia z tokenu 1:")
    print(f"Top 5: {probs[:5]}")
    
    # Aktualizacja feromonów
    paths = [[1, 2, 3], [1, 2, 4]]
    rewards = [0.8, 0.9]
    graph.update_pheromones(paths, rewards)
    
    print("\n✓ Feromony zaktualizowane")
    
    # Statystyki
    stats = graph.get_statistics()
    print(f"\nStatystyki grafu:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_4_ant():
    """Przykład 4: Mrówka treningowa"""
    print("\n" + "="*60)
    print("PRZYKŁAD 4: Mrówka treningowa")
    print("="*60 + "\n")
    
    # Graf
    graph = AntGraph(vocab_size=50, tau_init=0.1)
    
    # Mrówka
    ant = TrainingAnt(
        ant_id=1,
        graph=graph,
        alpha=1.0,
        beta=2.0,
        exploration_rate=0.1
    )
    
    print(f"Mrówka: {ant}")
    
    # Generowanie sekwencji
    sequence = ant.generate_sequence(
        start_token=0,
        max_length=10,
        temperature=1.0
    )
    
    print(f"\nWygenerowana sekwencja: {sequence}")
    
    # Trening na sekwencji docelowej
    target = [0, 1, 2, 3, 4, 5]
    reward = ant.train_on_sequence(target, temperature=1.0)
    
    print(f"Sekwencja docelowa: {target}")
    print(f"Nagroda: {reward:.4f}")


def example_5_generation():
    """Przykład 5: Generowanie tekstu (wymaga wytrenowanego modelu)"""
    print("\n" + "="*60)
    print("PRZYKŁAD 5: Generowanie tekstu")
    print("="*60 + "\n")
    
    # Sprawdzenie czy model istnieje
    if not os.path.exists('data/vocab.json'):
        print("⚠️ Brak wytrenowanego modelu!")
        print("Uruchom najpierw: python main.py train")
        return
    
    # Wczytanie konfiguracji
    with open('config/hyperparams.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Wczytanie tokenizera
    tokenizer = VexaTokenizer(vocab_path='data/vocab.json')
    
    # Wczytanie grafu
    checkpoint_dir = 'data/checkpoints'
    if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
        print("⚠️ Brak checkpointu modelu!")
        return
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ant')]
    latest = sorted(checkpoints,
                   key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))[-1]
    
    graph = AntGraph.load(os.path.join(checkpoint_dir, latest))
    
    # Inicjalizacja silnika i LLM
    engine = VexaEngine(graph=graph, tokenizer=tokenizer, config=config)
    llm = VexaLLM(graph=graph, tokenizer=tokenizer, engine=engine, config=config)
    
    # Generowanie
    prompts = [
        "Polska to",
        "Historia",
        "Warszawa"
    ]
    
    print("Generowanie tekstu:\n")
    for prompt in prompts:
        response = llm.generate(
            prompt=prompt,
            max_length=50,
            temperature=0.8
        )
        print(f"Prompt: {prompt}")
        print(f"Odpowiedź: {response}\n")


def example_6_online_learning():
    """Przykład 6: Online learning"""
    print("\n" + "="*60)
    print("PRZYKŁAD 6: Online Learning")
    print("="*60 + "\n")
    
    if not os.path.exists('data/vocab.json'):
        print("⚠️ Brak wytrenowanego modelu!")
        return
    
    # Wczytanie modelu (jak w przykładzie 5)
    with open('config/hyperparams.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tokenizer = VexaTokenizer(vocab_path='data/vocab.json')
    
    checkpoint_dir = 'data/checkpoints'
    if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
        print("⚠️ Brak checkpointu!")
        return
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ant')]
    latest = sorted(checkpoints,
                   key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))[-1]

    graph = AntGraph.load(os.path.join(checkpoint_dir, latest))
    engine = VexaEngine(graph=graph, tokenizer=tokenizer, config=config)
    llm = VexaLLM(graph=graph, tokenizer=tokenizer, engine=engine, config=config)
    
    # Symulacja interakcji
    print("Symulacja interakcji z użytkownikiem:\n")
    
    interactions = [
        ("Cześć!", 1.0),  # Dobra odpowiedź
        ("Opowiedz o Polsce", 0.8),  # Średnia odpowiedź
        ("Co to jest AI?", 0.9)  # Dobra odpowiedź
    ]
    
    for user_input, feedback in interactions:
        response = llm.chat(user_input, learn_from_interaction=True)
        print(f"User: {user_input}")
        print(f"Vexa: {response}")
        print(f"Feedback: {feedback}\n")
        
        # Dodatkowa aktualizacja z feedbackiem
        llm.provide_feedback(feedback)
    
    print("✓ Model zaktualizowany na podstawie interakcji")


def main():
    """Uruchomienie wszystkich przykładów"""
    print("\n" + "="*60)
    print("VEXA POLISH LLM - PRZYKŁADY UŻYCIA")
    print("="*60)
    
    # Tworzenie katalogu na przykłady
    os.makedirs('examples', exist_ok=True)
    
    # Uruchomienie przykładów
    example_1_tokenizer()
    example_2_cleaner()
    example_3_graph()
    example_4_ant()
    example_5_generation()
    example_6_online_learning()
    
    print("\n" + "="*60)
    print("✓ Wszystkie przykłady zakończone!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
