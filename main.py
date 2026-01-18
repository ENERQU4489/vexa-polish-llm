"""
Vexa Polish LLM - Main entry point script
Machine learning system based on Ant Colony Optimization (ACO) algorithm
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import VexaTokenizer, TextCleaner, WikiDownloader, DataSharder
from src.core import AntGraph, VexaEngine
from src.integration import VexaLLM


def load_config(config_path: str = 'config/hyperparams.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def download_data(num_articles: int = 100):
    """
    Download data from Wikipedia.
    
    Args:
        num_articles: Number of articles to download
    """
    print("\n" + "="*60)
    print("DOWNLOADING DATA FROM WIKIPEDIA")
    print("="*60 + "\n")
    
    downloader = WikiDownloader(language='pl', output_dir='data/raw')
    downloader.download_and_save(count=num_articles, method='random')
    
    print("\n✓ Data downloaded successfully!")


def prepare_data():
    """Prepare training data."""
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60 + "\n")
    
    if not os.path.exists('data/raw') or not os.listdir('data/raw'):
        print("⚠ No raw data. Run first: python main.py download")
        return
    
    config = load_config()
    
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        remove_numbers=False,
        lowercase=False,
        remove_extra_whitespace=True
    )
    
    sharder = DataSharder(tokenizer=None, max_sequence_length=config['max_sequence_length'])
    texts = sharder.load_text_files('data/raw')
    
    if not texts:
        print("⚠ No texts found to process")
        return
    
    print("Cleaning texts...")
    texts = cleaner.clean_batch(texts)
    texts = [cleaner.remove_wiki_markup(t) for t in texts]
    texts = [cleaner.normalize_polish_text(t) for t in texts]
    texts = cleaner.remove_short_texts(texts, min_length=50)
    
    print(f"✓ After cleaning: {len(texts)} texts")
    
    print("\nBuilding vocabulary...")
    tokenizer = VexaTokenizer()
    tokenizer.build_vocab(texts, min_freq=config.get('min_word_frequency', 2))
    tokenizer.save_vocab('data/vocab.json')
    
    print("\nCreating training sequences...")
    sharder.tokenizer = tokenizer
    input_seqs, target_seqs = sharder.create_sequences(texts)
    
    if input_seqs and target_seqs:
        sharder.save_training_data(input_seqs, target_seqs, 'data/training_data.bin')
        print("\n✓ Training data prepared!")
    else:
        print("\n⚠ Failed to create training data")


def train_model(resume: bool = False):
    """
    Train model.
    
    Args:
        resume: Whether to resume training from checkpoint
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60 + "\n")
    
    if not os.path.exists('data/training_data.bin'):
        print("⚠ No training data. Run first: python main.py prepare")
        return
    
    if not os.path.exists('data/vocab.json'):
        print("⚠ No vocabulary. Run first: python main.py prepare")
        return
    
    config = load_config()
    
    print("Loading tokenizer...")
    tokenizer = VexaTokenizer(vocab_path='data/vocab.json')
    
    print("Loading training data...")
    sharder = DataSharder(tokenizer=tokenizer, max_sequence_length=config['max_sequence_length'])
    input_seqs, target_seqs = sharder.load_training_data('data/training_data.bin')
    
    print("Initializing ACO graph...")
    graph = AntGraph(
        vocab_size=tokenizer.get_vocab_size(),
        tau_init=config['tau_init'],
        tau_min=config['tau_min'],
        tau_max=config['tau_max'],
        rho=config['rho'],
        use_gpu=config.get('use_gpu', False)
    )
    
    graph.update_heuristics_from_data(input_seqs)
    
    print("Initializing training engine...")
    engine = VexaEngine(graph=graph, tokenizer=tokenizer, config=config)
    
    if resume:
        latest_checkpoint = engine.get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming training from: {latest_checkpoint}")
            engine.load_checkpoint(latest_checkpoint)
        else:
            print("⚠ No checkpoint found, starting from scratch")
    
    try:
        engine.train(input_seqs, target_seqs, verbose=True)
        
        engine.save_checkpoint('final_model.ant')
        
        print("\n✓ Training completed successfully!")
        engine.print_stats()
        
    except KeyboardInterrupt:
        print("\n\n⏸ Training interrupted by user")
        engine.save_checkpoint('interrupted_model.ant')
        print("✓ Model saved")


def chat_mode():
    """Chat mode with model."""
    print("\n" + "="*60)
    print("VEXA POLISH LLM - CHAT MODE")
    print("="*60 + "\n")
    
    checkpoint_dir = 'data/checkpoints'
    if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
        print("⚠ No trained model. Run first: python main.py train")
        return
    
    config = load_config()
    
    print("Loading tokenizer...")
    tokenizer = VexaTokenizer(vocab_path='data/vocab.json')
    
    print("Loading model...")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ant')]
    latest_checkpoint = sorted(checkpoints, 
                              key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))[-1]
    
    graph = AntGraph.load(os.path.join(checkpoint_dir, latest_checkpoint), use_gpu=config.get('use_gpu', False))
    
    engine = VexaEngine(graph=graph, tokenizer=tokenizer, config=config)
    llm = VexaLLM(graph=graph, tokenizer=tokenizer, engine=engine, config=config)
    
    print("\n✓ Model loaded!")
    print("\nSpecial commands:")
    print("  /stats    - Display statistics")
    print("  /clear    - Clear history")
    print("  /feedback <0-1> - Rate last response")
    print("  /save     - Save conversation")
    print("  /quit     - Exit\n")
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                if user_input == '/quit':
                    print("\nGoodbye!")
                    break
                elif user_input == '/stats':
                    llm.print_stats()
                    engine.print_stats()
                    continue
                elif user_input == '/clear':
                    llm.clear_history()
                    continue
                elif user_input.startswith('/feedback'):
                    try:
                        rating = float(user_input.split()[1])
                        llm.provide_feedback(rating)
                    except (IndexError, ValueError):
                        print("⚠ Usage: /feedback <0-1>")
                    continue
                elif user_input == '/save':
                    llm.save_conversation('data/conversation.json')
                    continue
                else:
                    print("⚠ Unknown command")
                    continue
            
            print("Vexa: ", end="", flush=True)
            response = llm.chat(user_input)
            print(response)
            print()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    
    if llm.conversation_history:
        llm.save_conversation('data/last_conversation.json')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Vexa Polish LLM - ACO-based System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py download          # Download data from Wikipedia
  python main.py prepare           # Prepare training data
  python main.py train             # Train model
  python main.py train --resume    # Resume training
  python main.py chat              # Chat with model
  python main.py all               # Run all (download -> prepare -> train -> chat)
        """
    )
    
    parser.add_argument(
        'command',
        choices=['download', 'prepare', 'train', 'chat', 'all'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--articles',
        type=int,
        default=100,
        help='Number of articles to download (default: 100)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    args = parser.parse_args()
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    if args.command == 'download':
        download_data(args.articles)
    
    elif args.command == 'prepare':
        prepare_data()
    
    elif args.command == 'train':
        train_model(resume=args.resume)
    
    elif args.command == 'chat':
        chat_mode()
    
    elif args.command == 'all':
        print("\n🚀 Running full pipeline...\n")
        download_data(args.articles)
        prepare_data()
        train_model(resume=False)
        
        print("\n✓ Pipeline completed! Starting chat...\n")
        input("Press Enter to continue...")
        chat_mode()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
