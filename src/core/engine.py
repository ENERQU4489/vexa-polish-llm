"""
VexaEngine - Main training engine managing epochs and ants
Coordinates learning process and pheromone updates
"""

import numpy as np
import time
import os
from typing import List, Optional, Dict
from tqdm import tqdm
import threading
import queue


class VexaEngine:
    """
    Training engine for Vexa Polish LLM system.
    Manages ants, training epochs, and graph updates.
    """
    
    def __init__(self,
                 graph,
                 tokenizer,
                 config: Dict):
        """
        Initialize engine.
        
        Args:
            graph: AntGraph instance
            tokenizer: VexaTokenizer instance
            config: Configuration dictionary (from hyperparams.yaml)
        """
        self.graph = graph
        self.tokenizer = tokenizer
        self.config = config
        
        self.num_ants = config.get('num_ants', 50)
        self.num_epochs = config.get('num_epochs', 1000)
        self.sequence_length = config.get('sequence_length', 100)
        self.batch_size = config.get('batch_size', 10)
        
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 2.0)
        self.rho = config.get('rho', 0.1)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        self.temperature = config.get('temperature', 0.8)
        
        self.online_learning = config.get('online_learning', True)
        self.online_update_rate = config.get('online_update_rate', 0.5)

        self.use_gpu = config.get('use_gpu', False)

        self.checkpoint_interval = config.get('checkpoint_interval', 100)
        self.checkpoint_dir = 'data/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_sequences': 0,
            'avg_reward': 0.0,
            'best_reward': 0.0,
            'avg_loss': 0.0,
            'best_loss': float('inf'),
            'training_time': 0.0
        }

        self.is_training = False
        self.training_thread = None
        self.training_queue = queue.Queue()

        print("✓ VexaEngine initialized")
        print(f"  Number of ants: {self.num_ants}")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Online learning: {self.online_learning}")
        print(f"  GPU support: {self.use_gpu}")
    
    def train(self, 
              input_sequences: List[List[int]], 
              target_sequences: List[List[int]],
              verbose: bool = True) -> None:
        """
        Main training loop.
        
        Args:
            input_sequences: Input sequences
            target_sequences: Target sequences
            verbose: Whether to display detailed logs
        """
        from .agent import TrainingAnt
        
        print(f"\n{'='*60}")
        print(f"Starting training - {len(input_sequences)} sequences")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        num_sequences = len(input_sequences)
        
        global_step = self.training_stats.get('step', 0)
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            indices = np.random.permutation(num_sequences)
            
            epoch_rewards = []
            epoch_losses = []
            epoch_paths = []
            
            if verbose:
                pbar = tqdm(
                    total=num_sequences,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                    unit="seq",
                    ncols=100
                )
            
            for batch_start in range(0, num_sequences, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_sequences)
                batch_indices = indices[batch_start:batch_end]
                
                batch_rewards = []
                batch_paths = []
                
                for ant_id in range(self.num_ants):
                    ant = TrainingAnt(
                        ant_id=ant_id,
                        graph=self.graph,
                        alpha=self.alpha,
                        beta=self.beta,
                        exploration_rate=self.exploration_rate
                    )
                    
                    seq_idx = batch_indices[ant_id % len(batch_indices)]
                    target_seq = target_sequences[seq_idx]
                    
                    reward = ant.train_on_sequence(
                        target_sequence=target_seq,
                        temperature=self.temperature
                    )
                    
                    batch_rewards.append(reward)
                    batch_paths.append(ant.get_path())
                
                self.graph.update_pheromones(batch_paths, batch_rewards)
                
                batch_losses = [1.0 - r for r in batch_rewards]
                
                epoch_rewards.extend(batch_rewards)
                epoch_losses.extend(batch_losses)
                epoch_paths.extend(batch_paths)
                
                global_step += 1
                
                if verbose:
                    avg_batch_loss = np.mean(batch_losses)
                    avg_batch_reward = np.mean(batch_rewards)
                    pbar.set_postfix({
                        'step': global_step,
                        'loss': f'{avg_batch_loss:.4f}',
                        'reward': f'{avg_batch_reward:.4f}'
                    })
                    pbar.update(len(batch_indices))
            
            if verbose:
                pbar.close()
            
            avg_reward = np.mean(epoch_rewards)
            max_reward = np.max(epoch_rewards)
            avg_loss = np.mean(epoch_losses)
            min_loss = np.min(epoch_losses)
            epoch_time = time.time() - epoch_start
            
            self.training_stats['epoch'] = epoch + 1
            self.training_stats['step'] = global_step
            self.training_stats['total_sequences'] += len(epoch_rewards)
            self.training_stats['avg_reward'] = avg_reward
            self.training_stats['best_reward'] = max(self.training_stats['best_reward'], max_reward)
            self.training_stats['avg_loss'] = avg_loss
            self.training_stats['best_loss'] = min(self.training_stats['best_loss'], min_loss)
            self.training_stats['training_time'] = time.time() - start_time
            
            if verbose:
                print(f"\n📊 Epoch {epoch + 1}/{self.num_epochs} - Summary:")
                print(f"   Global step: {global_step}")
                print(f"   Average Loss: {avg_loss:.4f} | Best Loss: {min_loss:.4f}")
                print(f"   Average Reward: {avg_reward:.4f} | Best Reward: {max_reward:.4f}")
                print(f"   Epoch time: {epoch_time:.2f}s")
                print()
            
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.ant")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training completed!")
        print(f"{'='*60}")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Total steps: {global_step}")
        print(f"Best reward: {self.training_stats['best_reward']:.4f}")
        print(f"Best loss: {self.training_stats['best_loss']:.4f}")
        print(f"{'='*60}\n")
    
    def train_background(self,
                        input_sequences: List[List[int]],
                        target_sequences: List[List[int]]) -> None:
        """
        Start training in background (separate thread).
        Allows continuing training during user conversation.
        
        Args:
            input_sequences: Input sequences
            target_sequences: Target sequences
        """
        if self.is_training:
            print("⚠ Training already running in background")
            return
        
        self.is_training = True
        
        def training_worker():
            try:
                self.train(input_sequences, target_sequences, verbose=False)
            except Exception as e:
                print(f"⚠ Error in background training: {e}")
            finally:
                self.is_training = False
        
        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()
        
        print("✓ Training started in background")
    
    def stop_background_training(self) -> None:
        """Stop background training."""
        if self.is_training:
            self.is_training = False
            print("⏸ Stopping background training...")
            if self.training_thread:
                self.training_thread.join(timeout=5)
            print("✓ Training stopped")
    
    def update_from_interaction(self,
                               user_input: str,
                               model_output: str,
                               feedback_score: float = 1.0,
                               full_prompt: Optional[str] = None) -> None:
        """
        Update model based on user interaction (online learning).

        Args:
            user_input: User input
            model_output: Model response
            feedback_score: Response quality rating (0-1)
            full_prompt: Full prompt used for generation (optional)
        """
        if not self.online_learning:
            return

        if full_prompt:
            full_sequence = self.tokenizer.encode(full_prompt + model_output, add_special_tokens=False)
        else:
            input_tokens = self.tokenizer.encode(user_input, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(model_output, add_special_tokens=False)
            full_sequence = input_tokens + output_tokens

        reward = feedback_score * self.online_update_rate

        self.graph.update_pheromones_online(
            sequence=full_sequence,
            reward=reward,
            learning_rate=self.online_update_rate
        )
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        self.graph.save(checkpoint_path)
        
        stats_path = checkpoint_path.replace('.ant', '_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch: {self.training_stats['epoch']}\n")
            f.write(f"Global step: {self.training_stats['step']}\n")
            f.write(f"Sequences: {self.training_stats['total_sequences']}\n")
            f.write(f"Average reward: {self.training_stats['avg_reward']:.4f}\n")
            f.write(f"Best reward: {self.training_stats['best_reward']:.4f}\n")
            f.write(f"Average loss: {self.training_stats['avg_loss']:.4f}\n")
            f.write(f"Best loss: {self.training_stats['best_loss']:.4f}\n")
            f.write(f"Training time: {self.training_stats['training_time']:.2f}s\n")
        
        print(f"✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(checkpoint_path):
            print(f"⚠ Checkpoint does not exist: {filename}")
            return

        self.graph = self.graph.__class__.load(checkpoint_path, use_gpu=self.use_gpu)

        print(f"✓ Checkpoint loaded: {filename}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find latest checkpoint.
        
        Returns:
            Latest checkpoint filename or None
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.ant')]
        
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)))
        
        return checkpoints[-1]
    
    def get_stats(self) -> Dict:
        """
        Return training statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.training_stats.copy()
        stats['graph_stats'] = self.graph.get_statistics()
        stats['is_training'] = self.is_training
        
        return stats
    
    def print_stats(self) -> None:
        """Display training statistics."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("TRAINING STATISTICS")
        print(f"{'='*60}")
        print(f"Epoch: {stats['epoch']}")
        print(f"Global step: {stats['step']}")
        print(f"Processed sequences: {stats['total_sequences']}")
        print(f"\nMetrics:")
        print(f"  Average reward: {stats['avg_reward']:.4f}")
        print(f"  Best reward: {stats['best_reward']:.4f}")
        print(f"  Average loss: {stats['avg_loss']:.4f}")
        print(f"  Best loss: {stats['best_loss']:.4f}")
        print(f"\nTraining time: {stats['training_time']:.2f}s")
        print(f"\nGraph:")
        print(f"  Vocabulary size: {stats['graph_stats']['vocab_size']}")
        print(f"  Updates: {stats['graph_stats']['total_updates']}")
        print(f"  Average pheromone: {stats['graph_stats']['avg_pheromone']:.4f}")
        print(f"{'='*60}\n")
