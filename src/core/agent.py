"""
TrainingAnt - Class representing an ant in ACO algorithm
Ant moves through graph, choosing next tokens according to P_ij formula
"""

import numpy as np
from typing import List, Optional


class TrainingAnt:
    """
    Training ant moving through token graph.
    Chooses next tokens based on pheromones and heuristics.
    """
    
    def __init__(self, 
                 ant_id: int,
                 graph,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 exploration_rate: float = 0.1):
        """
        Initialize ant.
        
        Args:
            ant_id: Unique ant identifier
            graph: AntGraph instance
            alpha: Pheromone influence on path choice
            beta: Heuristic influence on path choice
            exploration_rate: Exploration probability (epsilon-greedy)
        """
        self.ant_id = ant_id
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.exploration_rate = exploration_rate
        
        self.path: List[int] = []
        self.current_position: Optional[int] = None
    
    def reset(self, start_token: int) -> None:
        """
        Reset ant to starting position.
        
        Args:
            start_token: Starting token
        """
        self.path = [start_token]
        self.current_position = start_token
    
    def choose_next_token(self, 
                         temperature: float = 1.0,
                         valid_tokens: Optional[np.ndarray] = None) -> int:
        """
        Choose next token based on ACO formula.
        P_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        
        Args:
            temperature: Sampling temperature (higher = more random)
            valid_tokens: Optional mask of allowed tokens
            
        Returns:
            Chosen token
        """
        if self.current_position is None:
            raise ValueError("Ant not initialized. Use reset().")
        
        if np.random.random() < self.exploration_rate:
            if valid_tokens is not None:
                valid_indices = np.where(valid_tokens > 0)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(0, self.graph.vocab_size)
        
        probs = self.graph.get_transition_probabilities(
            current_token=self.current_position,
            alpha=self.alpha,
            beta=self.beta,
            valid_tokens=valid_tokens
        )
        
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()
        
        try:
            next_token = np.random.choice(self.graph.vocab_size, p=probs)
        except ValueError:
            next_token = np.random.randint(0, self.graph.vocab_size)
        
        return next_token
    
    def move(self, next_token: int) -> None:
        """
        Move ant to next token.
        
        Args:
            next_token: Target token
        """
        self.path.append(next_token)
        self.current_position = next_token
    
    def generate_sequence(self, 
                         start_token: int,
                         max_length: int,
                         temperature: float = 1.0,
                         stop_tokens: Optional[List[int]] = None) -> List[int]:
        """
        Generate token sequence.
        
        Args:
            start_token: Starting token
            max_length: Maximum sequence length
            temperature: Generation temperature
            stop_tokens: List of tokens that stop generation
            
        Returns:
            Generated token sequence
        """
        self.reset(start_token)
        
        if stop_tokens is None:
            stop_tokens = []
        
        for _ in range(max_length - 1):
            next_token = self.choose_next_token(temperature=temperature)
            self.move(next_token)
            
            if next_token in stop_tokens:
                break
        
        return self.path
    
    def train_on_sequence(self, 
                         target_sequence: List[int],
                         temperature: float = 1.0) -> float:
        """
        Train ant on given target sequence.
        Calculate reward based on agreement with target.
        
        Args:
            target_sequence: Target sequence
            temperature: Generation temperature
            
        Returns:
            Reward for this path
        """
        if len(target_sequence) < 2:
            return 0.0
        
        generated = self.generate_sequence(
            start_token=target_sequence[0],
            max_length=len(target_sequence),
            temperature=temperature
        )
        
        reward = self.calculate_reward(generated, target_sequence)
        
        return reward
    
    def calculate_reward(self, 
                        generated: List[int], 
                        target: List[int]) -> float:
        """
        Calculate reward based on agreement between generated and target sequences.
        
        Args:
            generated: Generated sequence
            target: Target sequence
            
        Returns:
            Reward (0-1, where 1 = perfect match)
        """
        min_length = min(len(generated), len(target))
        
        if min_length == 0:
            return 0.0
        
        correct = sum(1 for i in range(min_length) if generated[i] == target[i])
        
        accuracy = correct / len(target)
        length_bonus = min_length / len(target)
        
        length_penalty = 1.0
        if len(generated) > len(target):
            length_penalty = 0.9
        
        reward = (accuracy * 0.7 + length_bonus * 0.3) * length_penalty
        
        return reward
    
    def get_path(self) -> List[int]:
        """
        Return traveled path.
        
        Returns:
            List of tokens in path
        """
        return self.path.copy()
    
    def get_path_length(self) -> int:
        """
        Return traveled path length.
        
        Returns:
            Path length
        """
        return len(self.path)
    
    def __repr__(self) -> str:
        """Text representation of ant."""
        return f"TrainingAnt(id={self.ant_id}, position={self.current_position}, path_length={len(self.path)})"
