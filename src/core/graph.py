"""
AntGraph - Graph with pheromone (τ) and heuristic (η) matrices
Represents state space for ACO algorithm
"""

import numpy as np
import json
import base64
import os
from typing import Optional, Tuple

# CUDA support
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class AntGraph:
    """
    Graph representing state space for ant colony algorithm.
    Stores pheromone (τ) and heuristic (η) matrices.
    """
    
    def __init__(self,
                 vocab_size: int,
                 tau_init: float = 0.1,
                 tau_min: float = 0.01,
                 tau_max: float = 10.0,
                 rho: float = 0.1,
                 use_gpu: bool = False):
        """
        Initialize graph.

        Args:
            vocab_size: Vocabulary size (number of nodes)
            tau_init: Initial pheromone value
            tau_min: Minimum pheromone value
            tau_max: Maximum pheromone value
            rho: Pheromone evaporation coefficient (0-1)
            use_gpu: Whether to use GPU acceleration
        """
        self.vocab_size = vocab_size
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.use_gpu = use_gpu and CUDA_AVAILABLE

        if self.use_gpu:
            print("✓ CUDA support enabled for AntGraph")
            self._init_cuda()
        else:
            print("ℹ Using CPU for AntGraph operations")

        self.pheromones = np.full((vocab_size, vocab_size), tau_init, dtype=np.float32)
        self.heuristics = np.ones((vocab_size, vocab_size), dtype=np.float32)
        self.transition_counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        self.total_updates = 0

        if self.use_gpu:
            self.pheromones_gpu = cuda.mem_alloc(self.pheromones.nbytes)
            self.heuristics_gpu = cuda.mem_alloc(self.heuristics.nbytes)
            cuda.memcpy_htod(self.pheromones_gpu, self.pheromones)
            cuda.memcpy_htod(self.heuristics_gpu, self.heuristics)

    def _init_cuda(self):
        """Initialize CUDA kernels."""
        # Kernel for computing transition probabilities
        transition_kernel = """
        __global__ void compute_transition_probs(float *pheromones, float *heuristics,
                                                float *probs, int vocab_size, int current_token,
                                                float alpha, float beta, float *valid_mask) {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= vocab_size) return;

            float tau = pheromones[current_token * vocab_size + j];
            float eta = heuristics[current_token * vocab_size + j];
            float mask = valid_mask ? valid_mask[j] : 1.0f;

            float numerator = powf(tau, alpha) * powf(eta, beta) * mask;
            probs[j] = numerator;
        }

        __global__ void normalize_probs(float *probs, int vocab_size, float denominator) {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= vocab_size) return;
            probs[j] /= denominator;
        }
        """

        # Kernel for pheromone evaporation
        evaporation_kernel = """
        __global__ void evaporate_pheromones(float *pheromones, int vocab_size, float evaporation_factor) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (i >= vocab_size || j >= vocab_size) return;

            int idx = i * vocab_size + j;
            pheromones[idx] *= evaporation_factor;
        }
        """

        # Kernel for clipping pheromones
        clip_kernel = """
        __global__ void clip_pheromones(float *pheromones, int vocab_size, float tau_min, float tau_max) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (i >= vocab_size || j >= vocab_size) return;

            int idx = i * vocab_size + j;
            pheromones[idx] = fminf(fmaxf(pheromones[idx], tau_min), tau_max);
        }
        """

        self.mod = SourceModule(transition_kernel + evaporation_kernel + clip_kernel)
        self.compute_transition_probs = self.mod.get_function("compute_transition_probs")
        self.normalize_probs = self.mod.get_function("normalize_probs")
        self.evaporate_pheromones = self.mod.get_function("evaporate_pheromones")
        self.clip_pheromones = self.mod.get_function("clip_pheromones")

    def update_heuristics_from_data(self, sequences: list) -> None:
        """
        Update heuristic matrix based on training data.
        η[i][j] = frequency of transition i->j in training data
        
        Args:
            sequences: List of token sequences
        """
        print("Computing heuristics from training data...")
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_token = sequence[i]
                next_token = sequence[i + 1]
                
                if current_token < self.vocab_size and next_token < self.vocab_size:
                    self.transition_counts[current_token, next_token] += 1
        
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        
        self.heuristics = self.transition_counts.astype(np.float32) / row_sums
        self.heuristics += 0.01

        if self.use_gpu:
            cuda.memcpy_htod(self.heuristics_gpu, self.heuristics)

        print(f"✓ Heuristics updated from {self.transition_counts.sum()} transitions")
    
    def get_transition_probabilities(self,
                                    current_token: int,
                                    alpha: float = 1.0,
                                    beta: float = 2.0,
                                    valid_tokens: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate transition probabilities from current token.
        P[j] = (τ[i][j]^α * η[i][j]^β) / Σ(τ[i][k]^α * η[i][k]^β)

        Args:
            current_token: Current token (node)
            alpha: Pheromone influence
            beta: Heuristic influence
            valid_tokens: Optional mask of allowed tokens

        Returns:
            Probability vector for all possible next tokens
        """
        if current_token >= self.vocab_size:
            probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
            return probs

        if self.use_gpu:
            # CUDA version
            probs_gpu = cuda.mem_alloc(self.vocab_size * 4)  # float32
            valid_mask_gpu = None
            if valid_tokens is not None:
                valid_mask_gpu = cuda.mem_alloc(valid_tokens.nbytes)
                cuda.memcpy_htod(valid_mask_gpu, valid_tokens.astype(np.float32))

            block_size = 256
            grid_size = (self.vocab_size + block_size - 1) // block_size

            self.compute_transition_probs(self.pheromones_gpu, self.heuristics_gpu, probs_gpu,
                                        np.int32(self.vocab_size), np.int32(current_token),
                                        np.float32(alpha), np.float32(beta), valid_mask_gpu,
                                        block=(block_size, 1, 1), grid=(grid_size, 1))

            probs = np.empty(self.vocab_size, dtype=np.float32)
            cuda.memcpy_dtoh(probs, probs_gpu)

            denominator = probs.sum()

            if denominator == 0:
                if valid_tokens is not None:
                    probs = valid_tokens / valid_tokens.sum()
                else:
                    probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
            else:
                self.normalize_probs(probs_gpu, np.int32(self.vocab_size), np.float32(denominator),
                                   block=(block_size, 1, 1), grid=(grid_size, 1))
                cuda.memcpy_dtoh(probs, probs_gpu)

            return probs
        else:
            # CPU version
            tau = self.pheromones[current_token]
            eta = self.heuristics[current_token]

            numerator = np.power(tau, alpha) * np.power(eta, beta)

            if valid_tokens is not None:
                numerator = numerator * valid_tokens

            denominator = numerator.sum()

            if denominator == 0:
                if valid_tokens is not None:
                    probs = valid_tokens / valid_tokens.sum()
                else:
                    probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
            else:
                probs = numerator / denominator

            return probs
    
    def update_pheromones(self, paths: list, rewards: list) -> None:
        """
        Update pheromones based on paths traveled by ants.

        Algorithm:
        1. Evaporation: τ[i][j] = (1 - ρ) * τ[i][j]
        2. Reinforcement: τ[i][j] += Σ(Δτ_k) for each ant k

        Args:
            paths: List of paths (token sequences) traveled by ants
            rewards: List of rewards for each path
        """
        if self.use_gpu:
            # Evaporation on GPU
            block_size = (16, 16, 1)
            grid_size = ((self.vocab_size + 15) // 16, (self.vocab_size + 15) // 16, 1)
            self.evaporate_pheromones(self.pheromones_gpu, np.int32(self.vocab_size),
                                    np.float32(1.0 - self.rho), block=block_size, grid=grid_size)

            # Copy back to CPU for reinforcement (sparse updates)
            cuda.memcpy_dtoh(self.pheromones, self.pheromones_gpu)

        else:
            self.pheromones *= (1.0 - self.rho)

        # Reinforcement on CPU (sparse)
        for path, reward in zip(paths, rewards):
            delta_tau = reward

            for i in range(len(path) - 1):
                current_token = path[i]
                next_token = path[i + 1]

                if current_token < self.vocab_size and next_token < self.vocab_size:
                    self.pheromones[current_token, next_token] += delta_tau

        if self.use_gpu:
            # Clipping on GPU
            self.clip_pheromones(self.pheromones_gpu, np.int32(self.vocab_size),
                               np.float32(self.tau_min), np.float32(self.tau_max),
                               block=block_size, grid=grid_size)
            # Copy back to CPU
            cuda.memcpy_dtoh(self.pheromones, self.pheromones_gpu)
        else:
            self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

        self.total_updates += 1
    
    def update_pheromones_online(self, sequence: list, reward: float, learning_rate: float = 0.5) -> None:
        """
        Update pheromones in online mode (during user conversation).
        
        Args:
            sequence: Token sequence
            reward: Reward for sequence
            learning_rate: Update strength (0-1)
        """
        delta_tau = reward * learning_rate
        
        for i in range(len(sequence) - 1):
            current_token = sequence[i]
            next_token = sequence[i + 1]
            
            if current_token < self.vocab_size and next_token < self.vocab_size:
                self.pheromones[current_token, next_token] *= (1.0 - self.rho * 0.1)
                self.pheromones[current_token, next_token] += delta_tau
        
        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)
    
    def get_statistics(self) -> dict:
        """
        Return graph statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'vocab_size': self.vocab_size,
            'total_updates': self.total_updates,
            'avg_pheromone': float(self.pheromones.mean()),
            'max_pheromone': float(self.pheromones.max()),
            'min_pheromone': float(self.pheromones.min()),
            'total_transitions': int(self.transition_counts.sum())
        }
    
    def save(self, filepath: str) -> None:
        """
        Save graph state to file in .ant format (JSON with base64-encoded arrays).
        
        Args:
            filepath: File path
        """
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        state = {
            'vocab_size': self.vocab_size,
            'tau_init': self.tau_init,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'rho': self.rho,
            'pheromones': base64.b64encode(self.pheromones.tobytes()).decode('ascii'),
            'heuristics': base64.b64encode(self.heuristics.tobytes()).decode('ascii'),
            'transition_counts': base64.b64encode(self.transition_counts.tobytes()).decode('ascii'),
            'total_updates': self.total_updates
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ Graph saved: {filepath} ({file_size_mb:.2f} MB)")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = False) -> 'AntGraph':
        """
        Load graph state from .ant file (JSON with base64-encoded arrays).

        Args:
            filepath: File path
            use_gpu: Whether to use GPU acceleration

        Returns:
            AntGraph instance
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        vocab_size = state['vocab_size']

        graph = cls(
            vocab_size=vocab_size,
            tau_init=state['tau_init'],
            tau_min=state['tau_min'],
            tau_max=state['tau_max'],
            rho=state['rho'],
            use_gpu=use_gpu
        )

        graph.pheromones = np.frombuffer(base64.b64decode(state['pheromones']), dtype=np.float32).reshape((vocab_size, vocab_size))
        graph.heuristics = np.frombuffer(base64.b64decode(state['heuristics']), dtype=np.float32).reshape((vocab_size, vocab_size))
        graph.transition_counts = np.frombuffer(base64.b64decode(state['transition_counts']), dtype=np.int32).reshape((vocab_size, vocab_size))
        graph.total_updates = state['total_updates']

        if graph.use_gpu:
            cuda.memcpy_htod(graph.pheromones_gpu, graph.pheromones)
            cuda.memcpy_htod(graph.heuristics_gpu, graph.heuristics)

        print(f"✓ Graph loaded: {filepath}")
        print(f"  Vocabulary size: {graph.vocab_size}")
        print(f"  Total updates: {graph.total_updates}")

        return graph
