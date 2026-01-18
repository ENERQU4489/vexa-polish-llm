"""
VexaLLM - Text generation and user interaction interface
Uses trained ACO graph to generate responses
"""

import numpy as np
from typing import List, Optional, Dict
import time


class VexaLLM:
    """
    LLM interface using ACO algorithm for text generation.
    Supports chat with user and online learning.
    """
    
    def __init__(self, 
                 graph,
                 tokenizer,
                 engine,
                 config: Dict):
        """
        Initialize LLM interface.
        
        Args:
            graph: AntGraph instance
            tokenizer: VexaTokenizer instance
            engine: VexaEngine instance
            config: Configuration dictionary
        """
        self.graph = graph
        self.tokenizer = tokenizer
        self.engine = engine
        self.config = config
        
        self.temperature = config.get('temperature', 0.8)
        self.max_length = config.get('max_generation_length', 500)
        self.top_k = config.get('top_k', 50)
        self.repetition_penalty = config.get('repetition_penalty', 1.2)
        
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 2.0)
        
        self.conversation_history: List[Dict[str, str]] = []
        
        self.generation_stats = {
            'total_generations': 0,
            'avg_generation_time': 0.0,
            'total_tokens_generated': 0
        }
        
        print("✓ VexaLLM initialized")
    
    def generate(self, 
                prompt: str,
                max_length: Optional[int] = None,
                temperature: Optional[float] = None,
                top_k: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input text (prompt)
            max_length: Maximum length of generated text
            temperature: Generation temperature
            top_k: Top-K sampling
            stop_sequences: List of sequences that stop generation
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if not prompt_tokens:
            return ""
        
        stop_token_ids = []
        if stop_sequences:
            for seq in stop_sequences:
                stop_token_ids.extend(self.tokenizer.encode(seq, add_special_tokens=False))
        
        stop_token_ids.append(self.tokenizer.get_eos_id())
        
        prompt_length = len(prompt_tokens)
        
        generated_tokens = self._generate_tokens(
            prompt_tokens=prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=stop_token_ids
        )
        
        new_tokens = generated_tokens[prompt_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens_generated'] += len(generated_tokens)
        
        n = self.generation_stats['total_generations']
        self.generation_stats['avg_generation_time'] = (
            (self.generation_stats['avg_generation_time'] * (n - 1) + generation_time) / n
        )
        
        return generated_text
    
    def _generate_tokens(self,
                        prompt_tokens: List[int],
                        max_length: int,
                        temperature: float,
                        top_k: int,
                        stop_token_ids: List[int]) -> List[int]:
        """
        Generate tokens using ACO.
        
        Args:
            prompt_tokens: Prompt tokens
            max_length: Maximum length
            temperature: Temperature
            top_k: Top-K sampling
            stop_token_ids: Stop tokens
            
        Returns:
            List of generated tokens
        """
        generated = prompt_tokens.copy()
        generated_set = set(generated)
        
        for _ in range(max_length):
            current_token = generated[-1]
            
            probs = self.graph.get_transition_probabilities(
                current_token=current_token,
                alpha=self.alpha,
                beta=self.beta
            )
            
            if self.repetition_penalty != 1.0:
                for token_id in generated_set:
                    if token_id < len(probs):
                        probs[token_id] /= self.repetition_penalty
                
                probs = probs / probs.sum()
            
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / probs.sum()
            
            if top_k > 0 and top_k < len(probs):
                top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
                top_k_probs = probs[top_k_indices]
                top_k_probs = top_k_probs / top_k_probs.sum()
                
                next_token = np.random.choice(top_k_indices, p=top_k_probs)
            else:
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except ValueError:
                    next_token = np.argmax(probs)
            
            generated.append(next_token)
            generated_set.add(next_token)
            
            if next_token in stop_token_ids:
                break
        
        return generated
    
    def chat(self, user_input: str, learn_from_interaction: bool = True) -> str:
        """
        Conduct conversation with user.
        
        Args:
            user_input: User input
            learn_from_interaction: Whether to learn from interaction
            
        Returns:
            Model response
        """
        context = self._build_context()

        if context:
            full_prompt = context + f"\nUser: {user_input}\nAssistant: "
        else:
            full_prompt = f"User: {user_input}\nAssistant: "

        response = self.generate(
            prompt=full_prompt,
            stop_sequences=["\n\nUser:", "\nAssistant:"]
        )
        
        response = response.strip()
        
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()
        
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        if learn_from_interaction and self.config.get('online_learning', True):
            self.engine.update_from_interaction(
                user_input=user_input,
                model_output=response,
                feedback_score=1.0,
                full_prompt=full_prompt
            )
        
        return response
    
    def _build_context(self, max_history: int = 5) -> str:
        """
        Build context from conversation history.
        
        Args:
            max_history: Maximum number of previous exchanges
            
        Returns:
            Context as string
        """
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_history * 2:]
        
        context_parts = []
        for entry in recent_history:
            if entry['role'] == 'user':
                context_parts.append(f"User: {entry['content']}")
            else:
                context_parts.append(f"Assistant: {entry['content']}")
        
        context = "\n".join(context_parts)

        return context
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def provide_feedback(self, rating: float) -> None:
        """
        Allow user to rate last response.
        
        Args:
            rating: Rating 0-1 (0 = bad, 1 = excellent)
        """
        if len(self.conversation_history) < 2:
            print("⚠ No last interaction to rate")
            return
        
        last_user = self.conversation_history[-2]['content']
        last_assistant = self.conversation_history[-1]['content']
        
        self.engine.update_from_interaction(
            user_input=last_user,
            model_output=last_assistant,
            feedback_score=rating
        )
        
        print(f"✓ Feedback saved: {rating:.2f}")
    
    def get_stats(self) -> Dict:
        """
        Return generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'generation_stats': self.generation_stats.copy(),
            'conversation_length': len(self.conversation_history),
            'engine_stats': self.engine.get_stats()
        }
    
    def print_stats(self) -> None:
        """Display statistics."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"Generated responses: {stats['generation_stats']['total_generations']}")
        print(f"Generated tokens: {stats['generation_stats']['total_tokens_generated']}")
        print(f"Average generation time: {stats['generation_stats']['avg_generation_time']:.3f}s")
        print(f"Conversation length: {stats['conversation_length']} exchanges")
        print(f"{'='*60}\n")
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history to file.
        
        Args:
            filepath: File path
        """
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Conversation saved: {filepath}")
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation history from file.
        
        Args:
            filepath: File path
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        
        print(f"✓ Conversation loaded: {filepath} ({len(self.conversation_history)} exchanges)")
