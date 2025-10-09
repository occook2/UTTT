"""
Opening position generator for tournament evaluation.
"""
import random
from typing import List
from uttt.env.state import UTTTEnv


def generate_random_opening(n_moves: int = 1) -> List[int]:
    """
    Generate a random opening sequence.
    
    Args:
        n_moves: Number of random moves to make
        
    Returns:
        List of actions representing the opening
    """
    env = UTTTEnv()
    opening_moves = []
    
    for _ in range(n_moves):
        if env.terminated:
            break
        
        legal_actions = env.legal_actions()
        if not legal_actions:
            break
            
        action = random.choice(legal_actions)
        opening_moves.append(action)
        
        env.step(action)
    
    return opening_moves


def generate_opening_book(n_openings: int = 20, moves_per_opening: int = 1) -> List[List[int]]:
    """
    Generate a book of random opening positions.
    
    Args:
        n_openings: Number of different openings to generate
        moves_per_opening: Number of moves in each opening
        
    Returns:
        List of opening sequences
    """
    openings = []
    attempts = 0
    max_attempts = n_openings * 10  # Avoid infinite loops
    
    while len(openings) < n_openings and attempts < max_attempts:
        opening = generate_random_opening(moves_per_opening)
        
        # Only add non-empty openings and avoid duplicates
        if opening and opening not in openings:
            openings.append(opening)
        
        attempts += 1
    
    return openings


def apply_opening_to_env(env: UTTTEnv, opening_moves: List[int]) -> UTTTEnv:
    """
    Apply an opening sequence to an environment.
    
    Args:
        env: Starting environment
        opening_moves: List of moves to apply
        
    Returns:
        Environment after applying the opening
    """
    for move in opening_moves:
        if env.terminated:
            break
            
        step_result = env.step(move)
        # Note: env.step() modifies env in-place, so we don't need to reassign
    
    return env


# Pre-generated opening book for consistency
DEFAULT_OPENING_BOOK = [
    [40],  # Center of center board
    [0],   # Top-left corner
    [8],   # Top-right corner  
    [72],  # Bottom-left corner
    [80],  # Bottom-right corner
    [4],   # Top-center
    [36],  # Left-center
    [44],  # Right-center
    [76],  # Bottom-center
    [13],  # Random position
    [31],  # Random position
    [49],  # Random position
    [67],  # Random position
    [22],  # Random position
    [58],  # Random position
]