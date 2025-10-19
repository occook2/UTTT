"""
Symmetry augmentation for AlphaZero training data.
Provides 4-fold rotational symmetry by rotating board states and policies.
"""
import numpy as np
from typing import List
from uttt.agents.az.self_play import TrainingExample


def rotate_example(example: TrainingExample, k: int) -> TrainingExample:
    """
    Rotate the board state and policy by 90*k degrees.
    
    Args:
        example: Original training example
        k: Number of 90-degree rotations (0=0°, 1=90°, 2=180°, 3=270°)
        
    Returns:
        New training example with rotated state and policy
    """
    # Rotate the board state (shape: 7, 9, 9) along the last two axes
    rotated_state = np.rot90(example.state, k=k, axes=(1, 2)).copy()
    
    # Rotate the policy (shape: 81) as a 9x9 grid, then flatten
    rotated_policy = np.rot90(example.policy.reshape(9, 9), k=k).flatten().copy()
    
    # Value stays the same (game outcome doesn't change with rotation)
    # Agent value also stays the same (evaluation doesn't change with rotation)
    return TrainingExample(
        state=rotated_state,
        policy=rotated_policy,
        value=example.value,
        agent_value=getattr(example, 'agent_value', 0.0)
    )

def reflect_example(example: TrainingExample) -> TrainingExample:
    # Reflect (mirror) the board state and policy horizontally (axis=2)
    reflected_state = np.flip(example.state, axis=2).copy()
    reflected_policy = np.flip(example.policy.reshape(9, 9), axis=1).flatten().copy()
    return TrainingExample(
        state=reflected_state,
        policy=reflected_policy,
        value=example.value,
        agent_value=getattr(example, 'agent_value', 0.0)
    )

def augment_examples_with_rotations(examples: List[TrainingExample]) -> List[TrainingExample]:
    """
    Create 4x training data by adding 90°, 180°, and 270° rotations.
    
    Args:
        examples: Original training examples
        
    Returns:
        Augmented list with 4x examples: [original, 90°, 180°, 270° rotations]
    """
    aug_examples = []
    ref_examples = []
    # Add all rotations: 0°, 90°, 180°, 270°
    for k in range(4):
        for example in examples:
            rotated_example = rotate_example(example, k)
            aug_examples.append(rotated_example)
            ref_examples.append(reflect_example(rotated_example))
    return aug_examples + ref_examples
