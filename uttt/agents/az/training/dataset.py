"""
Dataset utilities for AlphaZero training.
"""
from typing import List
import torch
from torch.utils.data import Dataset

from uttt.agents.az.self_play import TrainingExample


class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training examples."""
    
    def __init__(self, training_examples: List[TrainingExample]):
        self.examples = training_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'state': torch.FloatTensor(example.state),
            'policy': torch.FloatTensor(example.policy),
            'value': torch.FloatTensor([example.value])
        }