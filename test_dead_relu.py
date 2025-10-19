#!/usr/bin/env python3

"""
Test the dead ReLU percentage calculation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Simple test network with ReLUs
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def test_dead_relu_calculation():
    """Test the dead ReLU calculation function."""
    print("Testing dead ReLU calculation...")
    
    # Import the function
    from uttt.agents.az.training.metrics import calculate_dead_relu_percentage
    
    # Create test network
    net = TestNet()
    
    # Create test data - make some inputs that will cause dead ReLUs
    # Use very negative inputs to force some ReLUs to always output 0
    test_inputs = torch.cat([
        torch.randn(50, 10),  # Normal inputs
        -10 * torch.ones(50, 10)  # Very negative inputs (will kill some ReLUs)
    ])
    test_outputs = torch.randn(100, 1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(test_inputs, test_outputs)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Calculate dead ReLU percentage
    dead_pct = calculate_dead_relu_percentage(net, dataloader, "cpu", max_batches=3)
    
    print(f"Dead ReLU percentage: {dead_pct:.2f}%")
    print("✅ Dead ReLU calculation test completed!")
    
    return dead_pct

if __name__ == "__main__":
    test_dead_relu_calculation()