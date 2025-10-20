#!/usr/bin/env python3

"""
Test script to demonstrate checkpoint resuming functionality.
"""

from uttt.agents.az.training.utils import find_latest_checkpoint, list_available_checkpoints

def test_checkpoint_utilities():
    """Test the checkpoint utility functions."""
    print("Testing checkpoint utilities...")
    
    # Test with checkpoints directory
    checkpoints_dir = "checkpoints"
    print(f"\nSearching in: {checkpoints_dir}")
    
    # List available checkpoints
    available = list_available_checkpoints(checkpoints_dir)
    print(f"Available checkpoints: {len(available)}")
    for cp in available[:5]:  # Show first 5
        print(f"  {cp}")
    if len(available) > 5:
        print(f"  ... and {len(available) - 5} more")
    
    # Find latest checkpoint
    latest = find_latest_checkpoint(checkpoints_dir)
    if latest:
        print(f"\nLatest checkpoint: {latest}")
    else:
        print("\nNo checkpoints found")

if __name__ == "__main__":
    test_checkpoint_utilities()