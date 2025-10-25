"""
AlphaZero loss function implementation.
Combines policy (cross-entropy) and value (MSE) losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero loss function that combines policy and value losses.
    
    Loss = Policy Loss + Value Loss
    - Policy Loss: Cross-entropy between predicted policy and MCTS visit counts
    - Value Loss: Mean squared error between predicted value and game outcome
    """
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0):
        """
        Initialize AlphaZero loss function.
        
        Args:
            policy_weight: Weight for policy loss component
            value_weight: Weight for value loss component
        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
    
    def forward(
        self,
        pred_policy: torch.Tensor,
        target_policy: torch.Tensor,
        pred_value: torch.Tensor,
        target_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute AlphaZero loss.
        
        Args:
            pred_policy: Predicted policy logits from network [batch_size, 9]
            target_policy: Target policy from MCTS [batch_size, 9]
            pred_value: Predicted value from network [batch_size]
            target_value: Target value (game outcome) [batch_size]
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Policy loss: Cross-entropy between predicted and target policy
        # Apply log_softmax to predictions and use target as probabilities
        log_pred_policy = F.log_softmax(pred_policy, dim=1)
        policy_loss = -torch.sum(target_policy * log_pred_policy, dim=1).mean()
        
        # Value loss: Mean squared error between predicted and target value
        value_loss = F.mse_loss(pred_value, target_value)
        
        # Total loss: Weighted sum of policy and value losses
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss)
        
        return total_loss, policy_loss, value_loss


def test_alphazero_loss():
    """Test function for AlphaZero loss."""
    batch_size = 4
    action_size = 9
    
    # Create sample data with requires_grad=True for predictions
    pred_policy = torch.randn(batch_size, action_size, requires_grad=True)  # Raw logits
    target_policy = torch.softmax(torch.randn(batch_size, action_size), dim=1)  # Normalized probabilities
    pred_value = torch.randn(batch_size, requires_grad=True)
    target_value = torch.tensor([-1.0, 0.0, 1.0, 1.0])  # Game outcomes
    
    # Test loss function
    loss_fn = AlphaZeroLoss()
    total_loss, policy_loss, value_loss = loss_fn(pred_policy, target_policy, pred_value, target_value)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    
    # Test backward pass
    total_loss.backward()
    print("Backward pass successful!")
    
    # Check that gradients were computed
    assert pred_policy.grad is not None, "Policy gradients should be computed"
    assert pred_value.grad is not None, "Value gradients should be computed"
    
    assert total_loss.item() > 0, "Loss should be positive"
    assert policy_loss.item() > 0, "Policy loss should be positive"
    assert value_loss.item() >= 0, "Value loss should be non-negative"
    
    print("✅ AlphaZero loss test passed!")


if __name__ == "__main__":
    test_alphazero_loss()
