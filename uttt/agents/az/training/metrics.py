"""
Metrics logging utilities for AlphaZero training.
"""
import os
import csv
import torch


def log_training_metrics(run_dir: str, epoch: int, epoch_loss: float, policy_loss: float, value_loss: float):
    """
    Log training loss metrics to CSV file.
    
    Args:
        run_dir: Path to the run directory
        epoch: Current epoch number
        epoch_loss: Total epoch loss
        policy_loss: Policy loss component
        value_loss: Value loss component
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    loss_file = os.path.join(metrics_dir, "loss.csv")
    
    # Check if we need to write header
    write_header = not os.path.exists(loss_file)
    
    with open(loss_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "total_loss", "policy_loss", "value_loss"])
        writer.writerow([epoch, epoch_loss, policy_loss, value_loss])


def log_gradient_metrics(run_dir: str, epoch: int, model: torch.nn.Module):
    """
    Log gradient statistics to CSV file.
    
    Args:
        run_dir: Path to the run directory
        epoch: Current epoch number
        model: The neural network model
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    grad_file = os.path.join(metrics_dir, "gradients.csv")
    
    # Check if we need to write header
    write_header = not os.path.exists(grad_file)
    
    with open(grad_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "layer_name", "grad_mean", "grad_std", "grad_min", "grad_max", "grad_norm"])
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                grad_norm = torch.norm(grad).item()
                grad_np = grad.numpy()
                
                writer.writerow([
                    epoch, name,
                    float(grad_np.mean()),
                    float(grad_np.std()),
                    float(grad_np.min()),
                    float(grad_np.max()),
                    grad_norm
                ])


def log_parameter_metrics(run_dir: str, epoch: int, model: torch.nn.Module):
    """
    Log parameter statistics to CSV file.
    
    Args:
        run_dir: Path to the run directory
        epoch: Current epoch number
        model: The neural network model
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    param_file = os.path.join(metrics_dir, "parameters.csv")
    
    # Check if we need to write header
    write_header = not os.path.exists(param_file)
    
    with open(param_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "layer_name", "param_mean", "param_std", "param_min", "param_max", "param_norm"])
        
        for name, param in model.named_parameters():
            param_data = param.detach().cpu()
            param_norm = torch.norm(param_data).item()
            param_np = param_data.numpy()
            
            writer.writerow([
                epoch, name,
                float(param_np.mean()),
                float(param_np.std()),
                float(param_np.min()),
                float(param_np.max()),
                param_norm
            ])