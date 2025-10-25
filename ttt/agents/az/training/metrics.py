"""
Metrics logging utilities for AlphaZero training.
"""
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def calculate_dead_relu_percentage(model: torch.nn.Module, data_loader, device: str, max_batches: int = 5) -> float:
    """
    Calculate the percentage of dead ReLUs in the model.
    
    A ReLU is considered "dead" if it never activates (always outputs 0)
    across multiple batches of inputs.
    
    Args:
        model: The neural network model
        data_loader: DataLoader with training data
        device: Device to run computations on
        max_batches: Maximum number of batches to use for calculation
        
    Returns:
        Overall percentage of dead ReLUs across all ReLU layers
    """
    model.eval()
    
    # Dictionary to store activations for each ReLU layer
    relu_activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.ReLU):
                # Store whether each neuron activated (output > 0) for this batch
                activated = (output > 0).detach().cpu()
                
                if name not in relu_activations:
                    relu_activations[name] = []
                relu_activations[name].append(activated)
        return hook
    
    # Register hooks for all ReLU layers
    hooks = []
    relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
            relu_count += 1
    
    if relu_count == 0:
        # No ReLU layers found
        for hook in hooks:
            hook.remove()
        return 0.0
    
    # Run forward passes to collect activations
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    states = batch['state'].to(device)
                else:
                    states = batch[0].to(device)  # Assume first element is input
                
                # Forward pass to trigger hooks
                _ = model(states)
        
        # Calculate dead ReLU statistics
        total_dead_neurons = 0
        total_neurons = 0
        
        for layer_name, activations_list in relu_activations.items():
            if not activations_list:
                continue
                
            # Stack all batches: (num_batches, batch_size, ...)
            all_activations = torch.stack(activations_list, dim=0)
            
            # Check which neurons NEVER activated across all batches and samples
            # Shape: (num_batches, batch_size, ...) -> flatten last dims -> (num_batches * batch_size, num_neurons)
            flattened = all_activations.view(all_activations.size(0) * all_activations.size(1), -1)
            
            # A neuron is dead if it never activated across all samples
            never_activated = (flattened.sum(dim=0) == 0)  # Sum over all samples
            
            dead_count = never_activated.sum().item()
            neuron_count = never_activated.numel()
            
            total_dead_neurons += dead_count
            total_neurons += neuron_count
        
        # Calculate overall percentage
        dead_relu_percentage = (total_dead_neurons / total_neurons * 100) if total_neurons > 0 else 0.0
        
    finally:
        # Always remove hooks
        for hook in hooks:
            hook.remove()
        
        # Set model back to training mode
        model.train()
    
    return dead_relu_percentage


def log_training_metrics(run_dir: str, epoch: int, epoch_loss: float, policy_loss: float, value_loss: float, dead_relu_pct: float = None):
    """
    Log training loss metrics to CSV file.
    
    Args:
        run_dir: Path to the run directory
        epoch: Current epoch number
        epoch_loss: Total epoch loss
        policy_loss: Policy loss component
        value_loss: Value loss component
        dead_relu_pct: Percentage of dead ReLUs (optional)
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)  # Ensure directory exists
    loss_file = os.path.join(metrics_dir, "loss.csv")
    
    # Check if we need to write header
    write_header = not os.path.exists(loss_file)
    
    with open(loss_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            if dead_relu_pct is not None:
                writer.writerow(["epoch", "total_loss", "policy_loss", "value_loss", "dead_relu_pct"])
            else:
                writer.writerow(["epoch", "total_loss", "policy_loss", "value_loss"])
        
        if dead_relu_pct is not None:
            writer.writerow([epoch, epoch_loss, policy_loss, value_loss, dead_relu_pct])
        else:
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
    os.makedirs(metrics_dir, exist_ok=True)  # Ensure directory exists
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
    os.makedirs(metrics_dir, exist_ok=True)  # Ensure directory exists
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