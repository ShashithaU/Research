import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

class DifferentialPrivacyManager:
    """
    Manages differential privacy mechanisms for federated learning
    """
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.1,
                 sample_rate: float = 0.01):
        """
        Initialize DP manager
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise scale relative to sensitivity
            sample_rate: Sampling rate for DP-SGD
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.composition_steps = 0
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"DP Manager initialized with ε={epsilon}, δ={delta}")
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to bound sensitivity
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm before clipping
        """
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients if norm exceeds threshold
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model: nn.Module, device: torch.device):
        """
        Add Gaussian noise to gradients for differential privacy
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        noise_scale = self.noise_multiplier * self.max_grad_norm
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=param.grad.shape,
                    device=device
                )
                param.grad.data.add_(noise)
    
    def add_noise_to_parameters(self, parameters: List[torch.Tensor], 
                               device: torch.device) -> List[torch.Tensor]:
        """
        Add noise to model parameters before sharing
        
        Args:
            parameters: List of model parameters
            device: Device to run on
            
        Returns:
            Noisy parameters
        """
        noise_scale = self.noise_multiplier * self.max_grad_norm
        noisy_params = []
        
        for param in parameters:
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=param.shape,
                device=device
            )
            noisy_param = param + noise
            noisy_params.append(noisy_param)
        
        return noisy_params
    
    def compute_privacy_spent(self, num_steps: int, 
                            batch_size: int, 
                            dataset_size: int) -> Tuple[float, float]:
        """
        Compute privacy spent using RDP composition
        
        Args:
            num_steps: Number of training steps
            batch_size: Batch size used
            dataset_size: Total dataset size
            
        Returns:
            (epsilon_spent, delta_used)
        """
        # Simple composition (for demonstration)
        # In practice, use more sophisticated accounting like RDP
        q = batch_size / dataset_size  # Sampling probability
        
        # Gaussian mechanism privacy cost per step
        epsilon_per_step = (2 * q * num_steps * 
                           (self.noise_multiplier ** -2) * 
                           np.log(1 / self.delta))
        
        self.privacy_spent = min(epsilon_per_step, self.epsilon)
        self.composition_steps += num_steps
        
        self.logger.info(f"Privacy spent: ε={self.privacy_spent:.4f}, "
                        f"steps={self.composition_steps}")
        
        return self.privacy_spent, self.delta
    
    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon - self.privacy_spent)
    
    def should_continue_training(self) -> bool:
        """Check if we can continue training within privacy budget"""
        return self.privacy_spent < self.epsilon

class DPOptimizer:
    """
    Differentially Private optimizer wrapper
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 dp_manager: DifferentialPrivacyManager,
                 device: torch.device):
        self.optimizer = optimizer
        self.dp_manager = dp_manager
        self.device = device
        self.step_count = 0
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def step(self, model: nn.Module) -> float:
        """
        Perform DP optimization step
        
        Args:
            model: Model to optimize
            
        Returns:
            Gradient norm before clipping
        """
        # Clip gradients
        grad_norm = self.dp_manager.clip_gradients(model)
        
        # Add noise to gradients
        self.dp_manager.add_noise_to_gradients(model, self.device)
        
        # Perform optimization step
        self.optimizer.step()
        
        self.step_count += 1
        
        return grad_norm

def create_dp_noise(shape: tuple, 
                   sensitivity: float,
                   epsilon: float,
                   device: torch.device) -> torch.Tensor:
    """
    Create Laplace noise for differential privacy
    
    Args:
        shape: Shape of noise tensor
        sensitivity: Global sensitivity
        epsilon: Privacy parameter
        device: Device to create tensor on
        
    Returns:
        Noise tensor
    """
    scale = sensitivity / epsilon
    noise = torch.empty(shape, device=device)
    
    # Generate Laplace noise
    uniform = torch.rand(shape, device=device) - 0.5
    noise = -scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
    
    return noise

# Example usage and testing
if __name__ == "__main__":
    # Test DP manager
    dp_manager = DifferentialPrivacyManager(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.1
    )
    
    # Create dummy model
    model = nn.Linear(10, 1)
    
    # Create dummy gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    
    # Test gradient clipping
    grad_norm = dp_manager.clip_gradients(model)
    print(f"Gradient norm: {grad_norm}")
    
    # Test noise addition
    dp_manager.add_noise_to_gradients(model, torch.device('cpu'))
    
    # Test privacy accounting
    epsilon_spent, delta_used = dp_manager.compute_privacy_spent(
        num_steps=100,
        batch_size=32,
        dataset_size=1000
    )
    
    print(f"Privacy spent: ε={epsilon_spent:.4f}, δ={delta_used}")
    print("DP utilities created successfully!")