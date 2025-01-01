import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MLPBase(nn.Module):
    """Base MLP network for both actor and critic."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Initialize the base network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.network(x)
        x = self.layer_norm(x)
        return x

class Actor(nn.Module):
    """MAPPO actor network."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize the actor network.
        
        Args:
            obs_dim (int): Observation dimension
            action_dim (int): Action dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        self.base = MLPBase(obs_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network.
        
        Args:
            obs (torch.Tensor): Observation tensor
            
        Returns:
            torch.Tensor: Action logits
        """
        features = self.base(obs)
        action_logits = self.action_head(features)
        return action_logits

class Critic(nn.Module):
    """MAPPO centralized critic network."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """Initialize the critic network.
        
        Args:
            state_dim (int): Global state dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        self.base = MLPBase(state_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.
        
        Args:
            state (torch.Tensor): Global state tensor
            
        Returns:
            torch.Tensor: Value estimate
        """
        features = self.base(state)
        value = self.value_head(features)
        return value 