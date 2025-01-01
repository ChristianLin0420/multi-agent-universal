import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

class QNetwork(nn.Module):
    """Individual Q-network for each agent."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """Initialize Q-network.
        
        Args:
            input_dim (int): Dimension of input (observation space)
            output_dim (int): Dimension of output (action space)
            hidden_dim (int): Dimension of hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Q-values of shape (batch_size, output_dim)
        """
        return self.network(x)

class MixingNetwork(nn.Module):
    """QMIX mixing network for combining individual Q-values."""
    
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        """Initialize mixing network.
        
        Args:
            n_agents (int): Number of agents
            state_dim (int): Dimension of global state
            hidden_dim (int): Dimension of hidden layers
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        
        # Hypernetwork layers
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mixing network.
        
        Args:
            agent_q_values (torch.Tensor): Individual Q-values of shape 
                                         (batch_size, n_agents)
            states (torch.Tensor): Global states of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Mixed Q-values of shape (batch_size, 1)
        """
        batch_size = agent_q_values.size(0)
        
        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, -1)
        b1 = self.hyper_b1(states).view(-1, 1, -1)
        hidden = F.elu(torch.bmm(agent_q_values.view(-1, 1, self.n_agents), w1) + b1)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, -1, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        
        # Ensure positive weights for monotonicity
        w1 = torch.abs(w1)
        w2 = torch.abs(w2)
        
        # Output
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1) 