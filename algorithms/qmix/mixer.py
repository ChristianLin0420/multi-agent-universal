import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QMixer(nn.Module):
    """QMIX mixer network for combining individual Q-values."""
    
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        """Initialize the mixer network.
        
        Args:
            n_agents (int): Number of agents
            state_dim (int): Dimension of global state
            mixing_embed_dim (int): Dimension of mixing network
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Layers to generate hypernetwork weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, self.n_agents * self.mixing_embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim)
        )
        
        # State-dependent biases
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mixer network.
        
        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global states [batch_size, state_dim]
            
        Returns:
            Mixed Q-values [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = b1.view(-1, 1, self.mixing_embed_dim)
        
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.mixing_embed_dim, 1)
        b2 = b2.view(-1, 1, 1)
        
        # Output
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1) 