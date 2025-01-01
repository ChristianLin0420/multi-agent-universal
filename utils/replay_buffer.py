from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', 
    ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'infos'])

class ReplayBuffer:
    """Generic replay buffer for MARL algorithms."""
    
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: Union[int, tuple],
        action_shape: Union[int, tuple],
        device: str = "cpu"
    ):
        """Initialize replay buffer.
        
        Args:
            capacity (int): Buffer capacity
            n_agents (int): Number of agents
            obs_shape (Union[int, tuple]): Shape of observations
            action_shape (Union[int, tuple]): Shape of actions
            device (str): Device to store tensors on
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        
        # Initialize storage
        self.obs = np.zeros((capacity, n_agents, *self._to_tuple(obs_shape)))
        self.actions = np.zeros((capacity, n_agents, *self._to_tuple(action_shape)))
        self.rewards = np.zeros((capacity, n_agents))
        self.next_obs = np.zeros((capacity, n_agents, *self._to_tuple(obs_shape)))
        self.dones = np.zeros((capacity, n_agents), dtype=np.bool)
        
        self.pos = 0
        self.size = 0
    
    def _to_tuple(self, shape: Union[int, tuple]) -> tuple:
        """Convert shape to tuple."""
        return (shape,) if isinstance(shape, int) else shape
    
    def push(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        infos: Optional[Dict[str, Any]] = None
    ):
        """Add transition to buffer.
        
        Args:
            obs (Dict[str, np.ndarray]): Observations for each agent
            actions (Dict[str, np.ndarray]): Actions for each agent
            rewards (Dict[str, float]): Rewards for each agent
            next_obs (Dict[str, np.ndarray]): Next observations for each agent
            dones (Dict[str, bool]): Done flags for each agent
            infos (Optional[Dict[str, Any]]): Additional information
        """
        # Convert dict observations to array
        obs_array = np.stack([obs[f"agent_{i}"] for i in range(self.n_agents)])
        next_obs_array = np.stack([next_obs[f"agent_{i}"] for i in range(self.n_agents)])
        action_array = np.stack([actions[f"agent_{i}"] for i in range(self.n_agents)])
        reward_array = np.array([rewards[f"agent_{i}"] for i in range(self.n_agents)])
        done_array = np.array([dones[f"agent_{i}"] for i in range(self.n_agents)])
        
        # Store transition
        self.obs[self.pos] = obs_array
        self.actions[self.pos] = action_array
        self.rewards[self.pos] = reward_array
        self.next_obs[self.pos] = next_obs_array
        self.dones[self.pos] = done_array
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        to_torch: bool = True
    ) -> Union[Transition, Dict[str, torch.Tensor]]:
        """Sample batch of transitions.
        
        Args:
            batch_size (int): Size of batch to sample
            to_torch (bool): Whether to convert to PyTorch tensors
            
        Returns:
            Union[Transition, Dict[str, torch.Tensor]]: Batch of transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = Transition(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
            infos=None
        )
        
        if to_torch:
            batch = self._to_torch(batch)
        
        return batch
    
    def _to_torch(self, batch: Transition) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors.
        
        Args:
            batch (Transition): Batch of transitions
            
        Returns:
            Dict[str, torch.Tensor]: Batch as tensors
        """
        return {
            'obs': torch.FloatTensor(batch.obs).to(self.device),
            'actions': torch.LongTensor(batch.actions).to(self.device),
            'rewards': torch.FloatTensor(batch.rewards).to(self.device),
            'next_obs': torch.FloatTensor(batch.next_obs).to(self.device),
            'dones': torch.FloatTensor(batch.dones).to(self.device)
        }
    
    def __len__(self) -> int:
        return self.size 