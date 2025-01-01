import numpy as np
import torch
from typing import Dict, List, Any

class ReplayBuffer:
    """Experience replay buffer for QMIX."""
    
    def __init__(self, capacity: int, n_agents: int):
        """Initialize replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
            n_agents (int): Number of agents
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.position = 0
        self.size = 0
        
        # Storage for experiences
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.states = []
        self.next_states = []
    
    def push(self, 
             observations: Dict[str, np.ndarray],
             actions: Dict[str, np.ndarray],
             rewards: Dict[str, float],
             next_observations: Dict[str, np.ndarray],
             dones: Dict[str, bool],
             state: np.ndarray,
             next_state: np.ndarray):
        """Store a transition in the buffer.
        
        Args:
            observations: Current observations for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_observations: Next observations for each agent
            dones: Done flags for each agent
            state: Global state
            next_state: Next global state
        """
        if len(self.observations) < self.capacity:
            self.observations.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_observations.append(None)
            self.dones.append(None)
            self.states.append(None)
            self.next_states.append(None)
        
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_observations[self.position] = next_observations
        self.dones[self.position] = dones
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Dict containing batched transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert experiences to tensors
        batch = {
            "observations": {},
            "actions": {},
            "rewards": torch.zeros(batch_size, self.n_agents),
            "next_observations": {},
            "dones": torch.zeros(batch_size, dtype=torch.bool),
            "states": torch.zeros(batch_size, self.states[0].shape[0]),
            "next_states": torch.zeros(batch_size, self.states[0].shape[0])
        }
        
        for i, idx in enumerate(indices):
            # Process observations and actions for each agent
            for agent_id in self.observations[idx].keys():
                if agent_id not in batch["observations"]:
                    batch["observations"][agent_id] = torch.zeros(
                        batch_size, *self.observations[idx][agent_id].shape
                    )
                    batch["next_observations"][agent_id] = torch.zeros(
                        batch_size, *self.next_observations[idx][agent_id].shape
                    )
                    batch["actions"][agent_id] = torch.zeros(
                        batch_size, dtype=torch.int64
                    )
                
                batch["observations"][agent_id][i] = torch.FloatTensor(
                    self.observations[idx][agent_id]
                )
                batch["next_observations"][agent_id][i] = torch.FloatTensor(
                    self.next_observations[idx][agent_id]
                )
                batch["actions"][agent_id][i] = torch.LongTensor(
                    [self.actions[idx][agent_id]]
                )
            
            # Process rewards and dones
            for j, agent_id in enumerate(self.rewards[idx].keys()):
                batch["rewards"][i, j] = self.rewards[idx][agent_id]
            
            batch["dones"][i] = any(self.dones[idx].values())
            
            # Process states
            batch["states"][i] = torch.FloatTensor(self.states[idx])
            batch["next_states"][i] = torch.FloatTensor(self.next_states[idx])
        
        return batch
    
    def __len__(self) -> int:
        return self.size 