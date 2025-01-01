import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import defaultdict

class Buffer:
    """Universal buffer for both value-based and policy-based methods."""
    
    def __init__(self, capacity: int, n_agents: int, is_episodic: bool = False):
        """Initialize buffer.
        
        Args:
            capacity (int): Maximum number of transitions/episodes to store
            n_agents (int): Number of agents
            is_episodic (bool): Whether to store full episodes (for PPO) or transitions (for QMIX)
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.is_episodic = is_episodic
        self.position = 0
        self.size = 0
        
        # Storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.states = []
        self.next_states = []
        
        # Additional storage for policy-based methods
        if is_episodic:
            self.action_log_probs = []
            self.values = []
            self.returns = []
            self.advantages = []
    
    def push(self, 
             observations: Dict[str, np.ndarray],
             actions: Dict[str, np.ndarray],
             rewards: Dict[str, float],
             next_observations: Dict[str, np.ndarray],
             dones: Dict[str, bool],
             state: Optional[np.ndarray] = None,
             next_state: Optional[np.ndarray] = None,
             action_log_probs: Optional[Dict[str, np.ndarray]] = None,
             values: Optional[Dict[str, np.ndarray]] = None):
        """Store a transition/episode in the buffer.
        
        Args:
            observations: Current observations for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_observations: Next observations for each agent
            dones: Done flags for each agent
            state: Global state (optional)
            next_state: Next global state (optional)
            action_log_probs: Log probabilities of actions (for PPO)
            values: Value estimates (for PPO)
        """
        if len(self.observations) < self.capacity:
            self.observations.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_observations.append(None)
            self.dones.append(None)
            self.states.append(None)
            self.next_states.append(None)
            if self.is_episodic:
                self.action_log_probs.append(None)
                self.values.append(None)
                self.returns.append(None)
                self.advantages.append(None)
        
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_observations[self.position] = next_observations
        self.dones[self.position] = dones
        
        if state is not None:
            self.states[self.position] = state
        if next_state is not None:
            self.next_states[self.position] = next_state
        
        if self.is_episodic:
            self.action_log_probs[self.position] = action_log_probs
            self.values[self.position] = values
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions/episodes.
        
        Args:
            batch_size (int): Number of transitions/episodes to sample
            
        Returns:
            Dict containing batched data
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = defaultdict(dict)
        
        # Basic data present in all algorithms
        for i, idx in enumerate(indices):
            # Process observations and actions for each agent
            for agent_id in self.observations[idx].keys():
                # Initialize tensors if first iteration
                if i == 0:
                    batch["observations"][agent_id] = torch.zeros(
                        batch_size, *self.observations[idx][agent_id].shape
                    )
                    batch["next_observations"][agent_id] = torch.zeros(
                        batch_size, *self.next_observations[idx][agent_id].shape
                    )
                    batch["actions"][agent_id] = torch.zeros(
                        batch_size, dtype=torch.int64
                    )
                
                # Fill tensors
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
            if i == 0:
                batch["rewards"] = torch.zeros(batch_size, self.n_agents)
                batch["dones"] = torch.zeros(batch_size, dtype=torch.bool)
            
            for j, agent_id in enumerate(self.rewards[idx].keys()):
                batch["rewards"][i, j] = self.rewards[idx][agent_id]
            batch["dones"][i] = any(self.dones[idx].values())
            
            # Process states if present
            if self.states[idx] is not None:
                if i == 0:
                    batch["states"] = torch.zeros(batch_size, self.states[idx].shape[0])
                    batch["next_states"] = torch.zeros(batch_size, self.states[idx].shape[0])
                batch["states"][i] = torch.FloatTensor(self.states[idx])
                batch["next_states"][i] = torch.FloatTensor(self.next_states[idx])
            
            # Process policy-based data if present
            if self.is_episodic:
                if i == 0:
                    batch["action_log_probs"] = defaultdict(lambda: torch.zeros(batch_size))
                    batch["values"] = defaultdict(lambda: torch.zeros(batch_size))
                    batch["returns"] = torch.zeros(batch_size, self.n_agents)
                    batch["advantages"] = torch.zeros(batch_size, self.n_agents)
                
                for agent_id in self.action_log_probs[idx].keys():
                    batch["action_log_probs"][agent_id][i] = self.action_log_probs[idx][agent_id]
                    batch["values"][agent_id][i] = self.values[idx][agent_id]
                
                if self.returns[idx] is not None:
                    batch["returns"][i] = torch.FloatTensor(self.returns[idx])
                    batch["advantages"][i] = torch.FloatTensor(self.advantages[idx])
        
        return batch
    
    def compute_returns_and_advantages(self, 
                                     gamma: float = 0.99, 
                                     gae_lambda: float = 0.95):
        """Compute returns and advantages for PPO.
        
        Args:
            gamma (float): Discount factor
            gae_lambda (float): GAE parameter
        """
        if not self.is_episodic:
            raise ValueError("Returns and advantages can only be computed for episodic buffer")
        
        for episode_idx in range(self.size):
            returns = []
            advantages = []
            last_value = 0
            last_advantage = 0
            
            # Reverse iteration through episode
            for t in reversed(range(len(self.rewards[episode_idx]))):
                # Get current reward and value
                reward = self.rewards[episode_idx][t]
                value = self.values[episode_idx][t]
                done = self.dones[episode_idx][t]
                
                # Compute TD error and advantage
                if t == len(self.rewards[episode_idx]) - 1:
                    next_value = 0 if done else last_value
                else:
                    next_value = self.values[episode_idx][t + 1]
                
                delta = reward + gamma * next_value * (1 - done) - value
                advantage = delta + gamma * gae_lambda * (1 - done) * last_advantage
                
                returns.insert(0, advantage + value)
                advantages.insert(0, advantage)
                
                last_advantage = advantage
            
            self.returns[episode_idx] = returns
            self.advantages[episode_idx] = advantages
    
    def __len__(self) -> int:
        return self.size 