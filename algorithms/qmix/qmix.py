import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np

from ..base import MARLAlgorithm
from .networks import QNetwork
from .mixer import QMixer
from ..common.buffer import Buffer
from ...utils.config import Config

class QMIX(MARLAlgorithm):
    """QMIX algorithm implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize QMIX algorithm.
        
        Args:
            config (Dict[str, Any]): Algorithm configuration
        """
        super().__init__(config)
        
        self.n_agents = config["n_agents"]
        self.obs_dim = config["obs_dim"]
        self.state_dim = config["state_dim"]
        self.n_actions = config["n_actions"]
        
        # Network parameters
        self.hidden_dim = config.get("hidden_dim", 64)
        self.lr = config.get("lr", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.target_update_interval = config.get("target_update_interval", 200)
        self.epsilon = config.get("epsilon", 0.1)
        
        # Initialize networks
        self._setup_networks()
        
        # Initialize buffer
        buffer_size = config.get("buffer_size", 5000)
        self.buffer = Buffer(buffer_size, self.n_agents, is_episodic=False)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.q_networks.parameters()) + 
            list(self.mixer.parameters()), 
            lr=self.lr
        )
    
    def _setup_networks(self):
        """Initialize Q-networks and mixer network."""
        # Q-networks for each agent
        self.q_networks = nn.ModuleList([
            QNetwork(self.obs_dim, self.n_actions, self.hidden_dim).to(self.device)
            for _ in range(self.n_agents)
        ])
        
        # Target Q-networks
        self.target_q_networks = nn.ModuleList([
            QNetwork(self.obs_dim, self.n_actions, self.hidden_dim).to(self.device)
            for _ in range(self.n_agents)
        ])
        
        # Mixer networks
        self.mixer = QMixer(
            self.n_agents, 
            self.state_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_mixer = QMixer(
            self.n_agents, 
            self.state_dim, 
            self.hidden_dim
        ).to(self.device)
        
        # Initialize target networks
        self.update_target_networks()
    
    def select_actions(self, 
                      observations: Dict[str, np.ndarray],
                      agent_ids: List[str],
                      explore: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Select actions using epsilon-greedy policy."""
        actions = {}
        q_values = {}
        
        for i, agent_id in enumerate(agent_ids):
            obs = torch.FloatTensor(observations[agent_id]).to(self.device)
            
            with torch.no_grad():
                q_value = self.q_networks[i](obs)
            
            if explore and np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = q_value.argmax().item()
            
            actions[agent_id] = action
            q_values[agent_id] = q_value.cpu().numpy()
        
        return actions, {"q_values": q_values}
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update QMIX networks."""
        # Convert batch to tensors
        obs = {k: v.to(self.device) for k, v in batch["observations"].items()}
        next_obs = {k: v.to(self.device) for k, v in batch["next_observations"].items()}
        actions = {k: v.to(self.device) for k, v in batch["actions"].items()}
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        
        # Compute current Q-values
        current_q_values = []
        for i, agent_id in enumerate(obs.keys()):
            q_values = self.q_networks[i](obs[agent_id])
            q_value = q_values.gather(1, actions[agent_id].unsqueeze(1))
            current_q_values.append(q_value)
        
        current_q_values = torch.cat(current_q_values, dim=1)
        
        # Mix current Q-values
        mixed_q_values = self.mixer(current_q_values, states)
        
        # Compute target Q-values
        target_q_values = []
        for i, agent_id in enumerate(next_obs.keys()):
            target_q = self.target_q_networks[i](next_obs[agent_id])
            target_q_values.append(target_q.max(1, keepdim=True)[0])
        
        target_q_values = torch.cat(target_q_values, dim=1)
        
        # Mix target Q-values
        mixed_target_q = self.target_mixer(target_q_values, next_states)
        
        # Compute targets
        targets = rewards.sum(dim=1, keepdim=True) + \
                 self.gamma * (1 - dones.float()).unsqueeze(1) * mixed_target_q
        
        # Compute loss
        loss = F.mse_loss(mixed_q_values, targets.detach())
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + 
            list(self.mixer.parameters()), 
            10.0
        )
        self.optimizer.step()
        
        # Update target networks if needed
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_networks()
        
        return {
            "loss": loss.item(),
            "q_value": mixed_q_values.mean().item(),
            "target": targets.mean().item()
        }
    
    def update_target_networks(self):
        """Update target networks with current network parameters."""
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            target_q_net.load_state_dict(q_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def save(self, path: str) -> None:
        """Save algorithm state."""
        checkpoint = {
            'q_networks': [net.state_dict() for net in self.q_networks],
            'mixer': self.mixer.state_dict(),
            'target_q_networks': [net.state_dict() for net in self.target_q_networks],
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load algorithm state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(state_dict)
        
        self.mixer.load_state_dict(checkpoint['mixer'])
        
        for i, state_dict in enumerate(checkpoint['target_q_networks']):
            self.target_q_networks[i].load_state_dict(state_dict)
        
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = checkpoint['train_steps'] 