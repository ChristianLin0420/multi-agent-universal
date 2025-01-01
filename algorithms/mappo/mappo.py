import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np

from ..base import MARLAlgorithm
from .networks import Actor, Critic

class MAPPO(MARLAlgorithm):
    """Multi-Agent PPO algorithm implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MAPPO algorithm.
        
        Args:
            config (Dict[str, Any]): Algorithm configuration
        """
        super().__init__(config)
        
        self.n_agents = config["n_agents"]
        self.obs_dim = config["obs_dim"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        
        # PPO parameters
        self.hidden_dim = config.get("hidden_dim", 64)
        self.lr = config.get("lr", 0.0003)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_param = config.get("clip_param", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epoch = config.get("ppo_epoch", 10)
        self.mini_batch_size = config.get("mini_batch_size", 64)
        
        # Initialize networks
        self._setup_networks()
        
        # Setup optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def _setup_networks(self):
        """Initialize neural networks."""
        self.actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
    
    def select_actions(self, observations: Dict[str, torch.Tensor], 
                      agent_ids: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Select actions for the specified agents.
        
        Args:
            observations (Dict[str, torch.Tensor]): Current observations for each agent
            agent_ids (List[str]): List of agent IDs to select actions for
            
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict]: Selected actions and additional info
        """
        actions = {}
        action_log_probs = {}
        
        for agent_id in agent_ids:
            obs = observations[agent_id].to(self.device)
            
            with torch.no_grad():
                action_logits = self.actor(obs)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
            
            actions[agent_id] = action.cpu()
            action_log_probs[agent_id] = action_log_prob.cpu()
        
        return actions, {"action_log_probs": action_log_probs}
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update the algorithm's parameters using the provided batch of experiences.
        
        Args:
            batch (Dict[str, Any]): Batch of experiences
            
        Returns:
            Dict[str, float]: Dictionary of loss metrics
        """
        # Unpack batch
        obs = batch["observations"]
        actions = batch["actions"]
        old_action_log_probs = batch["action_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        states = batch["states"]
        
        # Convert to tensors and move to device
        obs = {k: v.to(self.device) for k, v in obs.items()}
        actions = {k: v.to(self.device) for k, v in actions.items()}
        old_action_log_probs = {k: v.to(self.device) for k, v in old_action_log_probs.items()}
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        states = states.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epoch):
            # Actor update
            for agent_id in obs.keys():
                action_logits = self.actor(obs[agent_id])
                dist = torch.distributions.Categorical(logits=action_logits)
                
                new_action_log_probs = dist.log_prob(actions[agent_id])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_action_log_probs - old_action_log_probs[agent_id])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                total_actor_loss += actor_loss
                total_entropy += entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)
            total_critic_loss += critic_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        num_updates = self.ppo_epoch * len(obs)
        total_actor_loss /= num_updates
        total_critic_loss /= self.ppo_epoch
        total_entropy /= num_updates
        
        return {
            "actor_loss": total_actor_loss.item(),
            "critic_loss": total_critic_loss.item(),
            "entropy": total_entropy.item()
        }
    
    def save(self, path: str) -> None:
        """Save the algorithm's parameters to disk."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load the algorithm's parameters from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer']) 