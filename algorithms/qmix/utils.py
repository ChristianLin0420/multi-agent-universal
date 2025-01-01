import torch
import numpy as np
from typing import Dict, List, Tuple

def build_td_lambda_targets(rewards: torch.Tensor,
                          terminated: torch.Tensor,
                          mask: torch.Tensor,
                          target_qs: torch.Tensor,
                          gamma: float,
                          td_lambda: float) -> torch.Tensor:
    """Compute TD(lambda) targets for Q-learning.
    
    Args:
        rewards: Reward tensor of shape [batch_size, episode_len, n_agents]
        terminated: Terminal state tensor of shape [batch_size, episode_len, 1]
        mask: Mask tensor of shape [batch_size, episode_len, 1]
        target_qs: Target Q-values of shape [batch_size, episode_len, n_agents]
        gamma: Discount factor
        td_lambda: TD(lambda) parameter
        
    Returns:
        TD(lambda) targets of shape [batch_size, episode_len, n_agents]
    """
    episode_len = rewards.shape[1]
    n_agents = rewards.shape[2]
    
    targets = torch.zeros_like(target_qs)
    running_lambda = torch.ones_like(target_qs[:, 0]) * td_lambda
    
    # Forward accumulate rewards
    for t in reversed(range(episode_len)):
        targets[:, t] = rewards[:, t] + gamma * (
            1 - terminated[:, t]) * (
                td_lambda * targets[:, t + 1] if t < episode_len - 1 else 0
            ) + (1 - td_lambda) * target_qs[:, t]
        
        running_lambda = running_lambda * gamma * td_lambda
        
    return targets

def compute_advantages(rewards: torch.Tensor,
                      values: torch.Tensor,
                      gamma: float,
                      gae_lambda: float) -> torch.Tensor:
    """Compute generalized advantage estimation (GAE).
    
    Args:
        rewards: Reward tensor of shape [batch_size, episode_len]
        values: Value estimates of shape [batch_size, episode_len]
        gamma: Discount factor
        gae_lambda: GAE parameter
        
    Returns:
        Advantage estimates of shape [batch_size, episode_len]
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(rewards.shape[1])):
        if t == rewards.shape[1] - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * gae_lambda * last_gae
    
    return advantages 