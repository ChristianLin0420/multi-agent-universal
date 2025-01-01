from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_tag_v3, simple_world_comm_v3

from ..base import MARLEnvironment

SUPPORTED_SCENARIOS = {
    "simple_spread": simple_spread_v3,
    "simple_adversary": simple_adversary_v3,
    "simple_tag": simple_tag_v3,
    "simple_world_comm": simple_world_comm_v3
}

class MPEEnvironment(MARLEnvironment):
    """Wrapper for Multi-Agent Particle Environments (MPE)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MPE environment wrapper.
        
        Args:
            config (Dict[str, Any]): Environment configuration containing:
                - scenario (str): Name of the MPE scenario
                - num_agents (int): Number of agents
                - max_cycles (int): Maximum number of cycles per episode
                - continuous_actions (bool): Use continuous action space
                - render_mode (str): Rendering mode ('human', 'rgb_array', None)
                - seed (int): Random seed
                - local_ratio (float): Ratio for local reward vs global reward
                - max_distance (float): Maximum distance for reward calculation
                - observation_radius (float): Radius for agent observations
                - collision_penalty (float): Penalty for agent collisions
                - shared_viewer (bool): Use shared viewer for rendering
        """
        super().__init__(config)
        
        # Extract configuration
        self.scenario = config.get("scenario", "simple_spread")
        self.num_agents = config.get("num_agents", 3)
        self.max_cycles = config.get("max_cycles", 100)
        self.continuous_actions = config.get("continuous_actions", False)
        self.render_mode = config.get("render_mode", "rgb_array")
        self.seed = config.get("seed", None)
        
        # Scenario-specific parameters
        self.local_ratio = config.get("local_ratio", 0.5)
        self.max_distance = config.get("max_distance", 1.0)
        self.observation_radius = config.get("observation_radius", 1.0)
        self.collision_penalty = config.get("collision_penalty", 0.0)
        self.shared_viewer = config.get("shared_viewer", True)
        
        # Validate scenario
        if self.scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported scenario: {self.scenario}. "
                           f"Supported scenarios: {list(SUPPORTED_SCENARIOS.keys())}")
        
        # Create environment
        env_module = SUPPORTED_SCENARIOS[self.scenario]
        self.env = env_module.parallel_env(
            N=self.num_agents if self.scenario != "simple_adversary" else self.num_agents - 1,
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
            render_mode=self.render_mode,
            local_ratio=self.local_ratio,
            max_distance=self.max_distance,
            observation_radius=self.observation_radius,
            collision_penalty=self.collision_penalty,
            shared_viewer=self.shared_viewer
        )
        
        if self.seed is not None:
            self.env.seed(self.seed)
        
        # Initialize environment
        self.env.reset()
        
        # Set up agent IDs
        self.agent_ids = list(self.env.agents)
        
        # Cache spaces
        self._setup_spaces()
        
        # Additional info
        self.episode_steps = 0
        self.episode_rewards = []
        self.collision_count = 0
        self.min_agent_distance = float('inf')
        self.coverage_score = 0.0
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        self._observation_space = {}
        self._action_space = {}
        
        for agent_id in self.agent_ids:
            obs_space = self.env.observation_space(agent_id)
            act_space = self.env.action_space(agent_id)
            
            self._observation_space[agent_id] = {
                "shape": obs_space.shape[0],
                "dtype": np.float32
            }
            
            if hasattr(act_space, "n"):  # Discrete action space
                self._action_space[agent_id] = {
                    "n": act_space.n,
                    "dtype": np.int64
                }
            else:  # Continuous action space
                self._action_space[agent_id] = {
                    "shape": act_space.shape[0],
                    "low": act_space.low,
                    "high": act_space.high,
                    "dtype": np.float32
                }
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute environment-specific metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            "collision_count": self.collision_count,
            "min_agent_distance": self.min_agent_distance,
            "coverage_score": self.coverage_score,
            "episode_length": self.episode_steps,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        }
        
        if self.scenario == "simple_spread":
            # Add scenario-specific metrics
            metrics.update(self._compute_spread_metrics())
        elif self.scenario == "simple_adversary":
            metrics.update(self._compute_adversary_metrics())
        
        return metrics
    
    def _compute_spread_metrics(self) -> Dict[str, float]:
        """Compute metrics specific to simple_spread scenario."""
        # Get agent positions
        positions = [self.env.get_agent_state(i)[:2] for i in range(self.num_agents)]
        positions = np.array(positions)
        
        # Compute minimum distance between agents
        distances = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        min_distance = min(distances) if distances else float('inf')
        self.min_agent_distance = min(self.min_agent_distance, min_distance)
        
        # Compute coverage
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        coverage_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        self.coverage_score = coverage_area
        
        return {
            "current_min_distance": min_distance,
            "current_coverage": coverage_area
        }
    
    def _compute_adversary_metrics(self) -> Dict[str, float]:
        """Compute metrics specific to simple_adversary scenario."""
        return {
            "adversary_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "good_agent_reward": np.mean([r for r in self.episode_rewards[:-1]]) if self.episode_rewards else 0.0
        }
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observations and info
        """
        observations = self.env.reset()
        
        self.episode_steps = 0
        self.episode_rewards = []
        self.collision_count = 0
        self.min_agent_distance = float('inf')
        self.coverage_score = 0.0
        
        info = {
            "episode_steps": 0,
            "metrics": self._compute_metrics(),
            "scenario": self.scenario,
            "num_agents": self.num_agents
        }
        
        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # dones
        Dict                    # info
    ]:
        """Execute actions in the environment.
        
        Args:
            actions (Dict[str, np.ndarray]): Actions for each agent
            
        Returns:
            Tuple containing observations, rewards, dones, and info
        """
        observations, rewards, dones, infos = self.env.step(actions)
        
        # Update episode info
        self.episode_steps += 1
        self.episode_rewards.append(sum(rewards.values()))
        
        # Update collision count
        for info in infos.values():
            if info.get("collision", False):
                self.collision_count += 1
        
        # Compute current metrics
        metrics = self._compute_metrics()
        
        # Compile info
        info = {
            "episode_steps": self.episode_steps,
            "metrics": metrics,
            "collisions": self.collision_count,
            "min_agent_distance": self.min_agent_distance,
            "coverage_score": self.coverage_score
        }
        info.update(infos)
        
        return observations, rewards, dones, info
    
    def render(self) -> np.ndarray:
        """Render the environment.
        
        Returns:
            np.ndarray: RGB array of the rendered frame
        """
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        return self._observation_space
    
    @property
    def action_space(self) -> Dict[str, Any]:
        return self._action_space 