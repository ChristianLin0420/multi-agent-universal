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
                # No additional parameters needed for MPE scenarios
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
            local_ratio=self.local_ratio
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
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observations and info
        """
        observations, _ = self.env.reset()
        
        self.episode_steps = 0
        self.episode_rewards = []
        self.collision_count = 0
        
        info = {
            "episode_steps": 0,
            "scenario": self.scenario,
            "num_agents": self.num_agents
        }
        
        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # truncateds
        Dict[str, bool],        # dones
        Dict                    # info
    ]:
        """Execute actions in the environment.
        
        Args:
            actions (Dict[str, np.ndarray]): Actions for each agent
            
        Returns:
            Tuple containing observations, rewards, dones, and info
        """
        observations, rewards, dones, truncateds, infos = self.env.step(actions)
        
        # Update episode info
        self.episode_steps += 1
        self.episode_rewards.append(sum(rewards.values()))
        
        # Update collision count
        for info in infos.values():
            if info.get("collision", False):
                self.collision_count += 1
        
        # Compile info
        info = {
            "episode_steps": self.episode_steps,
            "collisions": self.collision_count,
        }
        info.update(infos)
        
        return observations, rewards, truncateds, dones, info
    
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

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute environment-specific metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # Get agent positions
        agent_positions = []
        for agent_id in self.agent_ids:
            obs = self.env.observe(agent_id)
            # Extract position from observation (first 2 values are x,y coordinates)
            pos = obs[:2]
            agent_positions.append(pos)
        
        agent_positions = np.array(agent_positions)
        
        # Compute metrics
        metrics = {
            "collision_rate": self._compute_collision_rate(),
            "min_agent_distance": self._compute_min_agent_distance(agent_positions),
            "coverage_score": self._compute_coverage_score(agent_positions),
            "episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "episode_length": self.episode_steps
        }
        
        return metrics

    def _compute_collision_rate(self) -> float:
        """Compute collision rate for the current episode.
        
        Returns:
            float: Collision rate (collisions per step)
        """
        if self.episode_steps == 0:
            return 0.0
        return self.collision_count / self.episode_steps

    def _compute_min_agent_distance(self, positions: np.ndarray) -> float:
        """Compute minimum distance between any pair of agents.
        
        Args:
            positions (np.ndarray): Array of agent positions
            
        Returns:
            float: Minimum distance between agents
        """
        n_agents = len(positions)
        min_dist = float('inf')
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 0.0

    def _compute_coverage_score(self, positions: np.ndarray) -> float:
        """Compute coverage score based on agent positions.
        
        Args:
            positions (np.ndarray): Array of agent positions
            
        Returns:
            float: Coverage score (higher is better)
        """
        # Define the bounds of the environment
        env_size = 2.0  # MPE environments typically use [-1, 1] range
        grid_size = 10  # Number of grid cells for coverage computation
        
        # Create grid
        x = np.linspace(-env_size/2, env_size/2, grid_size)
        y = np.linspace(-env_size/2, env_size/2, grid_size)
        grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        
        # Compute minimum distance from each grid point to any agent
        min_distances = np.min([
            np.linalg.norm(grid_points - pos, axis=1)
            for pos in positions
        ], axis=0)
        
        # Convert distances to coverage score (using exponential decay)
        coverage = np.mean(np.exp(-min_distances))
        
        return coverage 