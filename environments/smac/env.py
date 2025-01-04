from typing import Dict, Any, Tuple
import numpy as np
from smac.env import StarCraft2Env
from smac.env.starcraft2.maps import get_map_params

from ..base import MARLEnvironment

DIFFICULTY_LEVELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A"]

class SMACEnvironment(MARLEnvironment):
    """Wrapper for StarCraft Multi-Agent Challenge (SMAC) environments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SMAC environment wrapper.
        
        Args:
            config (Dict[str, Any]): Environment configuration containing:
                - map_name (str): Name of the SMAC scenario
                - difficulty (str): Game difficulty (1-9, A)
                - reward_sparse (bool): Use sparse rewards
                - state_last_action (bool): Include last actions in state
                - obs_last_action (bool): Include last actions in observations
                - obs_own_pos (bool): Include agent's own position
                - obs_terrain_height (bool): Include terrain height
                - obs_instead_of_state (bool): Use observations instead of state
                - obs_timestep_number (bool): Include timestep in observations
                - reward_death_value (float): Reward value for unit death
                - reward_win (float): Reward value for winning
                - reward_defeat (float): Reward value for losing
                - reward_negative_scale (float): Scaling for negative rewards
                - reward_scale (bool): Whether to scale rewards
                - reward_scale_rate (float): Rate for reward scaling
                - replay_dir (str): Directory to save replays
                - replay_prefix (str): Prefix for replay files
                - window_size (Tuple[int, int]): Window size for rendering
                - seed (int): Random seed
                - debug (bool): Enable debug mode
        """
        super().__init__(config)
        
        # Extract configuration
        self.map_name = config.get("map_name", "3m")
        self.difficulty = config.get("difficulty", "7")
        
        # Validate difficulty
        if self.difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty level. Must be one of {DIFFICULTY_LEVELS}")
        
        # Get map params
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["episode_limit"]
        
        # Observation and state settings
        self.state_last_action = config.get("state_last_action", True)
        self.obs_last_action = config.get("obs_last_action", True)
        self.obs_own_pos = config.get("obs_own_pos", True)
        self.obs_terrain_height = config.get("obs_terrain_height", False)
        self.obs_instead_of_state = config.get("obs_instead_of_state", False)
        self.obs_timestep_number = config.get("obs_timestep_number", True)
        
        # Reward settings
        self.reward_sparse = config.get("reward_sparse", False)
        self.reward_death_value = config.get("reward_death_value", 10)
        self.reward_win = config.get("reward_win", 200)
        self.reward_defeat = config.get("reward_defeat", 0)
        self.reward_negative_scale = config.get("reward_negative_scale", 0.5)
        self.reward_scale = config.get("reward_scale", True)
        self.reward_scale_rate = config.get("reward_scale_rate", 20)
        
        # Replay settings
        self.replay_dir = config.get("replay_dir", None)
        self.replay_prefix = config.get("replay_prefix", None)
        
        # Display settings
        window_size = config.get("window_size", (1920, 1200))
        self.window_size_x, self.window_size_y = window_size
        
        # Debug settings
        self.debug = config.get("debug", False)
        self.seed = config.get("seed", None)
        
        # Create environment
        self.env = StarCraft2Env(
            map_name=self.map_name,
            step_mul=8,
            difficulty=self.difficulty,
            game_version=None,
            seed=self.seed,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=self.obs_last_action,
            obs_pathing_grid=False,
            obs_terrain_height=self.obs_terrain_height,
            obs_instead_of_state=self.obs_instead_of_state,
            obs_timestep_number=self.obs_timestep_number,
            state_last_action=self.state_last_action,
            state_timestep_number=True,
            reward_sparse=self.reward_sparse,
            reward_only_positive=True,
            reward_death_value=self.reward_death_value,
            reward_win=self.reward_win,
            reward_defeat=self.reward_defeat,
            reward_negative_scale=self.reward_negative_scale,
            reward_scale=self.reward_scale,
            reward_scale_rate=self.reward_scale_rate,
            replay_dir=self.replay_dir,
            replay_prefix=self.replay_prefix,
            window_size_x=self.window_size_x,
            window_size_y=self.window_size_y,
            debug=self.debug
        )
        
        # Set up agent IDs
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        
        # Cache spaces
        self._setup_spaces()
        
        # Additional info
        self.episode_steps = 0
        self.episode_rewards = []
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.unit_kill_count = {i: 0 for i in range(self.n_agents)}
        
        # Performance metrics
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.agents_alive = self.n_agents
        self.enemies_alive = self.n_enemies
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        self._observation_space = {
            agent_id: {
                "shape": self.env.get_obs_size(),
                "dtype": np.float32
            }
            for agent_id in self.agent_ids
        }
        
        self._action_space = {
            agent_id: {
                "n": self.env.n_actions,
                "dtype": np.int64
            }
            for agent_id in self.agent_ids
        }
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute environment-specific metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / max(1, self.battles_game),
            "timeouts": self.timeouts,
            "force_restarts": self.force_restarts,
            "episode_length": self.episode_steps,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "agents_alive": self.agents_alive,
            "enemies_alive": self.enemies_alive,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "damage_ratio": self.total_damage_dealt / max(1, self.total_damage_taken),
            "kill_death_ratio": sum(self.unit_kill_count.values()) / max(1, self.n_agents - self.agents_alive)
        }
        
        # Add per-agent metrics
        for i in range(self.n_agents):
            metrics[f"agent_{i}_kills"] = self.unit_kill_count[i]
        
        return metrics
    
    def _update_battle_stats(self, battle_won: bool, timeout: bool):
        """Update battle statistics.
        
        Args:
            battle_won (bool): Whether the battle was won
            timeout (bool): Whether the episode timed out
        """
        self.battles_game += 1
        if battle_won:
            self.battles_won += 1
        if timeout:
            self.timeouts += 1
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observations and info
        """
        raw_obs = self.env.reset()
        observations = {
            agent_id: obs for agent_id, obs in zip(self.agent_ids, raw_obs)
        }
        
        # Reset episode info
        self.episode_steps = 0
        self.episode_rewards = []
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.agents_alive = self.n_agents
        self.enemies_alive = self.n_enemies
        self.unit_kill_count = {i: 0 for i in range(self.n_agents)}
        
        info = {
            "state": self.env.get_state(),
            "avail_actions": self.env.get_avail_actions(),
            "battle_won": False,
            "metrics": self._compute_metrics(),
            "map_name": self.map_name,
            "difficulty": self.difficulty,
            "n_agents": self.n_agents,
            "n_enemies": self.n_enemies,
            "episode_limit": self.episode_limit,
            "ally_state": self.env.get_ally_units_info(),
            "enemy_state": self.env.get_enemy_units_info()
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
        # Convert actions dict to list
        action_list = [
            actions[agent_id] for agent_id in self.agent_ids
        ]
        
        # Step environment
        reward, terminated, env_info = self.env.step(action_list)
        raw_obs = self.env.get_obs()
        
        # Process observations
        observations = {
            agent_id: obs for agent_id, obs in zip(self.agent_ids, raw_obs)
        }
        
        # Update episode info
        self.episode_steps += 1
        self.episode_rewards.append(reward)
        
        # Update battle statistics
        if terminated:
            battle_won = env_info.get("battle_won", False)
            timeout = self.episode_steps >= self.episode_limit
            self._update_battle_stats(battle_won, timeout)
        
        # Update unit statistics
        self.agents_alive = sum([1 for u in self.env.get_ally_units_info() if u.health > 0])
        self.enemies_alive = sum([1 for u in self.env.get_enemy_units_info() if u.health > 0])
        
        # Update damage statistics
        current_stats = self.env.get_stats()
        if self.last_stats is not None:
            self.total_damage_dealt += current_stats["damage_dealt"] - self.last_stats["damage_dealt"]
            self.total_damage_taken += current_stats["damage_taken"] - self.last_stats["damage_taken"]
            
            # Update kill counts
            for i in range(self.n_agents):
                new_kills = current_stats[f"agent_{i}_kills"] - self.last_stats[f"agent_{i}_kills"]
                self.unit_kill_count[i] += new_kills
        
        self.last_stats = current_stats
        
        # Process rewards and dones
        rewards = {
            agent_id: reward for agent_id in self.agent_ids
        }
        
        dones = {
            agent_id: terminated for agent_id in self.agent_ids
        }
        
        # Compile info
        info = {
            "state": self.env.get_state(),
            "avail_actions": self.env.get_avail_actions(),
            "battle_won": env_info.get("battle_won", False),
            "metrics": self._compute_metrics(),
            "episode_steps": self.episode_steps,
            "episode_reward": sum(self.episode_rewards),
            "agents_alive": self.agents_alive,
            "enemies_alive": self.enemies_alive,
            "ally_state": self.env.get_ally_units_info(),
            "enemy_state": self.env.get_enemy_units_info(),
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken
        }
        info.update(env_info)
        
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
    
    def save_replay(self):
        """Save a replay."""
        self.env.save_replay()
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        return self._observation_space
    
    @property
    def action_space(self) -> Dict[str, Any]:
        return self._action_space 