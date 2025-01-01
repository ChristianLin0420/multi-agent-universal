from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import numpy.ma as ma

from ..base import MARLEnvironment

SUPPORTED_SCENARIOS = [
    "academy_3_vs_1_with_keeper",
    "academy_empty_goal_close",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_run_to_score_with_keeper",
    "academy_single_goal_versus_lazy",
    "academy_corner",
    "11_vs_11_kaggle",
    "11_vs_11_stochastic"
]

class GoogleFootballEnvironment(MARLEnvironment):
    """Wrapper for Google Research Football environment."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize football environment wrapper.
        
        Args:
            config (Dict[str, Any]): Environment configuration containing:
                - scenario_name (str): Name of the scenario
                - number_of_left_players (int): Number of left team players
                - number_of_right_players (int): Number of right team players
                - representation (str): Observation representation type
                - rewards (str): Reward shaping type
                - render (bool): Whether to render the environment
                - write_video (bool): Whether to write videos
                - write_full_episode_dumps (bool): Write full episode dumps
                - dump_frequency (int): Episode dump frequency
                - extra_players (str): Extra players configuration
                - other_config_options (dict): Additional configuration options
                - stacked (bool): Whether to stack observations
                - channel_dimensions (Tuple[int, int]): Channel dimensions
                - seed (int): Random seed
        """
        super().__init__(config)
        
        # Extract configuration
        self.scenario_name = config.get("scenario_name", "academy_3_vs_1_with_keeper")
        self.number_of_left_players = config.get("number_of_left_players", 3)
        self.number_of_right_players = config.get("number_of_right_players", 1)
        
        # Validate scenario
        if self.scenario_name not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported scenario: {self.scenario_name}. "
                           f"Supported scenarios: {SUPPORTED_SCENARIOS}")
        
        # Environment settings
        self.representation = config.get("representation", "raw")
        self.rewards = config.get("rewards", "scoring,checkpoints")
        self.render_mode = config.get("render", False)
        self.write_video = config.get("write_video", False)
        self.write_full_episode_dumps = config.get("write_full_episode_dumps", False)
        self.dump_frequency = config.get("dump_frequency", 1)
        self.extra_players = config.get("extra_players", None)
        self.other_config_options = config.get("other_config_options", {})
        
        # Observation settings
        self.stacked = config.get("stacked", False)
        self.channel_dimensions = config.get("channel_dimensions", (43, 43))
        self.seed = config.get("seed", None)
        
        # Create environment
        self.env = football_env.create_environment(
            env_name=self.scenario_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            write_goal_dumps=self.write_full_episode_dumps,
            write_full_episode_dumps=self.write_full_episode_dumps,
            render=self.render_mode,
            write_video=self.write_video,
            dump_frequency=self.dump_frequency,
            number_of_left_players_agent_controls=self.number_of_left_players,
            number_of_right_players_agent_controls=self.number_of_right_players,
            channel_dimensions=self.channel_dimensions,
            other_config_options=self.other_config_options
        )
        
        if self.seed is not None:
            self.env.seed(self.seed)
        
        # Set up agent IDs
        self.agent_ids = [f"player_{i}" for i in range(self.number_of_left_players)]
        
        # Cache spaces
        self._setup_spaces()
        
        # Additional info
        self.episode_steps = 0
        self.episode_rewards = []
        self.goals_scored = 0
        self.goals_conceded = 0
        self.possession_stats = {"left": 0, "right": 0}
        self.shots_taken = 0
        self.shots_on_target = 0
        self.passes_completed = 0
        self.passes_attempted = 0
        self.tackles_successful = 0
        self.tackles_attempted = 0
        self.yellow_cards = 0
        self.red_cards = 0
        self.offsides = 0
        self.corners = 0
        self.fouls = 0
        
        # Performance tracking
        self.games_won = 0
        self.games_played = 0
        self.draws = 0
        self.timeouts = 0
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        if self.representation == "raw":
            obs_shape = self.env.observation_space.shape[0]
        else:
            obs_shape = np.prod(self.channel_dimensions)
        
        self._observation_space = {
            agent_id: {
                "shape": obs_shape,
                "dtype": np.float32
            }
            for agent_id in self.agent_ids
        }
        
        self._action_space = {
            agent_id: {
                "n": self.env.action_space.n,
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
            "games_won": self.games_won,
            "games_played": self.games_played,
            "win_rate": self.games_won / max(1, self.games_played),
            "draws": self.draws,
            "timeouts": self.timeouts,
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "goal_difference": self.goals_scored - self.goals_conceded,
            "shots_taken": self.shots_taken,
            "shots_on_target": self.shots_taken,
            "shot_accuracy": self.shots_on_target / max(1, self.shots_taken),
            "passes_completed": self.passes_completed,
            "passes_attempted": self.passes_attempted,
            "pass_accuracy": self.passes_completed / max(1, self.passes_attempted),
            "tackles_successful": self.tackles_successful,
            "tackles_attempted": self.tackles_attempted,
            "tackle_success_rate": self.tackles_successful / max(1, self.tackles_attempted),
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "offsides": self.offsides,
            "corners": self.corners,
            "fouls": self.fouls,
            "possession_percentage": self.possession_stats["left"] / max(1, sum(self.possession_stats.values())) * 100,
            "episode_length": self.episode_steps,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        }
        
        return metrics
    
    def _update_match_stats(self, info: Dict[str, Any]):
        """Update match statistics.
        
        Args:
            info (Dict[str, Any]): Information from environment step
        """
        if "score" in info:
            self.goals_scored = info["score"][0]
            self.goals_conceded = info["score"][1]
        
        if "possession" in info:
            self.possession_stats = info["possession"]
        
        if "statistics" in info:
            stats = info["statistics"]
            self.shots_taken = stats.get("score_attempts", self.shots_taken)
            self.shots_on_target = stats.get("shots_on_target", self.shots_on_target)
            self.passes_completed = stats.get("passes_completed", self.passes_completed)
            self.passes_attempted = stats.get("passes_attempted", self.passes_attempted)
            self.tackles_successful = stats.get("tackles_successful", self.tackles_successful)
            self.tackles_attempted = stats.get("tackles_attempted", self.tackles_attempted)
            self.yellow_cards = stats.get("yellow_cards", self.yellow_cards)
            self.red_cards = stats.get("red_cards", self.red_cards)
            self.offsides = stats.get("offsides", self.offsides)
            self.corners = stats.get("corners", self.corners)
            self.fouls = stats.get("fouls", self.fouls)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observations and info
        """
        obs = self.env.reset()
        
        if not isinstance(obs, list):
            obs = [obs]
        
        observations = {
            agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        # Reset episode info
        self.episode_steps = 0
        self.episode_rewards = []
        self.goals_scored = 0
        self.goals_conceded = 0
        self.possession_stats = {"left": 0, "right": 0}
        self.shots_taken = 0
        self.shots_on_target = 0
        self.passes_completed = 0
        self.passes_attempted = 0
        self.tackles_successful = 0
        self.tackles_attempted = 0
        self.yellow_cards = 0
        self.red_cards = 0
        self.offsides = 0
        self.corners = 0
        self.fouls = 0
        
        info = {
            "score": [0, 0],
            "metrics": self._compute_metrics(),
            "scenario": self.scenario_name,
            "n_left_players": self.number_of_left_players,
            "n_right_players": self.number_of_right_players,
            "steps": 0,
            "game_mode": "normal",
            "ball_owned_team": -1,
            "ball_owned_player": -1
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
        obs, reward, done, info = self.env.step(action_list)
        
        if not isinstance(obs, list):
            obs = [obs]
            reward = [reward]
            done = [done]
        
        # Process observations
        observations = {
            agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        # Update episode info
        self.episode_steps += 1
        self.episode_rewards.append(sum(reward))
        
        # Update match statistics
        self._update_match_stats(info)
        
        # Update game statistics
        if done[0]:  # Assuming all agents finish simultaneously
            self.games_played += 1
            if self.goals_scored > self.goals_conceded:
                self.games_won += 1
            elif self.goals_scored == self.goals_conceded:
                self.draws += 1
            if self.episode_steps >= 3000:  # Default timeout
                self.timeouts += 1
        
        # Process rewards and dones
        rewards = {
            agent_id: reward[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        dones = {
            agent_id: done[i] for i, agent_id in enumerate(self.agent_ids)
        }
        
        # Compile info
        info.update({
            "metrics": self._compute_metrics(),
            "episode_steps": self.episode_steps,
            "episode_reward": sum(self.episode_rewards),
            "score": [self.goals_scored, self.goals_conceded],
            "possession": self.possession_stats,
            "statistics": {
                "shots_taken": self.shots_taken,
                "shots_on_target": self.shots_on_target,
                "passes_completed": self.passes_completed,
                "passes_attempted": self.passes_attempted,
                "tackles_successful": self.tackles_successful,
                "tackles_attempted": self.tackles_attempted,
                "yellow_cards": self.yellow_cards,
                "red_cards": self.red_cards,
                "offsides": self.offsides,
                "corners": self.corners,
                "fouls": self.fouls
            }
        })
        
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
    
    def write_dump(self, name: str):
        """Write an episode dump.
        
        Args:
            name (str): Name of the dump file
        """
        self.env.write_dump(name)
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        return self._observation_space
    
    @property
    def action_space(self) -> Dict[str, Any]:
        return self._action_space 