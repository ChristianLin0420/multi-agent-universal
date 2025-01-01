import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import wandb
import cv2

from ..algorithms.base import MARLAlgorithm
from ..environments.base import MARLEnvironment
from ..utils.logger import Logger
from ..utils.visualization import plot_episode_metrics, create_video

class Evaluator:
    """Evaluation pipeline for MARL algorithms."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        algorithm: MARLAlgorithm,
        env: MARLEnvironment,
        experiment_name: str
    ):
        """Initialize the evaluator.
        
        Args:
            config (Dict[str, Any]): Evaluation configuration
            algorithm (MARLAlgorithm): MARL algorithm instance
            env (MARLEnvironment): Environment instance
            experiment_name (str): Name of the experiment
        """
        self.config = config
        self.algorithm = algorithm
        self.env = env
        
        # Evaluation parameters
        self.n_episodes = config.get("n_episodes", 100)
        self.render = config.get("render", False)
        self.save_videos = config.get("save_videos", False)
        self.video_fps = config.get("video_fps", 30)
        self.record_trajectories = config.get("record_trajectories", False)
        self.save_heatmaps = config.get("save_heatmaps", False)
        
        # Setup directories
        self.experiment_dir = Path(config.get("output_dir", "experiments")) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_dir = self.experiment_dir / "videos"
        self.plot_dir = self.experiment_dir / "plots"
        self.trajectory_dir = self.experiment_dir / "trajectories"
        
        if self.save_videos:
            self.video_dir.mkdir(exist_ok=True)
        if self.save_heatmaps:
            self.plot_dir.mkdir(exist_ok=True)
        if self.record_trajectories:
            self.trajectory_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = Logger(
            config=config,
            experiment_name=experiment_name,
            log_dir=str(self.experiment_dir)
        )
        
        # Initialize metrics storage
        self.episode_metrics = []
        self.agent_metrics = {agent_id: [] for agent_id in env.agent_ids}
        self.trajectories = []
        
        # Environment-specific metrics
        self.env_type = config["environment"]["type"]
        self.custom_metrics = self._setup_custom_metrics()
    
    def _setup_custom_metrics(self) -> Dict[str, List[float]]:
        """Setup environment-specific metrics."""
        metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate": []
        }
        
        if self.env_type == "smac":
            metrics.update({
                "battles_won": [],
                "kill_death_ratio": [],
                "damage_dealt": [],
                "damage_taken": []
            })
        elif self.env_type == "football":
            metrics.update({
                "goals_scored": [],
                "possession_rate": [],
                "pass_accuracy": [],
                "shot_accuracy": []
            })
        elif self.env_type == "mpe":
            metrics.update({
                "collision_rate": [],
                "min_agent_distance": [],
                "coverage_score": []
            })
        
        return metrics
    
    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the algorithm.
        
        Args:
            checkpoint_path (Optional[str]): Path to model checkpoint
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if checkpoint_path:
            self.algorithm.load(checkpoint_path)
        
        self.algorithm.eval()  # Set to evaluation mode
        
        episode_rewards = []
        episode_lengths = []
        frames_buffer = []
        
        for episode in tqdm(range(self.n_episodes), desc="Evaluating"):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_frames = []
            episode_trajectory = []
            
            while True:
                if self.render or self.save_videos:
                    frame = self.env.render()
                    episode_frames.append(frame)
                
                # Select actions
                with torch.no_grad():
                    actions, action_info = self.algorithm.select_actions(
                        observations=obs,
                        agent_ids=self.env.agent_ids,
                        explore=False
                    )
                
                # Record trajectory
                if self.record_trajectories:
                    episode_trajectory.append({
                        "observations": obs,
                        "actions": actions,
                        "action_info": action_info
                    })
                
                # Execute actions
                next_obs, rewards, dones, step_info = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Update metrics
                self._update_metrics(step_info)
                
                if any(dones.values()):
                    break
                
                obs = next_obs
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Save video
            if self.save_videos and episode_frames:
                video_path = self.video_dir / f"episode_{episode}.mp4"
                create_video(episode_frames, str(video_path), fps=self.video_fps)
            
            # Save trajectory
            if self.record_trajectories:
                trajectory_path = self.trajectory_dir / f"episode_{episode}.json"
                with open(trajectory_path, 'w') as f:
                    json.dump(episode_trajectory, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            
            # Log episode metrics
            self.logger.log_metrics({
                "eval_episode": episode,
                "eval_episode_reward": episode_reward,
                "eval_episode_length": episode_length,
                **{f"eval_{k}": v[-1] for k, v in self.custom_metrics.items()}
            })
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(episode_rewards, episode_lengths)
        
        # Generate visualizations
        if self.save_heatmaps:
            self._generate_visualizations(metrics)
        
        # Save final metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _update_metrics(self, info: Dict[str, Any]):
        """Update environment-specific metrics.
        
        Args:
            info (Dict[str, Any]): Step information
        """
        if self.env_type == "smac":
            if "battle_won" in info:
                self.custom_metrics["battles_won"].append(int(info["battle_won"]))
            if "damage_dealt" in info and "damage_taken" in info:
                self.custom_metrics["damage_dealt"].append(info["damage_dealt"])
                self.custom_metrics["damage_taken"].append(info["damage_taken"])
                self.custom_metrics["kill_death_ratio"].append(
                    info.get("kill_death_ratio", 0.0)
                )
        
        elif self.env_type == "football":
            if "score" in info:
                self.custom_metrics["goals_scored"].append(info["score"][0])
            if "possession" in info:
                self.custom_metrics["possession_rate"].append(
                    info["possession"].get("left", 0)
                )
            if "statistics" in info:
                stats = info["statistics"]
                self.custom_metrics["pass_accuracy"].append(
                    stats.get("pass_accuracy", 0.0)
                )
                self.custom_metrics["shot_accuracy"].append(
                    stats.get("shot_accuracy", 0.0)
                )
        
        elif self.env_type == "mpe":
            metrics = info.get("metrics", {})
            self.custom_metrics["collision_rate"].append(
                metrics.get("collision_rate", 0.0)
            )
            self.custom_metrics["min_agent_distance"].append(
                metrics.get("min_agent_distance", 0.0)
            )
            self.custom_metrics["coverage_score"].append(
                metrics.get("coverage_score", 0.0)
            )
    
    def _compute_aggregate_metrics(self, 
                                 episode_rewards: List[float],
                                 episode_lengths: List[float]) -> Dict[str, float]:
        """Compute aggregate metrics across episodes.
        
        Args:
            episode_rewards (List[float]): List of episode rewards
            episode_lengths (List[float]): List of episode lengths
            
        Returns:
            Dict[str, float]: Aggregate metrics
        """
        metrics = {
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths)
        }
        
        # Add environment-specific metrics
        if self.env_type == "smac":
            metrics.update({
                "win_rate": np.mean(self.custom_metrics["battles_won"]),
                "mean_damage_dealt": np.mean(self.custom_metrics["damage_dealt"]),
                "mean_damage_taken": np.mean(self.custom_metrics["damage_taken"]),
                "mean_kd_ratio": np.mean(self.custom_metrics["kill_death_ratio"])
            })
        
        elif self.env_type == "football":
            metrics.update({
                "goals_per_episode": np.mean(self.custom_metrics["goals_scored"]),
                "mean_possession": np.mean(self.custom_metrics["possession_rate"]),
                "mean_pass_accuracy": np.mean(self.custom_metrics["pass_accuracy"]),
                "mean_shot_accuracy": np.mean(self.custom_metrics["shot_accuracy"])
            })
        
        elif self.env_type == "mpe":
            metrics.update({
                "mean_collision_rate": np.mean(self.custom_metrics["collision_rate"]),
                "mean_min_distance": np.mean(self.custom_metrics["min_agent_distance"]),
                "mean_coverage": np.mean(self.custom_metrics["coverage_score"])
            })
        
        return metrics
    
    def _generate_visualizations(self, metrics: Dict[str, float]):
        """Generate and save visualization plots.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
        """
        # Plot reward distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.custom_metrics["episode_rewards"])
        plt.title("Episode Reward Distribution")
        plt.savefig(self.plot_dir / "reward_distribution.png")
        plt.close()
        
        # Plot environment-specific metrics
        if self.env_type == "smac":
            self._plot_smac_metrics()
        elif self.env_type == "football":
            self._plot_football_metrics()
        elif self.env_type == "mpe":
            self._plot_mpe_metrics()
        
        # Save metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.experiment_dir / "metrics_summary.csv", index=False)
    
    def _plot_smac_metrics(self):
        """Generate SMAC-specific visualization plots."""
        # Plot damage dealt vs. taken
        plt.figure(figsize=(10, 6))
        plt.plot(self.custom_metrics["damage_dealt"], label="Damage Dealt")
        plt.plot(self.custom_metrics["damage_taken"], label="Damage Taken")
        plt.title("Damage Analysis")
        plt.legend()
        plt.savefig(self.plot_dir / "damage_analysis.png")
        plt.close()
        
        # Plot win rate over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.custom_metrics["battles_won"]) / 
                np.arange(1, len(self.custom_metrics["battles_won"]) + 1))
        plt.title("Cumulative Win Rate")
        plt.savefig(self.plot_dir / "win_rate.png")
        plt.close()
    
    def _plot_football_metrics(self):
        """Generate football-specific visualization plots."""
        # Plot possession and accuracy metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(self.custom_metrics["possession_rate"])
        ax1.set_title("Possession Rate")
        
        ax2.plot(self.custom_metrics["pass_accuracy"], label="Pass Accuracy")
        ax2.plot(self.custom_metrics["shot_accuracy"], label="Shot Accuracy")
        ax2.set_title("Accuracy Metrics")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / "football_metrics.png")
        plt.close()
    
    def _plot_mpe_metrics(self):
        """Generate MPE-specific visualization plots."""
        # Plot coverage and collision metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(self.custom_metrics["coverage_score"])
        ax1.set_title("Coverage Score")
        
        ax2.plot(self.custom_metrics["collision_rate"])
        ax2.set_title("Collision Rate")
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / "mpe_metrics.png")
        plt.close()
    
    def _save_metrics(self, metrics: Dict[str, float]):
        """Save evaluation metrics to file.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
        """
        # Save metrics as JSON
        metrics_path = self.experiment_dir / "eval_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log to wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.log({"eval/" + k: v for k, v in metrics.items()})
        
        # Print results
        self.logger.info("\nEvaluation Results:")
        self.logger.info("=" * 50)
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}") 