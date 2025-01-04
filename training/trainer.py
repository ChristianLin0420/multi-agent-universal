from typing import Dict, Any, Optional
import torch
from pathlib import Path
import wandb
from tqdm import tqdm
import json
import shutil

from algorithms.base import MARLAlgorithm
from environments.base import MARLEnvironment
from utils.logger import Logger
from evaluation.evaluator import Evaluator

class Trainer:
    """Base trainer for MARL algorithms."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        algorithm: MARLAlgorithm,
        env: MARLEnvironment,
        experiment_name: str
    ):
        """Initialize trainer.
        
        Args:
            config (Dict[str, Any]): Training configuration
            algorithm (MARLAlgorithm): Algorithm instance
            env (MARLEnvironment): Environment instance
            experiment_name (str): Name of the experiment
        """
        self.config = config
        self.algorithm = algorithm
        self.env = env
        self.experiment_name = experiment_name
        
        # Training parameters
        self.max_episodes = config.get("max_episodes", 1000)
        self.max_steps = config.get("max_steps", 100)
        self.eval_interval = config.get("eval_interval", 100)
        self.save_interval = config.get("save_interval", 100)
        self.log_interval = config.get("log_interval", 10)
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.logger = Logger(
            config=config,
            experiment_name=experiment_name,
            log_dir=str(self.log_dir)
        )
        
        # Initialize training metrics
        self.episode = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate": [],
            "losses": []
        }
        
        # Setup evaluator
        self.evaluator = Evaluator(
            config=config,
            algorithm=algorithm,
            env=env,
            experiment_name=experiment_name
        )
        
        # Initialize wandb if enabled
        self.setup_wandb()
        
        # Save configuration
        self.save_config()
    
    def setup_directories(self):
        """Setup experiment directories."""
        base_dir = Path(self.config.get("output_dir", "experiments"))
        self.experiment_dir = base_dir / self.experiment_name
        
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        self.eval_dir = self.experiment_dir / "eval"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "marl_framework"),
                name=self.experiment_name,
                config=self.config,
                dir=str(self.experiment_dir)
            )
    
    def save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save training checkpoint.
        
        Args:
            episode (int): Current episode number
            is_best (bool): Whether this is the best checkpoint so far
        """
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'training_metrics': self.training_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Keep only last few checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 5):
        """Clean up old checkpoints.
        
        Args:
            keep_last (int): Number of recent checkpoints to keep
        """
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_*.pt")],
            key=lambda x: int(x.stem.split('_')[1])
        )
        
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.best_reward = checkpoint['best_reward']
        self.algorithm.load_state_dict(checkpoint['algorithm_state'])
        self.training_metrics = checkpoint['training_metrics']
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode.
        
        Returns:
            Dict[str, float]: Episode metrics
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_metrics = {}
        
        while True:
            # Select actions
            actions, action_info = self.algorithm.select_actions(
                observations=obs,
                agent_ids=self.env.agent_ids,
                explore=True
            )
            
            # Execute actions
            next_obs, rewards, truncateds, dones, step_info = self.env.step(actions)
            
            # Store experience
            self.algorithm.store_transition(
                obs, actions, rewards, next_obs, dones, info
            )
            
            # Update metrics
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            # Update algorithm if needed
            if self.algorithm.ready_to_update():
                update_info = self.algorithm.update()
                episode_metrics.update(update_info)
            
            if any(dones.values()) or any(truncateds.values()) or episode_length >= self.max_steps:
                break
            
            obs = next_obs
            info = step_info
        
        # Compute episode metrics
        episode_metrics.update({
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": step_info.get("success", False)
        })
        
        return episode_metrics
    
    def log_metrics(self, metrics: Dict[str, float], episode: int):
        """Log training metrics.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
            episode (int): Current episode number
        """
        # Update stored metrics
        self.training_metrics["episode_rewards"].append(metrics["episode_reward"])
        self.training_metrics["episode_lengths"].append(metrics["episode_length"])
        self.training_metrics["success_rate"].append(float(metrics["success"]))
        
        if "loss" in metrics:
            self.training_metrics["losses"].append(metrics["loss"])
        
        # Log to wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.log({"train/" + k: v for k, v in metrics.items()}, step=episode)
        
        # Log to console periodically
        if episode % self.log_interval == 0:
            self.logger.info(
                f"Episode {episode}/{self.max_episodes} - "
                f"Reward: {metrics['episode_reward']:.2f}, "
                f"Length: {metrics['episode_length']}, "
                f"Success: {metrics['success']}"
            )
    
    def train(self, checkpoint_path: Optional[str] = None):
        """Train the algorithm.
        
        Args:
            checkpoint_path (Optional[str]): Path to checkpoint to resume from
        """
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        try:
            # Training loop
            with tqdm(total=self.max_episodes, initial=self.episode) as pbar:
                while self.episode < self.max_episodes:
                    # Train one episode
                    metrics = self.train_episode()
                    self.episode += 1
                    self.total_steps += metrics["episode_length"]
                    
                    # Log metrics
                    self.log_metrics(metrics, self.episode)
                    
                    # Save checkpoint
                    if self.episode % self.save_interval == 0:
                        is_best = metrics["episode_reward"] > self.best_reward
                        if is_best:
                            self.best_reward = metrics["episode_reward"]
                        self.save_checkpoint(self.episode, is_best)
                    
                    # Evaluate
                    if self.episode % self.eval_interval == 0:
                        eval_metrics = self.evaluator.evaluate()
                        if self.config.get("use_wandb", False):
                            wandb.log(
                                {"eval/" + k: v for k, v in eval_metrics.items()},
                                step=self.episode
                            )
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f"{metrics['episode_reward']:.2f}",
                        'success': metrics['success']
                    })
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Final evaluation
            self.evaluator.evaluate()
            
            # Save final checkpoint
            self.save_checkpoint(self.episode)
            
            # Clean up
            if self.config.get("use_wandb", False):
                wandb.finish() 