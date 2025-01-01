import argparse
from pathlib import Path
import torch
import yaml
import logging
import sys
from datetime import datetime
import random
import numpy as np
import wandb
import json

from marl_framework.utils.config import Config
from marl_framework.algorithms.qmix.qmix import QMIX
from marl_framework.algorithms.mappo.mappo import MAPPO
from marl_framework.environments.smac.env import SMACEnvironment
from marl_framework.environments.google_football.env import GoogleFootballEnvironment
from marl_framework.environments.mpe.env import MPEEnvironment
from marl_framework.evaluation.evaluator import Evaluator

def setup_logging(config: dict, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(config.get("output_dir", "experiments")) / experiment_name / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("MARL-Evaluation")

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_environment(config: dict, logger: logging.Logger) -> Any:
    """Setup evaluation environment."""
    env_config = config["environment"]
    env_type = env_config["type"]
    
    logger.info(f"Setting up environment: {env_type}")
    
    if env_type == "smac":
        return SMACEnvironment(env_config)
    elif env_type == "football":
        return GoogleFootballEnvironment(env_config)
    elif env_type == "mpe":
        return MPEEnvironment(env_config)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

def setup_algorithm(config: dict, logger: logging.Logger) -> Any:
    """Setup evaluation algorithm."""
    algo_config = config["algorithm"]
    algo_type = algo_config["type"]
    
    logger.info(f"Setting up algorithm: {algo_type}")
    
    if algo_type == "qmix":
        return QMIX(algo_config)
    elif algo_type == "mappo":
        return MAPPO(algo_config)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

def setup_wandb(config: dict, experiment_name: str):
    """Setup Weights & Biases logging."""
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "marl_framework"),
            name=f"{experiment_name}_eval",
            config=config,
            dir=str(Path(config.get("output_dir", "experiments")) / experiment_name)
        )

def save_eval_config(config: dict, args: argparse.Namespace, eval_dir: Path):
    """Save evaluation configuration."""
    eval_config = {
        "base_config": config,
        "checkpoint_path": args.checkpoint,
        "render": args.render,
        "save_videos": args.save_videos,
        "eval_episodes": config.get("n_episodes", 100),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(eval_dir / "eval_config.json", 'w') as f:
        json.dump(eval_config, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MARL agents")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save_videos", action="store_true", help="Save evaluation videos")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    
    # Setup experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment_name']}_eval_{timestamp}"
    
    # Create evaluation directory
    eval_dir = Path(config.get("output_dir", "experiments")) / experiment_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config, experiment_name)
    
    try:
        # Set random seed if provided
        if args.seed is not None:
            set_random_seed(args.seed)
            logger.info(f"Set random seed to {args.seed}")
        
        # Save evaluation configuration
        save_eval_config(config, args, eval_dir)
        
        # Update config with evaluation settings
        config["render"] = args.render
        config["save_videos"] = args.save_videos
        
        # Setup environment and algorithm
        env = setup_environment(config, logger)
        algorithm = setup_algorithm(config, logger)
        
        # Setup wandb
        setup_wandb(config, experiment_name)
        
        # Setup evaluator
        evaluator = Evaluator(
            config=config,
            algorithm=algorithm,
            env=env,
            experiment_name=experiment_name
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate(checkpoint_path=args.checkpoint)
        
        # Log results
        logger.info("\nEvaluation Results:")
        logger.info("=" * 50)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save results
        results_file = eval_dir / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"\nResults saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise
    finally:
        if config.get("use_wandb", False):
            wandb.finish()

if __name__ == "__main__":
    main() 