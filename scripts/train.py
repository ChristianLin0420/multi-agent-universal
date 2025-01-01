import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import logging
import sys
import os
from datetime import datetime
import random
import numpy as np
import wandb

from marl_framework.utils.config import Config
from marl_framework.algorithms.qmix.qmix import QMIX
from marl_framework.algorithms.mappo.mappo import MAPPO
from marl_framework.environments.smac.env import SMACEnvironment
from marl_framework.environments.google_football.env import GoogleFootballEnvironment
from marl_framework.environments.mpe.env import MPEEnvironment
from marl_framework.training.trainer import Trainer
from marl_framework.training.distributed.ddp_trainer import DDPTrainer

def setup_logging(config: dict, experiment_name: str) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config (dict): Configuration dictionary
        experiment_name (str): Name of the experiment
        
    Returns:
        logging.Logger: Configured logger
    """
    log_dir = Path(config.get("output_dir", "experiments")) / experiment_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("MARL-Training")

def set_random_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_environment(config: dict, logger: logging.Logger) -> Any:
    """Setup training environment.
    
    Args:
        config (dict): Configuration dictionary
        logger (logging.Logger): Logger instance
        
    Returns:
        Environment instance
    """
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
    """Setup training algorithm.
    
    Args:
        config (dict): Configuration dictionary
        logger (logging.Logger): Logger instance
        
    Returns:
        Algorithm instance
    """
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
    """Setup Weights & Biases logging.
    
    Args:
        config (dict): Configuration dictionary
        experiment_name (str): Name of the experiment
    """
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "marl_framework"),
            name=experiment_name,
            config=config,
            dir=str(Path(config.get("output_dir", "experiments")) / experiment_name)
        )

def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    """Worker function for distributed training.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Load configuration
    config = Config.load(args.config)
    
    # Setup experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment_name']}_{timestamp}"
    
    # Setup logging
    logger = setup_logging(config, experiment_name)
    
    try:
        # Set random seed
        seed = config.get("seed", 42) + rank
        set_random_seed(seed)
        
        # Setup environment and algorithm
        env = setup_environment(config, logger)
        algorithm = setup_algorithm(config, logger)
        
        # Setup wandb for rank 0 only
        if rank == 0:
            setup_wandb(config, experiment_name)
        
        # Setup trainer
        if args.distributed:
            trainer = DDPTrainer(
                config=config,
                algorithm=algorithm,
                env=env,
                experiment_name=experiment_name,
                local_rank=rank,
                world_size=world_size
            )
        else:
            trainer = Trainer(
                config=config,
                algorithm=algorithm,
                env=env,
                experiment_name=experiment_name
            )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in worker {rank}: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if args.distributed:
            dist.destroy_process_group()
        if rank == 0 and config.get("use_wandb", False):
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train MARL agents")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    
    # Set up distributed training if requested
    if args.distributed:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise ValueError("Distributed training requires at least 2 GPUs")
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(config.get("dist_port", 29500))
        
        mp.spawn(
            train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        train_worker(0, 1, args)

if __name__ == "__main__":
    main() 