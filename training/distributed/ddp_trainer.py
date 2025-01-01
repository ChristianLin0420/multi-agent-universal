import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional
import os

from ..trainer import Trainer
from ...algorithms.base import MARLAlgorithm
from ...environments.base import MARLEnvironment

class DDPTrainer(Trainer):
    """Distributed trainer using DistributedDataParallel."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        algorithm: MARLAlgorithm,
        env: MARLEnvironment,
        experiment_name: str,
        local_rank: int,
        world_size: int
    ):
        """Initialize distributed trainer.
        
        Args:
            config (Dict[str, Any]): Training configuration
            algorithm (MARLAlgorithm): Algorithm instance
            env (MARLEnvironment): Environment instance
            experiment_name (str): Name of the experiment
            local_rank (int): Local process rank
            world_size (int): Total number of processes
        """
        # Initialize process group
        self.setup_distributed(local_rank, world_size)
        
        # Move algorithm to correct device
        algorithm.to(f"cuda:{local_rank}")
        
        # Initialize base trainer
        super().__init__(config, algorithm, env, experiment_name)
        
        # Wrap algorithm in DDP
        self.algorithm = DDP(
            algorithm,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        
        self.local_rank = local_rank
        self.world_size = world_size
    
    def setup_distributed(self, local_rank: int, world_size: int):
        """Setup distributed training.
        
        Args:
            local_rank (int): Local process rank
            world_size (int): Total number of processes
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank
        )
        
        torch.cuda.set_device(local_rank)
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode with distributed synchronization."""
        # Synchronize random seeds across processes
        seed = torch.randint(0, 1000000, (1,)).to(f"cuda:{self.local_rank}")
        dist.broadcast(seed, src=0)
        torch.manual_seed(seed.item())
        
        metrics = super().train_episode()
        
        # Synchronize metrics across processes
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                tensor = torch.tensor(metrics[key]).to(f"cuda:{self.local_rank}")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics[key] = tensor.item() / self.world_size
        
        return metrics
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save checkpoint only from rank 0 process."""
        if self.local_rank == 0:
            super().save_checkpoint(episode, is_best)
    
    def log_metrics(self, metrics: Dict[str, float], episode: int):
        """Log metrics only from rank 0 process."""
        if self.local_rank == 0:
            super().log_metrics(metrics, episode)
    
    def cleanup(self):
        """Clean up distributed process group."""
        dist.destroy_process_group() 