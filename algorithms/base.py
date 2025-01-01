from typing import Dict, Any, Tuple, List
import torch
import numpy as np

class MARLAlgorithm:
    """Base class for MARL algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base algorithm.
        
        Args:
            config (Dict[str, Any]): Algorithm configuration
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.train_steps = 0
    
    def select_actions(self, 
                      observations: Dict[str, np.ndarray],
                      agent_ids: List[str],
                      explore: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Select actions for all agents.
        
        Args:
            observations (Dict[str, np.ndarray]): Observations for each agent
            agent_ids (List[str]): List of agent IDs
            explore (bool): Whether to use exploration
            
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Actions and info for each agent
        """
        raise NotImplementedError
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update algorithm parameters.
        
        Args:
            batch (Dict[str, Any]): Batch of experiences
            
        Returns:
            Dict[str, float]: Update metrics
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save algorithm state.
        
        Args:
            path (str): Path to save state
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load algorithm state.
        
        Args:
            path (str): Path to load state
        """
        raise NotImplementedError 