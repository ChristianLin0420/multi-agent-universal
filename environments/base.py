from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np

class MARLEnvironment(ABC):
    """Base class for all MARL environment wrappers.
    
    This abstract class defines the interface that all environment
    wrappers must implement for compatibility with the training framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the environment wrapper.
        
        Args:
            config (Dict[str, Any]): Environment configuration parameters
        """
        self.config = config
        self.agent_ids = []
        
    @abstractmethod
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment and return initial observations.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observations for each agent
                                               and additional info
        """
        pass
    
    @abstractmethod
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
            Tuple containing:
                - Dict[str, np.ndarray]: New observations for each agent
                - Dict[str, float]: Rewards for each agent
                - Dict[str, bool]: Done flags for each agent
                - Dict: Additional info
        """
        pass
    
    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the environment for visualization.
        
        Returns:
            np.ndarray: RGB array of the rendered frame
        """
        pass
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get the observation space for each agent.
        
        Returns:
            Dict[str, Any]: Observation space specifications
        """
        raise NotImplementedError
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Get the action space for each agent.
        
        Returns:
            Dict[str, Any]: Action space specifications
        """
        raise NotImplementedError 