from typing import Dict, Any, Optional, Union
import yaml
import json
from pathlib import Path
import os
from copy import deepcopy

class Config:
    """Configuration management utility."""
    
    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            path (Union[str, Path]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # Process includes
        if 'includes' in config:
            config = Config._process_includes(config, path.parent)
        
        # Validate configuration
        Config._validate_config(config)
        
        return config
    
    @staticmethod
    def _process_includes(config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """Process included configuration files.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            base_path (Path): Base path for relative includes
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        includes = config.pop('includes', [])
        if not isinstance(includes, list):
            includes = [includes]
        
        final_config = {}
        for include in includes:
            include_path = base_path / include
            included_config = Config.load(include_path)
            Config._deep_update(final_config, included_config)
        
        Config._deep_update(final_config, config)
        return final_config
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively update dictionary.
        
        Args:
            base_dict (Dict[str, Any]): Base dictionary to update
            update_dict (Dict[str, Any]): Dictionary with updates
        """
        for key, value in update_dict.items():
            if (
                key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)
            ):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = deepcopy(value)
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """Validate configuration structure.
        
        Args:
            config (Dict[str, Any]): Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {
            'algorithm': ['type'],
            'environment': ['type'],
            'training': ['max_episodes', 'max_steps']
        }
        
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}") 