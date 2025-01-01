import logging
from typing import Optional, Dict, Any
from pathlib import Path
import sys
from datetime import datetime
import json
import csv

class Logger:
    """Unified logging utility."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        log_dir: str,
        level: str = "INFO"
    ):
        """Initialize logger.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            experiment_name (str): Name of the experiment
            log_dir (str): Directory for log files
            level (str): Logging level
        """
        self.config = config
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(getattr(logging, level))
        
        # Create handlers
        self._setup_handlers()
        
        # Create metric loggers
        self.metric_files = {}
        self._setup_metric_loggers()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
    
    def _setup_metric_loggers(self):
        """Setup CSV loggers for different metric types."""
        metric_types = ['train', 'eval', 'test']
        
        for metric_type in metric_types:
            csv_file = self.log_dir / f"{metric_type}_metrics.csv"
            self.metric_files[metric_type] = csv_file
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        metric_type: str = 'train'
    ):
        """Log metrics to CSV file.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
            step (int): Current step/episode
            metric_type (str): Type of metrics (train/eval/test)
        """
        csv_file = self.metric_files[metric_type]
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Write header if file doesn't exist
        if not csv_file.exists():
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
        
        # Append metrics
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
    
    def log_config(self):
        """Log configuration."""
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg) 