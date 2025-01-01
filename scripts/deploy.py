import argparse
from pathlib import Path
import torch
import yaml
import logging
import sys
import os
from datetime import datetime
import json
import signal
import psutil
import GPUtil
from typing import Optional
import uvicorn
from prometheus_client import start_http_server, Counter, Gauge, Histogram

from marl_framework.utils.config import Config
from marl_framework.deployment.server import Server

# Prometheus metrics
REQUEST_COUNT = Counter('marl_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('marl_request_latency_seconds', 'Request latency in seconds')
ERROR_COUNT = Counter('marl_errors_total', 'Total number of errors')
GPU_UTILIZATION = Gauge('marl_gpu_utilization_percent', 'GPU utilization percentage')
MEMORY_USAGE = Gauge('marl_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('marl_cpu_usage_percent', 'CPU usage percentage')
MODEL_INFERENCE_TIME = Histogram('marl_model_inference_seconds', 'Model inference time in seconds')

class DeploymentManager:
    """Manager for model deployment server."""
    
    def __init__(self, config_path: str):
        """Initialize deployment manager.
        
        Args:
            config_path (str): Path to deployment configuration
        """
        self.config = Config.load(config_path)
        self.setup_logging()
        self.server = None
        self.metrics_server = None
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_dir = Path(log_config.get("dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"server_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("MARL-Server")
    
    def setup_metrics_server(self):
        """Setup Prometheus metrics server."""
        metrics_config = self.config.get("metrics", {})
        if metrics_config.get("enabled", False):
            metrics_port = metrics_config.get("port", 9090)
            start_http_server(metrics_port)
            self.logger.info(f"Started metrics server on port {metrics_port}")
    
    def update_metrics(self):
        """Update system metrics."""
        try:
            # Update GPU metrics
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                GPU_UTILIZATION.labels(gpu=gpu.id).set(gpu.load * 100)
            
            # Update memory metrics
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # Update CPU metrics
            CPU_USAGE.set(psutil.cpu_percent())
            
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {str(e)}")
    
    def validate_config(self):
        """Validate deployment configuration."""
        required_fields = ["algorithm", "model_checkpoint"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        if not Path(self.config["model_checkpoint"]).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.config['model_checkpoint']}")
    
    def save_deployment_info(self):
        """Save deployment information."""
        deploy_info = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cpu_count": os.cpu_count()
            }
        }
        
        info_path = Path("deployment_info.json")
        with open(info_path, 'w') as f:
            json.dump(deploy_info, f, indent=4)
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received shutdown signal. Cleaning up...")
        if self.server:
            self.server.close()
        sys.exit(0)
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Run the deployment server.
        
        Args:
            host (Optional[str]): Server host address
            port (Optional[int]): Server port number
        """
        try:
            # Validate configuration
            self.validate_config()
            
            # Save deployment information
            self.save_deployment_info()
            
            # Setup metrics server
            self.setup_metrics_server()
            
            # Create and run server
            self.server = Server(self.config)
            
            # Override host/port if provided
            host = host or self.config.get("host", "0.0.0.0")
            port = port or self.config.get("port", 8000)
            
            self.logger.info(f"Starting MARL model server on {host}:{port}")
            self.logger.info("Available endpoints:")
            self.logger.info("  - POST /predict : Get actions for given observations")
            self.logger.info("  - GET  /health : Check server health")
            self.logger.info("  - GET  /metrics : Get server metrics")
            
            # Run server
            uvicorn_config = self.config.get("uvicorn", {})
            uvicorn.run(
                self.server.app,
                host=host,
                port=port,
                workers=uvicorn_config.get("workers", 1),
                log_level=uvicorn_config.get("log_level", "info"),
                timeout_keep_alive=uvicorn_config.get("timeout_keep_alive", 30),
                limit_concurrency=uvicorn_config.get("limit_concurrency", None),
                limit_max_requests=uvicorn_config.get("limit_max_requests", None)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description="Deploy MARL model server")
    parser.add_argument("--config", type=str, required=True, help="Path to server config file")
    parser.add_argument("--host", type=str, default=None, help="Server host address")
    parser.add_argument("--port", type=int, default=None, help="Server port number")
    args = parser.parse_args()
    
    try:
        manager = DeploymentManager(args.config)
        manager.run(host=args.host, port=args.port)
    except Exception as e:
        logging.error(f"Deployment failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 