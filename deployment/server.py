from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from pathlib import Path
import yaml
import json
import time
import logging
from datetime import datetime
import psutil
import GPUtil

from ..algorithms.base import MARLAlgorithm
from ..utils.config import Config

class ObservationRequest(BaseModel):
    """Request model for observations."""
    observations: Dict[str, List[float]] = Field(..., description="Observations for each agent")
    agent_ids: List[str] = Field(..., description="List of agent IDs")
    request_id: Optional[str] = Field(None, description="Optional request identifier")

class ActionResponse(BaseModel):
    """Response model for actions."""
    actions: Dict[str, int] = Field(..., description="Actions for each agent")
    info: Dict[str, Any] = Field(..., description="Additional information")
    latency: float = Field(..., description="Processing time in milliseconds")
    request_id: Optional[str] = Field(None, description="Request identifier if provided")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    uptime: float = Field(..., description="Server uptime in seconds")
    gpu_usage: Optional[Dict[str, float]] = Field(None, description="GPU utilization if available")
    cpu_usage: float = Field(..., description="CPU utilization percentage")
    memory_usage: float = Field(..., description="Memory utilization percentage")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    request_count: int = Field(..., description="Total number of requests processed")
    average_latency: float = Field(..., description="Average request latency in milliseconds")

class Server:
    """FastAPI server for serving MARL models."""
    
    def __init__(self, config_path: str):
        """Initialize the server.
        
        Args:
            config_path (str): Path to server configuration file
        """
        self.config = Config.load(config_path)
        self.app = FastAPI(
            title="MARL Model Server",
            description="API for Multi-Agent Reinforcement Learning model inference",
            version="1.0.0"
        )
        
        # Setup logging
        self._setup_logging()
        
        # Setup CORS
        self._setup_cors()
        
        # Load algorithm
        self._load_algorithm()
        
        # Register routes
        self._setup_routes()
        
        # Initialize metrics
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0
        self.request_times = []
        self.error_count = 0
        
        # Initialize monitoring
        self.last_gpu_check = 0
        self.gpu_check_interval = 5  # seconds
        self.gpu_stats = None
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_file = log_config.get("file", "server.log")
        log_level = log_config.get("level", "INFO")
        
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MARL-Server")
    
    def _setup_cors(self):
        """Setup CORS middleware."""
        cors_config = self.config.get("cors", {})
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allow_origins", ["*"]),
            allow_credentials=cors_config.get("allow_credentials", True),
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"])
        )
    
    def _load_algorithm(self):
        """Load the MARL algorithm and model."""
        try:
            algorithm_config = self.config["algorithm"]
            algorithm_type = algorithm_config["type"]
            
            if algorithm_type == "qmix":
                from ..algorithms.qmix.qmix import QMIX
                self.algorithm = QMIX(algorithm_config)
            elif algorithm_type == "mappo":
                from ..algorithms.mappo.mappo import MAPPO
                self.algorithm = MAPPO(algorithm_config)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            
            # Load model checkpoint
            checkpoint_path = self.config["model_checkpoint"]
            self.algorithm.load(checkpoint_path)
            self.algorithm.eval()  # Set to evaluation mode
            
            self.logger.info(f"Successfully loaded {algorithm_type} model from {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load algorithm: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log incoming requests and their processing time."""
            start_time = time.time()
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            self.logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.2f}ms")
            return response
        
        @self.app.post("/predict", response_model=ActionResponse)
        async def predict(request: ObservationRequest, background_tasks: BackgroundTasks):
            try:
                start_time = time.time()
                
                # Convert observations to tensors
                observations = {
                    agent_id: torch.FloatTensor(obs).to(self.algorithm.device)
                    for agent_id, obs in request.observations.items()
                }
                
                # Get actions from algorithm
                with torch.no_grad():
                    actions, info = self.algorithm.select_actions(
                        observations=observations,
                        agent_ids=request.agent_ids,
                        explore=False
                    )
                
                # Convert actions to Python types
                actions = {
                    agent_id: action.item() if isinstance(action, torch.Tensor) else int(action)
                    for agent_id, action in actions.items()
                }
                
                # Convert info values to Python types
                info = json.loads(json.dumps(info, default=lambda x: x.item() if isinstance(x, torch.Tensor) else x))
                
                # Calculate latency
                latency = (time.time() - start_time) * 1000
                
                # Update metrics in background
                background_tasks.add_task(self._update_metrics, latency)
                
                return ActionResponse(
                    actions=actions,
                    info=info,
                    latency=latency,
                    request_id=request.request_id
                )
            
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error processing request: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            try:
                # Update GPU stats if needed
                current_time = time.time()
                if current_time - self.last_gpu_check > self.gpu_check_interval:
                    self.gpu_stats = self._get_gpu_stats()
                    self.last_gpu_check = current_time
                
                return HealthResponse(
                    status="healthy",
                    uptime=time.time() - self.start_time,
                    gpu_usage=self.gpu_stats,
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    model_info={
                        "type": self.config["algorithm"]["type"],
                        "checkpoint": self.config["model_checkpoint"],
                        "device": str(self.algorithm.device)
                    },
                    request_count=self.request_count,
                    average_latency=self.total_latency / max(1, self.request_count)
                )
            except Exception as e:
                self.logger.error(f"Error in health check: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get detailed server metrics."""
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "average_latency": self.total_latency / max(1, self.request_count),
                "latency_percentiles": self._compute_latency_percentiles(),
                "uptime": time.time() - self.start_time,
                "gpu_stats": self.gpu_stats,
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory()._asdict()
            }
    
    def _update_metrics(self, latency: float):
        """Update server metrics.
        
        Args:
            latency (float): Request processing time in milliseconds
        """
        self.request_count += 1
        self.total_latency += latency
        self.request_times.append(latency)
        
        # Keep only last 1000 request times
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
    
    def _compute_latency_percentiles(self) -> Dict[str, float]:
        """Compute latency percentiles."""
        if not self.request_times:
            return {}
        
        times = np.array(self.request_times)
        return {
            "p50": float(np.percentile(times, 50)),
            "p90": float(np.percentile(times, 90)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99))
        }
    
    def _get_gpu_stats(self) -> Optional[Dict[str, float]]:
        """Get GPU utilization statistics."""
        try:
            gpus = GPUtil.getGPUs()
            return {
                f"gpu_{i}": {
                    "utilization": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                }
                for i, gpu in enumerate(gpus)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU stats: {str(e)}")
            return None
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server.
        
        Args:
            host (str): Host address
            port (int): Port number
        """
        import uvicorn
        
        # Override host/port from config if provided
        host = self.config.get("host", host)
        port = self.config.get("port", port)
        
        self.logger.info(f"Starting MARL model server on {host}:{port}")
        
        uvicorn_config = self.config.get("uvicorn", {})
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=uvicorn_config.get("workers", 1),
            log_level=uvicorn_config.get("log_level", "info"),
            timeout_keep_alive=uvicorn_config.get("timeout_keep_alive", 30),
            limit_concurrency=uvicorn_config.get("limit_concurrency", None),
            limit_max_requests=uvicorn_config.get("limit_max_requests", None)
        ) 