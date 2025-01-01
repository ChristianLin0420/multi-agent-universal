import argparse
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import logging
import sys
import onnx
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import time
from tqdm import tqdm

from marl_framework.utils.config import Config
from marl_framework.algorithms.qmix.qmix import QMIX
from marl_framework.algorithms.mappo.mappo import MAPPO

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelConverter:
    """Convert trained MARL models to TensorRT format."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model converter.
        
        Args:
            config (Dict[str, Any]): Conversion configuration
        """
        self.config = config
        self.setup_logging()
        
        # Load algorithm
        self.algorithm = self._load_algorithm()
        
        # TensorRT settings
        self.workspace_size = config.get("workspace_size", 1 << 30)  # 1GB
        self.fp16_mode = config.get("fp16_mode", False)
        self.int8_mode = config.get("int8_mode", False)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.min_batch_size = config.get("min_batch_size", 1)
        self.optimal_batch_size = config.get("optimal_batch_size", 16)
        
        # Calibration settings
        self.calibration_batch_size = config.get("calibration_batch_size", 128)
        self.calibration_batches = config.get("calibration_batches", 10)
        
        # Performance profiling
        self.profile_shapes = config.get("profile_shapes", True)
        self.timing_cache = config.get("timing_cache", True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get("log_dir", "logs/conversion"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"conversion_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("ModelConverter")
    
    def _load_algorithm(self) -> Any:
        """Load algorithm based on config."""
        algo_type = self.config["algorithm"]["type"]
        algo_map = {
            "qmix": QMIX,
            "mappo": MAPPO
        }
        
        if algo_type not in algo_map:
            raise ValueError(f"Unsupported algorithm: {algo_type}")
        
        return algo_map[algo_type](self.config["algorithm"])
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.algorithm.load_state_dict(checkpoint["algorithm_state"])
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
            # Save model info
            self.model_info = {
                "checkpoint_path": checkpoint_path,
                "episode": checkpoint.get("episode", 0),
                "total_steps": checkpoint.get("total_steps", 0),
                "best_reward": checkpoint.get("best_reward", 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    def _optimize_onnx(self, model_path: str) -> str:
        """Optimize ONNX model for inference.
        
        Args:
            model_path (str): Path to ONNX model
            
        Returns:
            str: Path to optimized model
        """
        import onnxoptimizer
        
        model = onnx.load(model_path)
        passes = [
            "eliminate_unused_initializer",
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_bn_into_conv"
        ]
        
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        optimized_path = str(Path(model_path).with_suffix('.opt.onnx'))
        onnx.save(optimized_model, optimized_path)
        
        return optimized_path
    
    def export_onnx(self, output_path: str, input_shape: tuple):
        """Export model to ONNX format.
        
        Args:
            output_path (str): Path to save ONNX model
            input_shape (tuple): Input tensor shape
        """
        self.algorithm.eval()
        dummy_input = torch.randn(input_shape)
        
        # Export with dynamic axes
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        
        try:
            torch.onnx.export(
                self.algorithm,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=11,
                do_constant_folding=True,
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Optimize model
            if self.config.get("optimize_onnx", True):
                output_path = self._optimize_onnx(output_path)
            
            self.logger.info(f"Exported and verified ONNX model to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export ONNX model: {str(e)}")
            raise
    
    def _create_optimization_profile(
        self,
        builder: trt.Builder,
        network: trt.INetworkDefinition,
        config: trt.IBuilderConfig
    ):
        """Create optimization profile for dynamic shapes.
        
        Args:
            builder (trt.Builder): TensorRT builder
            network (trt.INetworkDefinition): Network definition
            config (trt.IBuilderConfig): Builder configuration
        """
        profile = builder.create_optimization_profile()
        
        input_name = "input"
        input_shape = network.get_input(0).shape
        
        # Set shape ranges
        min_shape = (self.min_batch_size,) + input_shape[1:]
        opt_shape = (self.optimal_batch_size,) + input_shape[1:]
        max_shape = (self.max_batch_size,) + input_shape[1:]
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    def build_engine(self, onnx_path: str, engine_path: str):
        """Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path (str): Path to ONNX model
            engine_path (str): Path to save TensorRT engine
        """
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self.logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size
        
        # Set precision flags
        if self.fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("Enabled FP16 precision")
        
        if self.int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            self.logger.info("Enabled INT8 precision")
            
            # Add INT8 calibrator if needed
            if hasattr(self, 'calibrator'):
                config.int8_calibrator = self.calibrator
        
        # Create optimization profile
        if self.profile_shapes:
            self._create_optimization_profile(builder, network, config)
        
        # Enable timing cache
        if self.timing_cache:
            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, False)
        
        # Build and save engine
        try:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"Successfully built and saved TensorRT engine to {engine_path}")
            
            # Save conversion info
            info_path = Path(engine_path).with_suffix('.json')
            self._save_conversion_info(info_path)
            
        except Exception as e:
            self.logger.error(f"Failed to build engine: {str(e)}")
            raise
    
    def _save_conversion_info(self, info_path: Path):
        """Save conversion information.
        
        Args:
            info_path (Path): Path to save info
        """
        info = {
            "model_info": self.model_info,
            "conversion_settings": {
                "fp16_mode": self.fp16_mode,
                "int8_mode": self.int8_mode,
                "workspace_size": self.workspace_size,
                "max_batch_size": self.max_batch_size
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
    
    def convert(self, checkpoint_path: str, output_dir: str, input_shape: tuple):
        """Convert model from checkpoint to TensorRT engine.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            output_dir (str): Directory to save converted models
            input_shape (tuple): Input tensor shape
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load checkpoint
            self.load_checkpoint(checkpoint_path)
            
            # Export to ONNX
            onnx_path = output_dir / "model.onnx"
            onnx_path = self.export_onnx(str(onnx_path), input_shape)
            
            # Build TensorRT engine
            engine_path = output_dir / "model.trt"
            self.build_engine(onnx_path, str(engine_path))
            
            self.logger.info("Model conversion completed successfully")
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Convert MARL model to TensorRT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 mode")
    parser.add_argument("--optimize", action="store_true", help="Enable ONNX optimization")
    parser.add_argument("--workspace_size", type=int, help="TensorRT workspace size")
    args = parser.parse_args()
    
    # Load config
    config = Config.load(args.config)
    
    # Update config with conversion settings
    config.update({
        "fp16_mode": args.fp16,
        "int8_mode": args.int8,
        "optimize_onnx": args.optimize
    })
    
    if args.workspace_size:
        config["workspace_size"] = args.workspace_size
    
    # Create converter
    converter = ModelConverter(config)
    
    # Determine input shape based on environment
    env_config = config["environment"]
    input_shapes = {
        "smac": (args.batch_size, env_config.get("state_dim", 120)),
        "football": (args.batch_size, env_config.get("state_dim", 115)),
        "mpe": (args.batch_size, env_config.get("state_dim", 64))
    }
    
    input_shape = input_shapes.get(
        env_config["type"],
        (args.batch_size, env_config.get("state_dim", 64))
    )
    
    # Convert model
    converter.convert(args.checkpoint, args.output_dir, input_shape)

if __name__ == "__main__":
    main() 