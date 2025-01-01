from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import wandb
from datetime import datetime

def plot_episode_metrics(
    metrics: Dict[str, List[float]],
    save_path: Path,
    title: Optional[str] = None,
    use_wandb: bool = False
):
    """Plot episode metrics.
    
    Args:
        metrics (Dict[str, List[float]]): Dictionary of metrics to plot
        save_path (Path): Path to save plots
        title (Optional[str]): Plot title
        use_wandb (bool): Whether to log to wandb
    """
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values)
        plt.title(f"{title} - {metric_name}" if title else metric_name)
        plt.xlabel("Episode")
        plt.ylabel(metric_name)
        
        # Save plot
        plot_path = save_path / f"{metric_name}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({f"plot/{metric_name}": wandb.Image(str(plot_path))})

def create_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """Create video from frames.
    
    Args:
        frames (List[np.ndarray]): List of frames
        output_path (str): Path to save video
        fps (int): Frames per second
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()

def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_dir: Path,
    smoothing: float = 0.0
):
    """Plot training curves with optional smoothing.
    
    Args:
        metrics (Dict[str, List[float]]): Training metrics
        save_dir (Path): Directory to save plots
        smoothing (float): Smoothing factor (0-1)
    """
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        
        if smoothing > 0:
            smoothed_values = []
            last = values[0]
            for value in values:
                smoothed = last * smoothing + (1 - smoothing) * value
                smoothed_values.append(smoothed)
                last = smoothed
            plt.plot(values, alpha=0.3, label='Raw')
            plt.plot(smoothed_values, label='Smoothed')
            plt.legend()
        else:
            plt.plot(values)
        
        plt.title(f"Training Curve - {metric_name}")
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.grid(True)
        
        plt.savefig(save_dir / f"{metric_name}_curve.png")
        plt.close()

def plot_agent_trajectories(
    trajectories: List[Dict[str, Any]],
    save_path: Path,
    env_bounds: Optional[Dict[str, float]] = None
):
    """Plot agent trajectories.
    
    Args:
        trajectories (List[Dict[str, Any]]): List of trajectory dictionaries
        save_path (Path): Path to save plot
        env_bounds (Optional[Dict[str, float]]): Environment boundaries
    """
    plt.figure(figsize=(10, 10))
    
    for agent_id, trajectory in trajectories.items():
        positions = np.array(trajectory['positions'])
        plt.plot(positions[:, 0], positions[:, 1], label=agent_id)
        plt.scatter(positions[0, 0], positions[0, 1], marker='o', label=f"{agent_id} Start")
        plt.scatter(positions[-1, 0], positions[-1, 1], marker='x', label=f"{agent_id} End")
    
    if env_bounds:
        plt.xlim(env_bounds['x_min'], env_bounds['x_max'])
        plt.ylim(env_bounds['y_min'], env_bounds['y_max'])
    
    plt.title("Agent Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close() 