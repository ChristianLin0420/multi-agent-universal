# Multi-Agent-Universal 🤖

> A comprehensive Multi-Agent Reinforcement Learning framework for research and development, featuring distributed training and multiple environment support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🌟 Introduction

MARL Framework is a scalable and modular platform designed for Multi-Agent Reinforcement Learning research and applications. It provides a unified interface for training, evaluating, and deploying MARL agents across various environments.

### Why MARL Framework?

- **Unified Interface**: Consistent API across different algorithms and environments
- **Scalability**: From single-GPU training to distributed multi-GPU setups
- **Modularity**: Easy to extend with new algorithms and environments
- **Research Ready**: Built-in logging, visualization, and experiment management
- **Production Grade**: Deployment tools for model serving

### Key Design Principles

1. **Flexibility**: 
   - Plug-and-play architecture for algorithms and environments
   - Customizable reward functions and observation spaces
   - Configurable training parameters

2. **Reproducibility**:
   - Comprehensive logging system
   - Configuration management
   - Experiment tracking
   - Seed control

3. **Performance**:
   - Optimized data handling
   - Efficient experience replay
   - Distributed training support
   - GPU acceleration

### Core Components

```
marl_framework/
├── algorithms/          # MARL algorithms (QMIX, MAPPO)
├── environments/        # Environment wrappers
├── training/           # Training infrastructure
├── evaluation/         # Evaluation tools
├── deployment/         # Deployment utilities
└── utils/             # Common utilities
```

### Supported Tasks

1. **Cooperative Navigation**
   - Multi-agent pathfinding
   - Formation control
   - Swarm behavior

2. **Team Sports**
   - Football scenarios
   - Strategic team play
   - Role-based coordination

3. **Combat Scenarios**
   - StarCraft II battles
   - Tactical decision making
   - Resource management

## 🌟 Features

Our framework provides a comprehensive suite of features designed to support both research and practical applications in multi-agent reinforcement learning.

### 🎮 Supported Environments
Each environment is carefully wrapped to provide consistent interfaces while maintaining their unique characteristics:

- **StarCraft II (SMAC)**
  - Challenging micro-management scenarios
  - Rich observation and state spaces
  - Multiple battle scenarios (3m, 8m, 2s3z, etc.)
  - Configurable difficulty levels (1-7)
  - Built-in reward shaping options
  - Detailed battle statistics tracking

- **Google Research Football**
  - Realistic football simulation
  - Multiple game formats (3v3, 5v5, 11v11)
  - Academy learning scenarios for skill development
  - Customizable rewards and scenarios
  - Built-in performance metrics
  - Replay saving and analysis

- **Multi-Agent Particle (MPE)**
  - Lightweight 2D physics simulation
  - Perfect for algorithm prototyping
  - Cooperative/competitive scenarios
  - Customizable agent dynamics
  - Easy visualization
  - Fast execution speed

### 🧠 Algorithms
Our implemented algorithms represent state-of-the-art approaches in MARL:

- **QMIX**
  - Value-based MARL algorithm
  - Monotonic value function factorization
  - Centralized training with decentralized execution
  - Efficient experience replay system
  - Hyperparameter optimization ready

- **MAPPO**
  - Policy-based MARL algorithm
  - Multi-agent version of PPO
  - Shared critic network architecture
  - GAE advantage estimation
  - Adaptive learning rate

### 🛠️ Core Features
Essential tools and utilities for effective MARL development:

- **Distributed Training**
  - PyTorch DDP integration
  - Multi-GPU synchronization
  - Efficient batch processing
  - Customizable worker processes
  - Automatic checkpoint management

- **Visualization Tools**
  - Real-time training curves
  - Agent trajectory visualization
  - Interactive environment rendering
  - Custom metric plotting
  - WandB integration

## 📦 Installation

Our installation process is streamlined to get you started quickly while ensuring all dependencies are properly set up.

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (for distributed training)
- StarCraft II (for SMAC environment)
- Git

```bash
# Clone repository
git clone https://github.com/username/marl_framework.git
cd multi-agent-universal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install environment dependencies
bash install_environments.sh
```

## 🚀 Quick Start

Get started with MARL Framework in minutes. Our quick start guide covers the essential workflows:

### Training
The training system supports both single-GPU and distributed training modes with automatic logging and checkpointing:

```bash
# Single GPU training
python scripts/train.py \
    --config configs/qmix_smac.yaml \
    --experiment_name my_first_run

# Distributed training (multi-GPU)
python scripts/train.py \
    --config configs/mappo_football.yaml \
    --distributed \
    --world_size 2
```

### Advanced Training Options
```bash
# Training with custom hyperparameters
python scripts/train.py \
    --config configs/qmix_smac.yaml \
    --experiment_name custom_run \
    --override_config training.batch_size=64 algorithm.lr=0.0001

# Training with specific GPU devices
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
    --config configs/mappo_football.yaml \
    --distributed \
    --world_size 2
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint path/to/model.pt \
    --render
```

### Deployment

```bash
# Start model server
python scripts/deploy.py \
    --config configs/deploy.yaml \
    --port 8000
```

## 📝 Configuration Guide

Our configuration system is designed to be flexible and maintainable, supporting inheritance and override capabilities.

### Configuration Hierarchy
```
configs/
├── base/                  # Base configurations
│   ├── algorithm/        # Algorithm-specific bases
│   └── environment/      # Environment-specific bases
├── experiments/          # Experiment configurations
└── deployment/          # Deployment configurations
```

### Basic Structure
```yaml
# configs/example.yaml
algorithm:
  type: "qmix"
  hidden_dim: 64
  lr: 0.001

environment:
  type: "smac"
  map_name: "3m"
  difficulty: "7"

training:
  max_episodes: 10000
  batch_size: 32
```

### Config Inheritance
```yaml
# Base config (base.yaml)
algorithm:
  type: "qmix"
  gamma: 0.99

# Extended config (extended.yaml)
includes: ["base.yaml"]
environment:
  type: "smac"
```

## 📊 Experiment Management

Our experiment management system helps track, analyze, and compare different runs:

### Features
- Automatic experiment versioning
- Comprehensive metric logging
- Resource usage tracking
- Visualization tools
- Easy experiment comparison

### Directory Structure
```
experiments/
├── run_name/
│   ├── checkpoints/     # Model checkpoints
│   ├── logs/           # Training logs
│   ├── plots/          # Visualizations
│   └── config.json     # Experiment config
```

### Logging Example
```python
from marl_framework.utils.logger import Logger

logger = Logger(
    config=config,
    experiment_name="my_experiment"
)

# Log metrics
logger.log_metrics(
    metrics={"reward": 10.5},
    step=100
)
```

## 🔧 Development Guide

### Adding New Algorithm

1. Create directory structure:
```bash
mkdir -p algorithms/new_algo
```

2. Implement algorithm:
```python
from ..base import MARLAlgorithm

class NewAlgo(MARLAlgorithm):
    def __init__(self, config):
        super().__init__(config)
```

### Adding New Environment

```python
from ..base import MARLEnvironment

class NewEnvironment(MARLEnvironment):
    def __init__(self, config):
        super().__init__(config)
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```yaml
# Reduce batch size in config
training:
  batch_size: 16
  grad_accumulation_steps: 2
```

#### Distributed Training
```bash
# Test NCCL installation
python -c "import torch; print(torch.cuda.nccl.version())"
```

## 📚 Documentation

For detailed documentation, visit our [Wiki](https://github.com/username/marl_framework/wiki).

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- StarCraft II environment by DeepMind
- Google Research Football
- OpenAI MPE