# SNN Project

A framework for Spiking Neural Networks (SNNs) with a focus on temporal and spatial pattern recognition.

## Project Structure

```
snn_project/
├── snn_lib/                  # Core SNN functionality 
│   ├── data_loaders.py       # Data loading utilities 
│   ├── optimizers.py         # Optimizer implementation
│   ├── schedulers.py         # Learning rate schedulers
│   ├── snn_layers.py         # SNN layer implementations
│   └── utilities.py          # Core SNN utilities
│
├── models/                   # Model definitions
│   ├── base_model.py         # Base model classes 
│   └── synthetic_models.py   # Synthetic data models
│
├── training/                 # Training utilities
│   └── train_utils.py        # Consolidated training utilities
│
├── utils/                    # General utilities
│   ├── config_utils.py       # Configuration utilities
│   ├── data_utils.py         # Data processing utilities
│   ├── visualization.py      # Visualization utilities
│   └── metrics/              # Metrics collection and processing
│       ├── collector.py      # Main metrics collector
│       ├── computation.py    # Metrics computation functions
│       └── visualization.py  # Metrics visualization
│
├── experiments/              # Experiment runners
│   ├── run_synthetic_experiments.py
│   ├── snn_synthetic.yaml
│   ├── snn_synthetic_data_adapter.py
│   └── synthetic_data_generator.py
│
├── scripts/                  # Analysis and utility scripts
│   ├── analyze_noise_impact.py
│   ├── benchmark_difficulty_tiers.py
│   ├── run_visualization.py
│   └── visualize_performance.py
│
├── tests/                    # Unit tests
│
├── configs/                  # Configuration files
│   ├── experiments/
│   │   └── temporal_benchmark.yaml
│   └── default.yaml
│
├── data/                     # Data directory (may be gitignored)
│
├── results/                  # Results directory (may be gitignored)
│
├── main.py                   # Main entry point for experiments
└── README.md                 # This documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- OmegaConf
- Scikit-learn

Install the required packages:

```bash
pip install torch numpy matplotlib omegaconf scikit-learn
```

### Running Experiments

The main entry point for running experiments is `main.py`. It provides a command-line interface for training and evaluating SNN models on synthetic datasets.

#### Training a Model

```bash
python main.py --data_path ./data/temporal_tier1_easy.npz --config_file configs/default.yaml --train --visualize
```

#### Testing a Model

```bash
python main.py --data_path ./data/temporal_tier1_easy.npz --checkpoint ./results/checkpoints/snn_synthetic_temporal_tier1_easy_20250330_epoch_100.pt --test
```

#### Analyzing Model Performance

```bash
python main.py --data_path ./data/temporal_tier1_easy.npz --checkpoint ./results/checkpoints/snn_synthetic_temporal_tier1_easy_20250330_epoch_100.pt --analyze --visualize
```

### Configuration

The project uses YAML configuration files to define model hyperparameters and training settings. The default configuration is specified in `configs/default.yaml`. Experiment-specific configurations can be placed in `configs/experiments/`.

Example configuration structure:

```yaml
pytorch_seed: 42
model:
  model_type: "synthetic"
  
optimizer:
  optimizer_choice: "Adam"
  Adam:
    lr: 0.001
    
scheduler:
  scheduler_choice: "MultiStepLR"
  MultiStepLR:
    milestones: [30, 60, 90]
    gamma: 0.1
    
hyperparameters:
  batch_size: 32
  epoch: 100
  hidden_size: 128
  tau_m: 4.0
  tau_s: 1.0
  dropout_rate: 0.3
```

## Using Synthetic Data

The project includes utilities for generating synthetic spike train data for testing and development. You can generate data with different difficulty tiers using the scripts in the `experiments/` directory.

```bash
python experiments/run_synthetic_experiments.py --generate
```

This will create multiple datasets with different levels of difficulty for both temporal and spatial pattern recognition tasks.

## Analyzing Results

After training models, you can use the scripts in the `scripts/` directory to analyze results and visualize performance.

```bash
python scripts/visualize_performance.py --history_file ./results/snn_synthetic_temporal_tier1_easy_20250330_history.json --output_dir ./results/visualizations
```

You can also analyze the impact of noise on model performance:

```bash
python scripts/analyze_noise_impact.py --results_dir ./results --output_dir ./results/noise_analysis
```

## Creating Custom Models

To create a custom SNN model, extend the `BaseSNN` or `SyntheticSNN` class in `models/base_model.py`. Implement your model architecture in a new class, ensuring it follows the same interface as the existing models.

Example:

```python
from models.base_model import SyntheticSNN

class CustomSNN(SyntheticSNN):
    def __init__(self, input_size, hidden_size, output_size, length, batch_size, tau_m=4.0, tau_s=1.0):
        super().__init__(input_size, hidden_size, output_size, length, batch_size, tau_m, tau_s)
        
        # Define your layers here
        
    def forward(self, x):
        # Implement the forward pass
        return output
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.