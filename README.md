# Neural Network Comparison Framework

A comprehensive framework for comparing Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs) on identical datasets, with a focus on temporal and spatial pattern recognition tasks.

## Overview

This framework enables direct performance comparisons between traditional ANNs and biologically-inspired SNNs. It includes tools for:

- Generating synthetic temporal and spatial datasets with varying difficulty levels
- Training both ANN and SNN models on identical datasets
- Comprehensive performance analysis and visualization
- Detailed model comparison metrics

The goal is to provide quantitative insights into the relative strengths and weaknesses of ANNs vs SNNs across different pattern recognition tasks.

## Project Structure

```
NN_Compare/
├── configs/                  # Configuration files
│   ├── default.yaml          # Base configuration
│   ├── experiments/          # Experiment configurations
│   │   ├── spatial_benchmark.yaml
│   │   └── temporal_benchmark.yaml
│   └── models/               # Model-specific configurations
│       ├── ann_spatial.yaml
│       ├── ann_synthetic.yaml
│       ├── snn_spatial.yaml
│       └── snn_synthetic.yaml
│
├── data/                     # Data directories (created at runtime)
│   ├── spatial/              # Spatial pattern datasets
│   └── synthetic/            # Temporal pattern datasets
│
├── experiments/              # Experiment generation scripts
│   ├── run_synthetic_experiments.py
│   ├── snn_synthetic_data_adapter.py
│   └── synthetic_data_generator.py
│
├── models/                   # Model implementations
│   ├── __init__.py
│   ├── base_model.py         # Base model classes for ANNs and SNNs
│   ├── model_registry.py     # Factory functions for model creation
│   ├── ann/                  # ANN implementations
│   │   ├── __init__.py
│   │   ├── spatial_ann.py    # ANN for spatial patterns
│   │   └── synthetic_ann.py  # ANN for synthetic patterns
│   └── snn/                  # SNN implementations
│       ├── __init__.py
│       ├── spatial_snn.py    # SNN for spatial patterns
│       └── synthetic_snn.py  # SNN for synthetic patterns
│
├── snn_lib/                  # Core SNN functionality 
│   ├── __init__.py
│   ├── data_loaders.py       # Data loading utilities 
│   ├── optimizers.py         # Optimizer implementations
│   ├── schedulers.py         # Learning rate schedulers
│   ├── snn_layers.py         # SNN layer implementations
│   └── utilities.py          # Core SNN utilities
│
├── training/                 # Training utilities
│   ├── __init__.py
│   ├── ann_trainer.py        # ANN-specific training methods
│   ├── snn_trainer.py        # SNN-specific training methods
│   ├── train_utils.py        # Common training utilities
│   └── trainer.py            # Base trainer class
│
├── utils/                    # General utilities
│   ├── __init__.py
│   ├── config_utils.py       # Configuration utilities
│   ├── data_utils.py         # Data processing utilities
│   ├── visualization.py      # Visualization utilities
│   └── metrics/              # Metrics collection and processing
│       ├── __init__.py
│       ├── collector.py      # Main metrics collector
│       ├── computation.py    # Metrics computation functions
│       └── visualization.py  # Metrics visualization
│
├── results/                  # Results directory (created at runtime)
│
├── batch_compare.py          # Batch comparison script
├── cleanup.sh                # Cleanup script
├── main.py                   # Main entry point for individual experiments
├── run_benchmarks.sh         # Main benchmark script
└── README.md                 # This documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- Pandas
- OmegaConf
- Scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NN_Compare.git
   cd NN_Compare
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv snn_env
   source snn_env/bin/activate  # On Windows: snn_env\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install torch numpy matplotlib pandas omegaconf scikit-learn
   ```

## Running Benchmarks

The main benchmark script `run_benchmarks.sh` orchestrates the entire comparison process:

```bash
# Activate the virtual environment
source snn_env/bin/activate

# Run the full benchmark (may take several hours)
./run_benchmarks.sh

# Run a quicker benchmark with fewer epochs
./run_benchmarks.sh --quick
```

This will:
1. Generate synthetic datasets for both temporal and spatial patterns
2. Train both ANN and SNN models on all datasets
3. Generate performance comparisons and visualizations
4. Create a comprehensive report in the results directory

### Viewing Results

After running the benchmarks, results are available in the `results` directory:

```
results/
├── master_report/           # Overall summary and key visualizations
├── temporal_benchmark/      # Detailed results for temporal patterns
└── spatial_benchmark/       # Detailed results for spatial patterns
```

Each benchmark directory contains subdirectories for each dataset, with both SNN and ANN results, visualizations, and comparison metrics.

## Dataset Generation

The framework can generate synthetic datasets with various difficulty levels:

```bash
# Generate all datasets
python experiments/run_synthetic_experiments.py --generate
```

### Temporal Patterns

Four difficulty tiers are generated:
- **Tier 1 (Easy)**: Distinct pattern types (regular, burst, synchronous, and oscillatory)
- **Tier 2 (Medium)**: Variations of the same pattern type with different parameters
- **Tier 3 (Hard)**: Similar patterns with subtle timing differences
- **Tier 4 (Expert)**: Nearly identical patterns with minimal statistical differences

Additionally, noise variants are created to test robustness.

### Spatial Patterns

Four difficulty tiers are generated:
- **Tier 1 (Easy)**: Clear spatial patterns (horizontal/vertical lines, expanding circles)
- **Tier 2 (Medium)**: Same patterns with mild noise
- **Tier 3 (Hard)**: Patterns with significant noise and temporal jitter
- **Tier 4 (Expert)**: Very similar patterns with subtle variations

## Individual Model Training

For training specific models on specific datasets (rather than running the full benchmark), use `main.py`:

```bash
# Train an SNN model on a temporal dataset
python main.py --data_path ./data/synthetic/temporal_tier1_easy.npz --train --config_file configs/models/snn_synthetic.yaml

# Train an ANN model on a spatial dataset
python main.py --data_path ./data/spatial/spatial_tier1_easy.npz --train --config_file configs/models/ann_spatial.yaml --experiment_name my_spatial_experiment
```

### Command-line Options for main.py

- `--data_path`: Path to the dataset file (.npz)
- `--config_file`: Path to the configuration file
- `--train`: Train a model
- `--test`: Test a model
- `--analyze`: Analyze model performance
- `--checkpoint`: Path to a model checkpoint for testing/analysis
- `--output_dir`: Directory to save results
- `--experiment_name`: Custom experiment name
- `--visualize`: Generate visualizations
- `--device`: Device to use (e.g., "cpu", "cuda:0")
- `--verbose`: Print verbose output

## Configuration System

The framework uses a hierarchical configuration system with YAML files:

- `configs/default.yaml`: Base configuration with default settings
- `configs/models/*.yaml`: Model-specific configurations
- `configs/experiments/*.yaml`: Experiment-specific configurations

Example model configuration:
```yaml
# Extend default configuration
defaults:
  - ../default

# Override settings for spatial SNN model
model:
  model_type: "spatial"
  activation: "spike"

# Hyperparameters
hyperparameters:
  batch_size: 32
  hidden_size: 64
  tau_m: 4.0  # Membrane time constant
  tau_s: 1.0  # Synaptic time constant
  dropout_rate: 0.3
  epoch: 50

# Optimizer settings
optimizer:
  optimizer_choice: "Adam"
  Adam:
    lr: 0.0005
```

## Cleaning Up Generated Files

To clean up generated files, use the `cleanup.sh` script:

```bash
# Show help
./cleanup.sh --help

# Remove all generated files (with confirmation prompts)
./cleanup.sh --all

# Remove everything for a completely clean slate
./cleanup.sh --clean-slate

# Remove specific types of files
./cleanup.sh --data          # Remove data files
./cleanup.sh --results       # Remove results
./cleanup.sh --checkpoints   # Remove model checkpoints
./cleanup.sh --temp          # Remove temporary files

# Skip confirmation prompts
./cleanup.sh --all --force
```

## Customization

### Adding New Models

To add a new model:

1. Create a new implementation file in `models/ann/` or `models/snn/`
2. Extend the appropriate base class (`SyntheticANN`, `SyntheticSNN`, etc.)
3. Register the model in `models/model_registry.py`
4. Create a configuration file in `configs/models/`

### Adding New Datasets

To add a new dataset type:

1. Extend `synthetic_data_generator.py` with new pattern generation functions
2. Modify `run_synthetic_experiments.py` to generate the new patterns
3. Update configuration files in `configs/experiments/` to use the new datasets

## Performance Metrics

The framework collects and compares various metrics:

- **Accuracy**: Classification accuracy on test datasets
- **Loss**: Training and test loss values
- **Training Time**: Time required for model training
- **Convergence Speed**: Number of epochs to reach accuracy thresholds
- **Parameter Counts**: Number of trainable parameters in each model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.