"""
Main entry point for SNN experiments.

This script provides a command-line interface for training and evaluating
SNN models on synthetic datasets.
"""

import argparse
import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader

# Import from project modules
from models.snn.synthetic_snn import SyntheticSpikeModel, SpatialSpikeModel
from utils.data_utils import load_data_from_npz, SpikeDataset, split_dataset
from snn_lib.optimizers import get_optimizer
from snn_lib.schedulers import get_scheduler
from utils.metrics.collector import TrainingMetricsCollector
from utils.metrics.computation import compute_confusion_matrix
from utils.metrics.visualization import plot_training_curves, plot_confusion_matrix
from training.train_utils import train_model, analyze_model_predictions, load_checkpoint

import omegaconf
from omegaconf import OmegaConf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SNN Experiments')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data file (.npz)')
    parser.add_argument('--config_file', type=str, default='configs/default.yaml',
                       help='Path to the configuration file')
    
    # Experiment type
    parser.add_argument('--train', action='store_true',
                       help='Train a model')
    parser.add_argument('--test', action='store_true',
                       help='Test a model')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze model performance')
    
    # Model loading/saving
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for testing/analyzing')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (default: auto-generated)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations of results')
    
    # Misc
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (e.g., "cuda:0", "cpu")')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    return parser.parse_args()

def main():
    """Main entry point for SNN experiments."""
    args = parse_args()
    
    # Load configuration
    conf = OmegaConf.load(args.config_file)
    
    # Override config with command line arguments
    if args.seed is not None:
        conf.pytorch_seed = args.seed
    
    # Set device - always use CPU for these small models
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        print("Using CPU - most efficient for these small models")
    
    # Set random seeds for reproducibility
    torch.manual_seed(conf.pytorch_seed)
    np.random.seed(conf.pytorch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(conf.pytorch_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    patterns, labels = load_data_from_npz(args.data_path)
    
    # Get dimensions
    n_samples, n_neurons, length = patterns.shape
    n_classes = len(np.unique(labels))
    
    print(f"Loaded {n_samples} samples with {n_neurons} neurons and {length} time steps")
    print(f"Number of classes: {n_classes}")
    
    # Split data into train and test sets
    train_data, train_labels, test_data, test_labels = split_dataset(
        patterns, labels, train_ratio=0.8, shuffle=True
    )
    
    # Create datasets
    train_dataset = SpikeDataset(train_data, train_labels)
    test_dataset = SpikeDataset(test_data, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=conf.hyperparameters.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=conf.hyperparameters.batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    # Create model
    model_type = conf.model.model_type
    if model_type == "synthetic":
        model = SyntheticSpikeModel(
            input_size=n_neurons,
            hidden_size=conf.hyperparameters.hidden_size,
            output_size=n_classes,
            length=length,
            batch_size=conf.hyperparameters.batch_size,
            tau_m=conf.hyperparameters.tau_m,
            tau_s=conf.hyperparameters.tau_s,
            dropout_rate=conf.hyperparameters.get('dropout_rate', 0.3)
        ).to(device)
    elif model_type == "spatial":
        # Determine if we have a spatial pattern (try to infer dimensions)
        height = int(np.sqrt(n_neurons))
        if height * height == n_neurons:  # Perfect square
            width = height
        else:
            # Default to a reasonable aspect ratio
            height = int(np.sqrt(n_neurons * 0.75))
            width = n_neurons // height
        
        model = SpatialSpikeModel(
            input_size=n_neurons,
            hidden_size=conf.hyperparameters.hidden_size,
            output_size=n_classes,
            spatial_shape=(height, width),
            length=length,
            batch_size=conf.hyperparameters.batch_size,
            tau_m=conf.hyperparameters.tau_m,
            tau_s=conf.hyperparameters.tau_s
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Created {model_type} model with {model.count_parameters()} trainable parameters")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), conf)
    scheduler = get_scheduler(optimizer, conf)
    
    # Training
    if args.train:
        # Create a unique experiment name if not provided
        if args.experiment_name:
            experiment_name = args.experiment_name
        else:
            # Extract dataset name from path
            dataset_name = os.path.basename(args.data_path).split('.')[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"snn_{model_type}_{dataset_name}_{timestamp}"
        
        # Create metrics collector
        metrics_collector = TrainingMetricsCollector(
            experiment_name=experiment_name,
            model_type=model_type,
            save_dir=args.output_dir,
            config=OmegaConf.to_container(conf)
        )
        
        # Set dataset information
        metrics_collector.set_dataset_info(
            n_samples=n_samples,
            n_neurons=n_neurons,
            length=length,
            n_classes=n_classes
        )
        
        print("\nStarting training...")
        # Train the model using the consolidated train_model function
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=conf.hyperparameters.epoch,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
            experiment_name=experiment_name,
            metrics_collector=metrics_collector,
            save_freq=conf.get('save_checkpoint_freq', 10),
            verbose=args.verbose
        )
        
        # Generate final training curves plot if visualization is enabled
        if args.visualize:
            vis_dir = os.path.join(args.output_dir, 'visualizations', experiment_name)
            os.makedirs(vis_dir, exist_ok=True)
            
            plot_training_curves(
                history=metrics_collector.get_history(),
                output_path=os.path.join(vis_dir, f"training_curves.png"),
                dpi=150,
                show=False
            )
            
            print(f"Visualizations saved to {vis_dir}")
        
        print(f"\nTraining completed. Full history saved to {metrics_collector.history_path}")
    
    # Testing or analyzing an existing model
    if args.test or args.analyze:
        if args.checkpoint:
            # Load model from checkpoint
            load_checkpoint(model, args.checkpoint, optimizer)
        
        # Test the model
        print("\nEvaluating model on test data...")
        test_acc, test_loss = compute_confusion_matrix(model, test_loader, device)[1:3]
        print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        
        # Perform detailed analysis if requested
        if args.analyze:
            # Create visualization directory if needed
            vis_dir = None
            if args.visualize:
                if args.experiment_name:
                    vis_dir = os.path.join(args.output_dir, 'visualizations', args.experiment_name)
                else:
                    vis_dir = os.path.join(args.output_dir, 'visualizations', 'analysis')
                os.makedirs(vis_dir, exist_ok=True)
            
            # Analyze model predictions
            print("\nPerforming detailed model analysis...")
            analyze_model_predictions(
                model=model,
                data_loader=test_loader,
                device=device,
                output_dir=vis_dir
            )
            
            if vis_dir:
                print(f"Analysis visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()