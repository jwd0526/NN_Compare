"""
Training metrics collection utilities.

This module provides a class to collect, save, and manage comprehensive 
training metrics throughout the training process.
"""

import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

class TrainingMetricsCollector:
    """
    Class for collecting and managing training metrics throughout the training process.
    
    This can be integrated into existing training loops to enhance the metrics
    collected during training without disrupting the main training flow.
    """
    
    def __init__(self, 
                experiment_name: str, 
                model_type: str, 
                save_dir: str = './results', 
                config: Optional[Dict] = None):
        """
        Initialize the metrics collector.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of the model being trained
            save_dir: Directory to save the metrics
            config: Optional model configuration
        """
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.save_dir = save_dir
        self.config = config
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics
        self.history = {
            'config': config,
            'model_type': model_type,
            'experiment_name': experiment_name,
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': [],
            'epoch_metrics': [],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {}
        }
        
        # Path to save the history
        self.history_path = os.path.join(save_dir, f"{experiment_name}_history.json")
    
    def set_dataset_info(self, 
                       n_samples: int, 
                       n_neurons: int, 
                       length: int, 
                       n_classes: int) -> None:
        """
        Set information about the dataset being used.
        
        Args:
            n_samples: Number of samples in the dataset
            n_neurons: Number of neurons in each sample
            length: Length of each sample (time steps)
            n_classes: Number of classes in the dataset
        """
        self.history['dataset_info'] = {
            'n_samples': n_samples,
            'n_neurons': n_neurons,
            'length': length,
            'n_classes': n_classes,
        }
    
    def record_epoch(self, 
                   epoch: int, 
                   train_acc: float, 
                   train_loss: float,
                   test_acc: float, 
                   test_loss: float, 
                   train_cm: Optional[np.ndarray] = None,
                   test_cm: Optional[np.ndarray] = None, 
                   save: bool = True) -> None:
        """
        Record metrics for a completed epoch.
        
        Args:
            epoch: Current epoch number (1-based)
            train_acc: Training accuracy for the epoch
            train_loss: Training loss for the epoch
            test_acc: Test accuracy for the epoch
            test_loss: Test loss for the epoch
            train_cm: Optional confusion matrix for training data
            test_cm: Optional confusion matrix for test data
            save: Whether to save the history after recording
        """
        # Record basic metrics
        self.history['train_acc'].append(train_acc)
        self.history['train_loss'].append(train_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_loss'].append(test_loss)
        
        # Prepare detailed epoch metrics
        train_metrics = {
            'accuracy': train_acc,
            'loss': train_loss
        }
        
        test_metrics = {
            'accuracy': test_acc,
            'loss': test_loss
        }
        
        # Add confusion matrices if available
        if train_cm is not None:
            train_metrics['confusion_matrix'] = train_cm.tolist()
            train_metrics['per_class_accuracy'] = [
                train_cm[i, i] / train_cm[i].sum() if train_cm[i].sum() > 0 else 0
                for i in range(len(train_cm))
            ]
        
        if test_cm is not None:
            test_metrics['confusion_matrix'] = test_cm.tolist()
            test_metrics['per_class_accuracy'] = [
                test_cm[i, i] / test_cm[i].sum() if test_cm[i].sum() > 0 else 0
                for i in range(len(test_cm))
            ]
        
        # Record detailed metrics
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'test': test_metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.history['epoch_metrics'].append(epoch_data)
        
        # Save history if requested
        if save:
            self.save_history()
    
    def save_history(self) -> None:
        """Save the training history to a JSON file."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_history(self) -> Dict[str, Any]:
        """
        Get the current training history.
        
        Returns:
            Dictionary containing the complete training history
        """
        return self.history
    
    def plot_progress(self, 
                    output_path: Optional[str] = None, 
                    show: bool = False, 
                    dpi: int = 150) -> None:
        """
        Plot the current training progress.
        
        Args:
            output_path: Optional path to save the plot
            show: Whether to show the plot interactively
            dpi: DPI for the saved figure
        """
        if not self.history['train_acc']:
            print("No training data recorded yet.")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], 'b-', label='Train Accuracy')
        plt.plot(self.history['test_acc'], 'r-', label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], 'b-', label='Train Loss')
        plt.plot(self.history['test_loss'], 'r-', label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Progress plot saved to {output_path}")
        
        # Show or close the plot
        if show:
            plt.show()
        else:
            plt.close()