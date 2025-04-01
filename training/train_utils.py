"""
Training utilities for SNN models.

This module contains consolidated functions for training and evaluating SNN models,
collecting performance metrics, and managing training sessions.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(model: torch.nn.Module, 
                           data_loader: DataLoader, 
                           device: torch.device) -> Tuple[np.ndarray, float, float]:
    """
    Compute confusion matrix and accuracy metrics for a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
    
    Returns:
        Tuple of (confusion_matrix, accuracy, loss)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # For SNNs, we typically sum over the time dimension
            if output.dim() > 2 and output.shape[-1] > 1:
                spike_count = torch.sum(output, dim=2)
                loss = criterion(spike_count, target.long())
            else:
                loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Get predictions
            if output.dim() > 2 and output.shape[-1] > 1:
                _, predicted = torch.max(spike_count.data, 1)
            else:
                _, predicted = torch.max(output.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate accuracy
    accuracy = np.sum(all_predictions == all_labels) / len(all_labels)
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return cm, accuracy, avg_loss

def evaluate_model(model: torch.nn.Module,
                 data_loader: DataLoader,
                 device: torch.device) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
    
    Returns:
        Tuple of (accuracy, loss)
    """
    cm, accuracy, loss = compute_confusion_matrix(model, data_loader, device)
    return accuracy, loss

def train_epoch(model: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               train_loader: DataLoader,
               device: torch.device,
               criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()) -> Tuple[float, float]:
    """
    Train the model for a single epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        train_loader: DataLoader for training data
        device: Device to run the model on
        criterion: Loss function
    
    Returns:
        Tuple of (train_loss, train_accuracy)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # For SNNs, we typically sum over the time dimension
        if output.dim() > 2 and output.shape[-1] > 1:
            spike_count = torch.sum(output, dim=2)
            loss = criterion(spike_count, target.long())
        else:
            loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        train_loss += loss.item()
        
        # Calculate accuracy
        if output.dim() > 2 and output.shape[-1] > 1:
            _, predicted = torch.max(spike_count.data, 1)
        else:
            _, predicted = torch.max(output.data, 1)
        
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    return avg_loss, accuracy

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
    
    def plot_current_progress(self, 
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

def collect_epoch_metrics(model: torch.nn.Module, 
                        train_loader: DataLoader, 
                        test_loader: DataLoader, 
                        device: torch.device, 
                        epoch: int, 
                        metrics_collector: TrainingMetricsCollector) -> Tuple[float, float, float, float]:
    """
    Collect metrics for a completed epoch and record them.
    
    Args:
        model: PyTorch model to evaluate
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to run the model on
        epoch: Current epoch number (1-based)
        metrics_collector: TrainingMetricsCollector instance
    
    Returns:
        Tuple of (train_accuracy, train_loss, test_accuracy, test_loss)
    """
    # Compute metrics for training data
    train_cm, train_acc, train_loss = compute_confusion_matrix(model, train_loader, device)
    
    # Compute metrics for test data
    test_cm, test_acc, test_loss = compute_confusion_matrix(model, test_loader, device)
    
    # Record metrics
    metrics_collector.record_epoch(
        epoch=epoch,
        train_acc=train_acc,
        train_loss=train_loss,
        test_acc=test_acc,
        test_loss=test_loss,
        train_cm=train_cm,
        test_cm=test_cm
    )
    
    return train_acc, train_loss, test_acc, test_loss

def train_model(model: torch.nn.Module,
               train_loader: DataLoader,
               test_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
               device: torch.device,
               epochs: int = 100,
               checkpoint_dir: str = "./checkpoint",
               experiment_name: str = "snn_experiment",
               metrics_collector: Optional[TrainingMetricsCollector] = None,
               save_freq: int = 10,
               verbose: bool = True,
               early_convergence_epochs: int = 5) -> Dict[str, List[float]]:
    """
    Train a model for multiple epochs with evaluation and checkpointing.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: The optimizer to use
        scheduler: Optional learning rate scheduler
        device: Device to use for training
        epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of the experiment for checkpoint naming
        metrics_collector: Optional TrainingMetricsCollector to record metrics
        save_freq: Frequency of epochs to save checkpoints
        verbose: Whether to print progress
        early_convergence_epochs: Number of consecutive epochs with perfect test accuracy 
                                  (1.0) to consider training converged
        
    Returns:
        Dictionary containing training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tracking variables if no metrics collector provided
    if metrics_collector is None:
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Variable to track perfect accuracy epochs
    perfect_accuracy_count = 0
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            spike_count = torch.sum(output, dim=2)
            
            # Calculate loss
            loss = criterion(spike_count, target.long())
            train_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(spike_count.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Print progress
            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} '
                      f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        # Calculate training metrics
        train_accuracy = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Evaluation phase
        test_cm, test_accuracy, test_loss = compute_confusion_matrix(model, test_loader, device)
        
        # Step scheduler if needed
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        if metrics_collector is not None:
            # Compute training confusion matrix for metrics collector
            train_cm, _, _ = compute_confusion_matrix(model, train_loader, device)
            metrics_collector.record_epoch(
                epoch=epoch + 1,
                train_acc=train_accuracy,
                train_loss=train_loss,
                test_acc=test_accuracy,
                test_loss=test_loss,
                train_cm=train_cm,
                test_cm=test_cm
            )
        else:
            # Store metrics locally
            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        
        # Print epoch results
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Acc: {train_accuracy:.4f}, Loss: {train_loss:.6f}, '
                  f'Test Acc: {test_accuracy:.4f}, Loss: {test_loss:.6f}, '
                  f'Time: {time.time() - epoch_start_time:.2f}s')
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{experiment_name}_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_accuracy,
                'test_acc': test_accuracy,
            }, checkpoint_path)
            
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Check if we've reached perfect test accuracy
        if test_accuracy == 1.0:
            perfect_accuracy_count += 1
            if verbose and perfect_accuracy_count > 1:
                print(f"Perfect test accuracy for {perfect_accuracy_count} consecutive epochs.")
                
            # If we've had perfect accuracy for the specified number of epochs, stop training early
            if perfect_accuracy_count >= early_convergence_epochs:
                if verbose:
                    print(f"\nEarly stopping: Perfect test accuracy maintained for {early_convergence_epochs} epochs.")
                    print(f"Training converged after {epoch+1} epochs out of {epochs}.")
                
                # Save final checkpoint before stopping
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"{experiment_name}_converged_epoch_{epoch+1}.pt"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_accuracy,
                    'test_acc': test_accuracy,
                }, checkpoint_path)
                
                if verbose:
                    print(f"Converged model saved to {checkpoint_path}")
                
                break
        else:
            # Reset counter if accuracy drops below 1.0
            perfect_accuracy_count = 0
    
    # Return training history
    if metrics_collector is not None:
        return metrics_collector.get_history()
    else:
        history = {
            'train_acc': train_acc_list,
            'test_acc': test_acc_list,
            'train_loss': train_loss_list,
            'test_loss': test_loss_list,
        }
        return history

def analyze_model_predictions(model: torch.nn.Module, 
                             data_loader: DataLoader,
                             device: torch.device,
                             class_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comprehensive analysis of model predictions.
    
    Args:
        model: The model to analyze
        data_loader: DataLoader for test data
        device: Device to run the model on
        class_names: Optional list of class names for reporting
        
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    
    # Store predictions and ground truth
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            spike_count = torch.sum(output, dim=2)
            
            # Get predictions
            _, predicted = torch.max(spike_count.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute detailed metrics
    from sklearn.metrics import (
        confusion_matrix, 
        classification_report, 
        precision_recall_fscore_support
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n--- Confusion Matrix ---")
    print(cm)
    
    # Classification Report
    print("\n--- Classification Report ---")
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_labels)))]
    
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names
    ))
    
    # Precision, Recall, F1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average=None
    )
    
    # Detailed per-class performance
    print("\n--- Detailed Per-Class Performance ---")
    for i, name in enumerate(class_names):
        print(f"Class {name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-Score:  {f1[i]:.4f}")
        print(f"  Support:   {support[i]}")
    
    # Misclassification analysis
    misclassified_indices = np.where(all_predictions != all_labels)[0]
    print(f"\nTotal Misclassified Samples: {len(misclassified_indices)}")
    
    if len(misclassified_indices) > 0:
        print("\n--- Sample Misclassification Details ---")
        for idx in misclassified_indices[:10]:  # Show first 10 misclassifications
            print(f"Sample {idx}: True Label = {all_labels[idx]}, "
                  f"Predicted Label = {all_predictions[idx]}")
    
    return all_predictions, all_labels



def load_checkpoint(checkpoint_path, device):
    """
    Load a checkpoint from a file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
    
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        # First try to load with the default weights_only=True
        return torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Warning: Error loading checkpoint with weights_only=True: {e}")
        print("Attempting to load with weights_only=False (this is less secure but compatible with older checkpoints)")
        
        # Try loading with weights_only=False
        try:
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e2:
            # Final attempt: try with safe_globals context manager
            try:
                import torch.serialization
                with torch.serialization.safe_globals(['numpy._core.multiarray.scalar']):
                    return torch.load(checkpoint_path, map_location=device)
            except Exception as e3:
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}\nErrors: {e}\n{e2}\n{e3}")