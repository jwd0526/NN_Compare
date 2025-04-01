"""
Training utilities specific to ANN models.

This module contains functions for training and evaluating ANN models,
with a focus on compatibility with the SNN training pipeline for fair comparisons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any, Union

from utils.metrics.collector import TrainingMetricsCollector
from utils.metrics.computation import compute_confusion_matrix

def evaluate_ann(model: nn.Module, 
               data_loader: DataLoader, 
               device: torch.device,
               criterion: nn.Module = nn.CrossEntropyLoss()) -> Tuple[float, float]:
    """
    Evaluate an ANN model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        criterion: Loss function to use
        
    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # For compatibility with SNN comparison, sum across time dimension
            if output.dim() > 2 and output.shape[-1] > 1:
                output = torch.sum(output, dim=2)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return accuracy, avg_loss

def train_ann_epoch(model: nn.Module, 
                  optimizer: torch.optim.Optimizer,
                  train_loader: DataLoader,
                  device: torch.device,
                  criterion: nn.Module = nn.CrossEntropyLoss()) -> Tuple[float, float]:
    """
    Train the ANN model for a single epoch.
    
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
        
        # For compatibility with SNN comparison, sum across time dimension
        if output.dim() > 2 and output.shape[-1] > 1:
            output = torch.sum(output, dim=2)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # Calculate averages
    accuracy = correct / total if total > 0 else 0
    avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    return avg_loss, accuracy

def train_ann_model(model: nn.Module,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                  device: torch.device,
                  epochs: int = 100,
                  checkpoint_dir: str = "./checkpoint",
                  experiment_name: str = "ann_experiment",
                  metrics_collector: Optional[TrainingMetricsCollector] = None,
                  save_freq: int = 10,
                  verbose: bool = True) -> Dict[str, List[float]]:
    """
    Train an ANN model for multiple epochs with evaluation and checkpointing.
    
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
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_accuracy = train_ann_epoch(
            model, optimizer, train_loader, device, criterion
        )
        
        # Evaluate on test set
        test_accuracy, test_loss = evaluate_ann(model, test_loader, device, criterion)
        
        # Step scheduler if needed
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        if metrics_collector is not None:
            # For compatibility with SNN training, compute confusion matrices
            train_cm, _, _ = compute_confusion_matrix(model, train_loader, device)
            test_cm, _, _ = compute_confusion_matrix(model, test_loader, device)
            
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

def compare_ann_snn(ann_history: Dict[str, List[float]],
                  snn_history: Dict[str, List[float]],
                  metrics: List[str] = ['test_acc', 'train_time'],
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Compare the performance of ANN and SNN models.
    
    Args:
        ann_history: Training history for the ANN model
        snn_history: Training history for the SNN model
        metrics: List of metrics to compare
        verbose: Whether to print the comparison
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for metric in metrics:
        if metric in ann_history and metric in snn_history:
            if metric.endswith('acc'):
                # For accuracy metrics, compare the maximum values
                ann_value = max(ann_history[metric])
                snn_value = max(snn_history[metric])
                comparison[f'max_{metric}'] = {
                    'ann': ann_value,
                    'snn': snn_value,
                    'diff': ann_value - snn_value,
                    'relative_diff_pct': (ann_value - snn_value) / snn_value * 100 if snn_value != 0 else float('inf')
                }
                
                # Also compare final values
                ann_final = ann_history[metric][-1]
                snn_final = snn_history[metric][-1]
                comparison[f'final_{metric}'] = {
                    'ann': ann_final,
                    'snn': snn_final,
                    'diff': ann_final - snn_final,
                    'relative_diff_pct': (ann_final - snn_final) / snn_final * 100 if snn_final != 0 else float('inf')
                }
                
                # Compare convergence speed
                threshold = 0.9  # 90% accuracy threshold
                ann_epochs = next((i+1 for i, acc in enumerate(ann_history[metric]) if acc >= threshold), float('inf'))
                snn_epochs = next((i+1 for i, acc in enumerate(snn_history[metric]) if acc >= threshold), float('inf'))
                
                comparison[f'epochs_to_{int(threshold*100)}pct_{metric}'] = {
                    'ann': ann_epochs,
                    'snn': snn_epochs,
                    'diff': ann_epochs - snn_epochs,
                    'relative_diff_pct': (ann_epochs - snn_epochs) / snn_epochs * 100 if snn_epochs != float('inf') else float('inf')
                }
            elif metric == 'train_time':
                # Compare total training time
                ann_time = sum(ann_history.get('epoch_times', [0]))
                snn_time = sum(snn_history.get('epoch_times', [0]))
                comparison['train_time'] = {
                    'ann': ann_time,
                    'snn': snn_time,
                    'diff': ann_time - snn_time,
                    'relative_diff_pct': (ann_time - snn_time) / snn_time * 100 if snn_time != 0 else float('inf')
                }
    
    # Print the comparison if requested
    if verbose:
        print("\n=== ANN vs SNN Performance Comparison ===")
        for metric_name, values in comparison.items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  ANN: {values['ann']}")
            print(f"  SNN: {values['snn']}")
            print(f"  Difference: {values['diff']:+.4f}")
            
            if values['relative_diff_pct'] != float('inf'):
                print(f"  Relative Difference: {values['relative_diff_pct']:+.2f}%")
            else:
                print("  Relative Difference: N/A (division by zero)")
    
    return comparison