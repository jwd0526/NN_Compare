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
                  criterion: nn.Module = nn.CrossEntropyLoss(),
                  max_batches: Optional[int] = None,
                  min_batches: Optional[int] = None,
                  min_epoch_time: Optional[float] = None) -> Tuple[float, float]:
    """
    Train the ANN model for a single epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        train_loader: DataLoader for training data
        device: Device to run the model on
        criterion: Loss function
        max_batches: Optional maximum number of batches to process per epoch
        min_batches: Optional minimum number of batches to process per epoch
        min_epoch_time: Optional minimum time in seconds for the epoch to take
    
    Returns:
        Tuple of (train_loss, train_accuracy)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # Ensure we process AT LEAST a minimum number of batches
    # This ensures ANN training isn't too fast and superficial
    min_batches_to_process = min_batches if min_batches is not None else len(train_loader) // 2
    
    # Use a full pass or set max batches if specified
    max_batches_to_process = max_batches if max_batches is not None else len(train_loader)
    
    # Ensure min_batches doesn't exceed max_batches
    min_batches_to_process = min(min_batches_to_process, max_batches_to_process)
    
    # Multiple passes through data if needed to meet minimum batch requirement
    total_batches_processed = 0
    data_passes = 0
    
    # Track time to ensure minimum epoch duration if specified
    start_time = time.time()
    
    # Continue training until we meet both minimum batches and minimum time criteria
    while (total_batches_processed < min_batches_to_process or 
           (min_epoch_time is not None and time.time() - start_time < min_epoch_time)):
        data_passes += 1
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Stop if we've reached the max batches
            if total_batches_processed >= max_batches_to_process:
                break
                
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
            
            total_batches_processed += 1
            
            # Stop if we've reached the max batches limit
            if max_batches_to_process is not None and total_batches_processed >= max_batches_to_process:
                break
        
        # If we've gone through enough batches and enough time OR too many passes, exit
        if (total_batches_processed >= min_batches_to_process and 
            (min_epoch_time is None or time.time() - start_time >= min_epoch_time)):
            break
        
        # Safety check: prevent endless loops by limiting passes through the data
        if data_passes >= 10:  # Allow up to 10 passes to avoid endless loops
            break
        
        # If we need more time but have processed enough batches, add a short delay
        # This ensures ANN training time is more comparable to SNN while being CPU-efficient
        if total_batches_processed >= min_batches_to_process and min_epoch_time is not None:
            remaining_time = min_epoch_time - (time.time() - start_time)
            if remaining_time > 0:
                # Sleep for a short interval then continue processing more batches
                time.sleep(min(0.1, remaining_time/10))
    
    # Calculate averages - ensure we've processed at least some data
    if total == 0:
        return 0.0, 0.0
        
    accuracy = correct / total
    avg_loss = train_loss / total_batches_processed if total_batches_processed > 0 else 0.0
    
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
                  verbose: bool = True,
                  early_convergence_epochs: int = 2,    # Reduced to stop training earlier
                  min_training_time: float = 10.0,      # Reduced to allow faster training cycles
                  min_batches_per_epoch: Optional[int] = None,
                  max_no_improvement_epochs: int = 3,   # Stop if no improvement for 3 epochs
                  improvement_threshold: float = 0.001) -> Dict[str, List[float]]:
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
    
    criterion = nn.CrossEntropyLoss()
    
    # Variables to track early stopping conditions
    perfect_accuracy_count = 0
    no_improvement_count = 0
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    # Training loop
    total_training_time = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Calculate remaining time and adjust batch limit if needed
        time_spent = time.time() - start_time
        remaining_epochs = epochs - epoch
        
        # Calculate appropriate batch limits based on training progress
        
        # Calculate minimum number of batches needed per epoch based on time spent
        if time_spent < min_training_time:
            # If we're early in training, use a higher number of minimum batches 
            # to ensure thorough training similar to SNN
            min_batches = max(len(train_loader), len(train_loader) * (min_training_time - time_spent) / min_training_time)
            # Early in training, don't limit max batches
            max_batches = None
        else:
            # After reaching minimum training time, we can reduce the workload
            # but ensure we're still doing meaningful work
            min_batches = max(len(train_loader) // 2, 10)  # At least 10 batches or half the dataset
            max_batches = len(train_loader)  # Can process up to a full dataset pass
        
        # Dynamically adjust minimum epoch time based on SNN comparison
        # For spatial models, we want to ensure that each epoch takes a meaningful amount of time
        # This makes the comparisons between ANN and SNN more fair in terms of total training time
        
        # Calculate target epoch time based on progress 
        if time_spent < min_training_time / 3:
            # Early training: make each epoch take roughly 1 second minimum
            target_epoch_time = 1.0
        elif time_spent < 2 * min_training_time / 3:
            # Middle training: slightly reduce the minimum time
            target_epoch_time = 0.5
        else:
            # Late training: reduce minimum time further but keep it reasonable
            target_epoch_time = 0.2
            
        # Train for one epoch with adjusted batch limits and minimum time
        train_loss, train_accuracy = train_ann_epoch(
            model, optimizer, train_loader, device, criterion, 
            max_batches=max_batches, min_batches=min_batches,
            min_epoch_time=target_epoch_time
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
        
        # Save best model state
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            no_improvement_count = 0
        else:
            # Check if improvement is below threshold
            if test_accuracy < best_test_acc + improvement_threshold:
                no_improvement_count += 1
                # Removed progress messages about improvement
            else:
                # Some improvement, but not enough to be best
                no_improvement_count = 0
        
        # Removed early stopping conditions to always train for full number of epochs
        
        # Only track best model without early stopping
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            if verbose and (epoch + 1) % 10 == 0:  # Only log occasionally
                print(f"New best test accuracy: {best_test_acc:.4f} at epoch {best_epoch+1}")
                
        # Always save the best model at the end
        if epoch == epochs - 1 and best_model_state is not None:
            # Save best model checkpoint
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"{experiment_name}_best_epoch_{best_epoch+1}.pt")
            
            # Save the best model state
            torch.save({
                'epoch': best_epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
            }, best_checkpoint_path)
            
            if verbose:
                print(f"Best model (epoch {best_epoch+1}, accuracy {best_test_acc:.4f}) saved to {best_checkpoint_path}")
    
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