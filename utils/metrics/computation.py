"""
Functions for computing model performance metrics.

This module provides functions to compute common metrics such as accuracy,
loss, and confusion matrices for SNN model evaluation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

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
            
            # Calculate loss
            # For SNNs, we typically sum over the time dimension
            if output.dim() > 2 and output.shape[-1] > 1:  # Likely SNN output with time dimension
                spike_count = torch.sum(output, dim=2)
                loss = criterion(spike_count, target.long())
            else:
                loss = criterion(output, target.long())
            
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

def collect_epoch_metrics(model: torch.nn.Module, 
                        train_loader: DataLoader, 
                        test_loader: DataLoader, 
                        device: torch.device, 
                        epoch: int, 
                        metrics_collector) -> Tuple[float, float, float, float]:
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

def compute_per_class_metrics(model: torch.nn.Module,
                            data_loader: DataLoader,
                            device: torch.device,
                            n_classes: int) -> Dict[str, List[float]]:
    """
    Compute per-class performance metrics.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        n_classes: Number of classes
    
    Returns:
        Dictionary with per-class precision, recall, and F1 scores
    """
    cm, accuracy, loss = compute_confusion_matrix(model, data_loader, device)
    
    # Calculate per-class precision, recall, and F1 score
    precision = []
    recall = []
    f1_score = []
    
    for i in range(n_classes):
        # True positives are the diagonal elements
        tp = cm[i, i]
        
        # False positives are the sum of column i minus the diagonal element
        fp = np.sum(cm[:, i]) - tp
        
        # False negatives are the sum of row i minus the diagonal element
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate precision, recall, and F1 score
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }