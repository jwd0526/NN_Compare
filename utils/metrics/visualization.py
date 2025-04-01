"""
Visualization utilities for training metrics.

This module provides functions for visualizing training metrics,
confusion matrices, and other model performance visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import os
from matplotlib.ticker import MaxNLocator

def plot_training_curves(history: Dict[str, List[float]], 
                       output_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 5),
                       dpi: int = 150,
                       show: bool = False) -> plt.Figure:
    """
    Plot training and test accuracy/loss curves.
    
    Args:
        history: Dictionary containing training history with train_acc, test_acc, train_loss, test_loss
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_confusion_matrix(cm: np.ndarray, 
                        class_names: Optional[List[str]] = None,
                        normalize: bool = False,
                        output_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        dpi: int = 150,
                        show: bool = False) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_class_accuracies(accuracies: List[float],
                        class_names: Optional[List[str]] = None,
                        output_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        dpi: int = 150,
                        show: bool = False) -> plt.Figure:
    """
    Plot per-class accuracies.
    
    Args:
        accuracies: List of accuracy values for each class
        class_names: Optional list of class names
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(accuracies))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(class_names, accuracies, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Add mean accuracy line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='r', linestyle='--', 
              label=f'Mean Accuracy: {mean_acc:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Class accuracies saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_learning_curves_by_class(history: Dict[str, List[Dict]],
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8),
                               dpi: int = 150,
                               show: bool = False) -> plt.Figure:
    """
    Plot learning curves for each class over time.
    
    Args:
        history: Dictionary containing training history with epoch_metrics
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    if 'epoch_metrics' not in history or not history['epoch_metrics']:
        raise ValueError("No epoch metrics found in history")
    
    # Extract per-class accuracies for each epoch
    epochs = []
    class_accuracies = []
    
    for epoch_data in history['epoch_metrics']:
        if 'test' in epoch_data and 'per_class_accuracy' in epoch_data['test']:
            epochs.append(epoch_data['epoch'])
            class_accuracies.append(epoch_data['test']['per_class_accuracy'])
    
    if not class_accuracies:
        raise ValueError("No per-class accuracy data found in history")
    
    # Transpose the data to get time series for each class
    n_classes = len(class_accuracies[0])
    class_time_series = [[] for _ in range(n_classes)]
    
    for epoch_acc in class_accuracies:
        for i, acc in enumerate(epoch_acc):
            class_time_series[i].append(acc)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the learning curves
    for i, accuracies in enumerate(class_time_series):
        ax.plot(epochs, accuracies, marker='o', markersize=3, 
              label=f'Class {i}')
    
    ax.set_title('Per-Class Accuracy Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    # Use integer ticks for epochs
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Per-class learning curves saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_overfitting_analysis(history: Dict[str, List[float]],
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           dpi: int = 150,
                           show: bool = False) -> plt.Figure:
    """
    Analyze potential overfitting in the model.
    
    Args:
        history: Dictionary containing training history with train_acc and test_acc
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    train_acc = history['train_acc']
    test_acc = history['test_acc']
    epochs = range(1, len(train_acc) + 1)
    
    # Calculate the gap between train and test accuracy
    acc_gap = [train - test for train, test in zip(train_acc, test_acc)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both accuracies
    ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    ax.plot(epochs, test_acc, 'r-', linewidth=2, label='Test Accuracy')
    
    # Fill the gap between curves to highlight overfitting
    ax.fill_between(epochs, train_acc, test_acc, 
                  where=[train > test for train, test in zip(train_acc, test_acc)],
                  alpha=0.3, color='orange', label='Overfitting Gap')
    
    # Find maximum gap and mark it
    if acc_gap:
        max_gap_idx = np.argmax(acc_gap)
        max_gap = acc_gap[max_gap_idx]
        max_gap_epoch = epochs[max_gap_idx]
        
        ax.annotate(f'Max Gap: {max_gap:.4f}',
                  xy=(max_gap_epoch, train_acc[max_gap_idx]),
                  xytext=(max_gap_epoch, train_acc[max_gap_idx] + 0.05),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                  fontsize=9)
    
    ax.set_title('Overfitting Analysis')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Use integer ticks for epochs
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Overfitting analysis saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig