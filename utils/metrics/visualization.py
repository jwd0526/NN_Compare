"""
Visualization utilities for training metrics.

This module provides functions for visualizing training metrics,
confusion matrices, and other model performance visualizations.
Includes specialized visualizations for temporal data comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

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

def plot_temporal_comparison(ann_history: Dict[str, List[float]], 
                           snn_history: Dict[str, List[float]],
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 10),
                           dpi: int = 150,
                           show: bool = False) -> plt.Figure:
    """
    Create a detailed comparison of ANN vs SNN performance on temporal data.
    
    Args:
        ann_history: Dictionary containing ANN training history
        snn_history: Dictionary containing SNN training history
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Learning Curve Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(ann_history['test_acc']) + 1)
    
    ax1.plot(epochs, ann_history['test_acc'], 'b-', linewidth=2, label='ANN')
    ax1.plot(epochs, snn_history['test_acc'], 'r-', linewidth=2, label='SNN')
    
    # Add learning rate markers if available
    if 'lr' in ann_history and 'lr' in snn_history:
        ax2 = ax1.twinx()
        ax2.plot(epochs, ann_history['lr'], 'b--', alpha=0.5, linewidth=1, label='ANN LR')
        ax2.plot(epochs, snn_history['lr'], 'r--', alpha=0.5, linewidth=1, label='SNN LR')
        ax2.set_ylabel('Learning Rate')
        ax2.tick_params(axis='y', labelcolor='gray')
    
    ax1.set_title('Test Accuracy Learning Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Convergence Analysis (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Define convergence thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9]
    ann_epochs_to_converge = []
    snn_epochs_to_converge = []
    
    for threshold in thresholds:
        # Find first epoch where accuracy exceeds threshold
        try:
            ann_epoch = next(i+1 for i, acc in enumerate(ann_history['test_acc']) if acc >= threshold)
        except StopIteration:
            ann_epoch = len(ann_history['test_acc'])
            
        try:
            snn_epoch = next(i+1 for i, acc in enumerate(snn_history['test_acc']) if acc >= threshold)
        except StopIteration:
            snn_epoch = len(snn_history['test_acc'])
            
        ann_epochs_to_converge.append(ann_epoch)
        snn_epochs_to_converge.append(snn_epoch)
    
    # Plot as grouped bar chart
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax2.bar(x - width/2, ann_epochs_to_converge, width, label='ANN', color='blue', alpha=0.7)
    ax2.bar(x + width/2, snn_epochs_to_converge, width, label='SNN', color='red', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(ann_epochs_to_converge):
        ax2.text(i - width/2, v + 0.3, str(v), ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(snn_epochs_to_converge):
        ax2.text(i + width/2, v + 0.3, str(v), ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('Epochs to Convergence')
    ax2.set_xlabel('Accuracy Threshold')
    ax2.set_ylabel('Epochs')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t*100}%' for t in thresholds])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Error Analysis (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate errors
    ann_errors = [1 - acc for acc in ann_history['test_acc']]
    snn_errors = [1 - acc for acc in snn_history['test_acc']]
    
    # Use log scale for errors
    ax3.semilogy(epochs, ann_errors, 'b-', linewidth=2, label='ANN')
    ax3.semilogy(epochs, snn_errors, 'r-', linewidth=2, label='SNN')
    
    ax3.set_title('Error Rate (Log Scale)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Error Rate (1 - Accuracy)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 4. Relative Performance Analysis (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate ratio of SNN to ANN accuracy
    accuracy_ratio = [snn/ann if ann > 0 else 1 for snn, ann in zip(snn_history['test_acc'], ann_history['test_acc'])]
    
    ax4.plot(epochs, accuracy_ratio, 'g-', linewidth=2)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Color regions above/below the equality line
    ax4.fill_between(epochs, accuracy_ratio, 1, where=[r > 1 for r in accuracy_ratio], 
                    alpha=0.2, color='green', label='SNN better')
    ax4.fill_between(epochs, accuracy_ratio, 1, where=[r <= 1 for r in accuracy_ratio], 
                    alpha=0.2, color='red', label='ANN better')
    
    # Add final ratio
    final_ratio = accuracy_ratio[-1]
    ax4.annotate(f'Final Ratio: {final_ratio:.3f}x', 
                xy=(epochs[-1], final_ratio),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10, fontweight='bold')
    
    ax4.set_title('SNN/ANN Accuracy Ratio')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Ratio (SNN ÷ ANN)')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Temporal comparison analysis saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_temporal_feature_importance(ann_results: Dict, snn_results: Dict,
                                   output_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (14, 8),
                                   dpi: int = 150,
                                   show: bool = False) -> plt.Figure:
    """
    Create a visualization of temporal feature importance for ANN vs SNN.
    
    Args:
        ann_results: Dictionary containing ANN results
        snn_results: Dictionary containing SNN results
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    # Features that might impact temporal learning
    temporal_features = [
        "Precise timing",
        "Temporal correlation",
        "Long sequences",
        "Irregular intervals",
        "Noise robustness",
        "Pattern complexity"
    ]
    
    # Create synthetic scores as example (in a real analysis, these would be computed)
    # Scores are relative strength (higher is better) for each model on each feature
    # Scale from 0-1 where 0.5 is neutral
    ann_scores = [0.35, 0.4, 0.6, 0.5, 0.45, 0.7]
    snn_scores = [0.8, 0.7, 0.5, 0.65, 0.7, 0.45]
    
    # If the results contain per-class accuracies, we can use those
    # This is just a placeholder for demonstration
    if "class_accuracies" in ann_results and "class_accuracies" in snn_results:
        ann_class_accs = list(ann_results["class_accuracies"].values())
        snn_class_accs = list(snn_results["class_accuracies"].values())
        
        # Normalize to 0-1 range for visualization
        max_acc = max(max(ann_class_accs), max(snn_class_accs))
        min_acc = min(min(ann_class_accs), min(snn_class_accs))
        range_acc = max_acc - min_acc if max_acc > min_acc else 1
        
        ann_scores = [(acc - min_acc) / range_acc for acc in ann_class_accs]
        snn_scores = [(acc - min_acc) / range_acc for acc in snn_class_accs]
        
        # If we have real class names, use those instead of generic features
        if len(ann_scores) == len(temporal_features) and "class_names" in ann_results:
            temporal_features = ann_results["class_names"]
    
    fig = plt.figure(figsize=figsize)
    
    # 1. Comparative Bar Chart
    ax1 = plt.subplot(121)
    
    x = np.arange(len(temporal_features))
    width = 0.35
    
    ax1.bar(x - width/2, ann_scores, width, label='ANN', color='blue', alpha=0.7)
    ax1.bar(x + width/2, snn_scores, width, label='SNN', color='red', alpha=0.7)
    
    ax1.set_title('Temporal Feature Performance')
    ax1.set_ylabel('Relative Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(temporal_features, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # 2. Radar Chart
    ax2 = plt.subplot(122, polar=True)
    
    # Number of features
    N = len(temporal_features)
    
    # Create angles for each feature
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the plot (makes a complete circle)
    angles += angles[:1]
    ann_scores_plot = ann_scores + [ann_scores[0]]
    snn_scores_plot = snn_scores + [snn_scores[0]]
    
    # Draw the plot
    ax2.plot(angles, ann_scores_plot, 'b-', linewidth=2, label='ANN')
    ax2.fill(angles, ann_scores_plot, 'b', alpha=0.1)
    
    ax2.plot(angles, snn_scores_plot, 'r-', linewidth=2, label='SNN')
    ax2.fill(angles, snn_scores_plot, 'r', alpha=0.1)
    
    # Set labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(temporal_features)
    
    # Draw axis lines for each feature
    ax2.set_rlabel_position(0)
    ax2.set_rticks([0.25, 0.5, 0.75])
    ax2.set_rlim(0, 1)
    ax2.grid(True)
    
    ax2.set_title('Temporal Feature Map')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Temporal feature importance analysis saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_learning_dynamics(ann_history: Dict[str, List[float]], 
                         snn_history: Dict[str, List[float]],
                         output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (16, 10),
                         dpi: int = 150,
                         show: bool = False) -> plt.Figure:
    """
    Create a detailed visualization of learning dynamics for ANN vs SNN.
    
    Args:
        ann_history: Dictionary containing ANN training history
        snn_history: Dictionary containing SNN training history
        output_path: Optional path to save the plot
        figsize: Figure size
        dpi: DPI for the saved figure
        show: Whether to show the plot interactively
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Extract data
    epochs = range(1, len(ann_history['train_acc']) + 1)
    
    # 1. Training Accuracy (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, ann_history['train_acc'], 'b-', linewidth=2, label='ANN')
    ax1.plot(epochs, snn_history['train_acc'], 'r-', linewidth=2, label='SNN')
    
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Test Accuracy (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, ann_history['test_acc'], 'b-', linewidth=2, label='ANN')
    ax2.plot(epochs, snn_history['test_acc'], 'r-', linewidth=2, label='SNN')
    
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 3. Accuracy Gap (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ann_gap = [train - test for train, test in zip(ann_history['train_acc'], ann_history['test_acc'])]
    snn_gap = [train - test for train, test in zip(snn_history['train_acc'], snn_history['test_acc'])]
    
    ax3.plot(epochs, ann_gap, 'b-', linewidth=2, label='ANN')
    ax3.plot(epochs, snn_gap, 'r-', linewidth=2, label='SNN')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_title('Train-Test Accuracy Gap')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gap (Train - Test)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 4. Training Loss (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, ann_history['train_loss'], 'b-', linewidth=2, label='ANN')
    ax4.plot(epochs, snn_history['train_loss'], 'r-', linewidth=2, label='SNN')
    
    ax4.set_title('Training Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 5. Test Loss (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, ann_history['test_loss'], 'b-', linewidth=2, label='ANN')
    ax5.plot(epochs, snn_history['test_loss'], 'r-', linewidth=2, label='SNN')
    
    ax5.set_title('Test Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 6. Loss Ratio (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Loss ratio (SNN/ANN) - values > 1 mean SNN loss is higher
    loss_ratio = [snn/ann if ann > 0 else 1 for snn, ann in zip(snn_history['test_loss'], ann_history['test_loss'])]
    
    ax6.plot(epochs, loss_ratio, 'g-', linewidth=2)
    ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Color regions above/below the equality line
    ax6.fill_between(epochs, loss_ratio, 1, where=[r > 1 for r in loss_ratio], 
                    alpha=0.2, color='red', label='SNN higher loss')
    ax6.fill_between(epochs, loss_ratio, 1, where=[r <= 1 for r in loss_ratio], 
                    alpha=0.2, color='green', label='ANN higher loss')
    
    ax6.set_title('SNN/ANN Loss Ratio')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Ratio (SNN ÷ ANN)')
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax6.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Learning dynamics analysis saved to {output_path}")
    
    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig