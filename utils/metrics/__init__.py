"""
Metrics collection and visualization for SNN models.

This package provides tools for collecting, computing, and visualizing 
training metrics for SNN models.
"""

from utils.metrics.collector import TrainingMetricsCollector
from utils.metrics.computation import (
    compute_confusion_matrix,
    evaluate_model,
    collect_epoch_metrics,
    compute_per_class_metrics
)
from utils.metrics.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_class_accuracies,
    plot_learning_curves_by_class,
    plot_overfitting_analysis
)

__all__ = [
    'TrainingMetricsCollector',
    'compute_confusion_matrix',
    'evaluate_model',
    'collect_epoch_metrics',
    'compute_per_class_metrics',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_class_accuracies',
    'plot_learning_curves_by_class',
    'plot_overfitting_analysis'
]