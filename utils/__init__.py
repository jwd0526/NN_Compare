"""
Utility functions for the SNN project.

This package provides various utilities for the SNN project, including data processing,
visualization, metrics collection, and configuration management.
"""

# Import commonly used functions for easy access
from utils.data_utils import (
    load_data_from_npz,
    split_dataset,
    SpikeDataset,
    create_spiking_dataloader
)

from utils.visualization import (
    plot_raster,
    plot_filtered_spikes,
    plot_neuron_activity
)

# The metrics subpackage is imported directly as utils.metrics

__all__ = [
    'load_data_from_npz',
    'split_dataset',
    'SpikeDataset',
    'create_spiking_dataloader',
    'plot_raster',
    'plot_filtered_spikes',
    'plot_neuron_activity'
]