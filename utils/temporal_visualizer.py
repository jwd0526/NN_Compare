#!/usr/bin/env python3
"""
Temporal Visualizer - Creates sample visualization images for temporal analysis.

This script generates sample visualization images that can be used in the 
temporal analysis document when actual benchmark results aren't available.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def create_temporal_feature_importance():
    """Create a sample temporal feature importance visualization."""
    # Features that impact temporal learning
    temporal_features = [
        "Precise timing",
        "Temporal correlation",
        "Long sequences",
        "Irregular intervals",
        "Noise robustness",
        "Pattern complexity"
    ]
    
    # Create synthetic scores as example
    ann_scores = [0.35, 0.4, 0.6, 0.5, 0.45, 0.7]
    snn_scores = [0.8, 0.7, 0.5, 0.65, 0.7, 0.45]
    
    # Set output directory
    output_dir = "./results/temporal_analysis/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
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
    
    # Save the figure
    output_path = os.path.join(output_dir, "temporal_feature_importance.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Temporal feature importance visualization saved to {output_path}")
    return output_path

if __name__ == "__main__":
    print("Generating sample temporal visualizations...")
    create_temporal_feature_importance()
    print("Done.")