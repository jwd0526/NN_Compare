#!/usr/bin/env python3
"""
Batch comparison script for ANN vs SNN models.

This script processes multiple datasets, training and comparing
both ANN and SNN models on each, and generates comprehensive 
visualization and comparison reports.

Usage:
    python batch_compare.py --datadir ./data --outdir ./results/comparisons
"""

import os
import sys
import argparse
import json
import time
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, List, Any

# Import necessary project components
from utils.data_utils import load_data_from_npz, split_dataset, SpikeDataset
from snn_lib.optimizers import get_optimizer
from snn_lib.schedulers import get_scheduler
from models.model_registry import create_model
from training.ann_trainer import train_ann_model
from training.train_utils import train_model as train_snn_model
from utils.metrics.collector import TrainingMetricsCollector
from utils.metrics.computation import compute_confusion_matrix

import omegaconf
from omegaconf import OmegaConf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch ANN vs SNN Comparison')
    
    # Input/output arguments
    parser.add_argument('--datadir', type=str, default='./data',
                      help='Directory containing dataset files')
    parser.add_argument('--outdir', type=str, default='./results/comparisons',
                      help='Directory to save comparison results')
    parser.add_argument('--config', type=str, default='./configs/default.yaml',
                      help='Path to the configuration file')
    
    # Model training arguments
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs to train (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (e.g., "cuda:0", "cpu")')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (overrides config)')
    
    # Execution control
    parser.add_argument('--filter', type=str, default=None,
                      help='Only process datasets matching this pattern')
    parser.add_argument('--skip-snn', action='store_true', 
                      help='Skip SNN training')
    parser.add_argument('--skip-ann', action='store_true',
                      help='Skip ANN training')
    
    return parser.parse_args()

def setup_environment(args, conf):
    """Set up the training environment."""
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using CUDA acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = conf.pytorch_seed
    
    # Apply seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    return device, seed

def get_dataset_files(data_dir, filter_pattern=None):
    """Get all .npz dataset files in the directory, optionally filtered."""
    if filter_pattern:
        pattern = os.path.join(data_dir, f"*{filter_pattern}*.npz")
        files = glob.glob(pattern)
    else:
        pattern = os.path.join(data_dir, "*.npz")
        files = glob.glob(pattern)
    
    return sorted(files)

def determine_model_type(dataset_name):
    """Determine the appropriate model type from the dataset name."""
    if 'spatial' in dataset_name.lower():
        return 'spatial'
    else:
        return 'synthetic'

def load_and_prepare_data(data_path, batch_size):
    """Load and prepare data for training."""
    print(f"Loading data from {data_path}")
    patterns, labels = load_data_from_npz(data_path)
    
    # Get dimensions
    n_samples, n_neurons, length = patterns.shape
    n_classes = len(np.unique(labels))
    
    print(f"Loaded {n_samples} samples with {n_neurons} neurons and {length} time steps")
    print(f"Number of classes: {n_classes}")
    
    # Split data into train and test sets
    train_data, train_labels, test_data, test_labels = split_dataset(
        patterns, labels, train_ratio=0.8, shuffle=True
    )
    
    # Create datasets
    train_dataset = SpikeDataset(train_data, train_labels)
    test_dataset = SpikeDataset(test_data, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    # Data info dictionary
    data_info = {
        "n_samples": n_samples,
        "n_neurons": n_neurons,
        "length": length,
        "n_classes": n_classes,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "dataset_name": os.path.basename(data_path)
    }
    
    return train_loader, test_loader, data_info



def train_snn(model_type, train_loader, test_loader, data_info, config, 
             device, epochs, experiment_name, output_dir):
    """Train and evaluate an SNN model."""
    print(f"\nTraining SNN model ({model_type}) for {epochs} epochs...")
    
    # Create model
    model = create_model(
        model_arch='snn',
        model_type=model_type,
        n_neurons=data_info["n_neurons"],
        n_classes=data_info["n_classes"],
        length=data_info["length"],
        config=config,
        device=device
    )
    
    print(f"Created SNN {model_type} model with {model.count_parameters()} parameters")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config)
    
    # Metrics collector
    metrics_collector = TrainingMetricsCollector(
        experiment_name=experiment_name,
        model_type=f"snn_{model_type}",
        save_dir=output_dir,
        config=convert_to_serializable(config)  # Convert to serializable
    )
    
    # Set dataset information
    metrics_collector.set_dataset_info(
        n_samples=data_info["n_samples"],
        n_neurons=data_info["n_neurons"],
        length=data_info["length"],
        n_classes=data_info["n_classes"]
    )
    
    # Train model
    start_time = time.time()
    
    history = train_snn_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name=experiment_name,
        metrics_collector=metrics_collector,
        verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"SNN training completed in {train_time:.2f} seconds")
    
    # Final evaluation
    test_cm, test_acc, test_loss = compute_confusion_matrix(model, test_loader, device)
    
    results = {
        "model_type": f"snn_{model_type}",
        "parameters": model.count_parameters(),
        "train_time": train_time,
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_loss),
        "best_test_acc": float(max(history["test_acc"])),
        "history": {k: convert_to_serializable(v) for k, v in history.items() if k != 'epoch_metrics'}
    }
    
    # Save results summary
    with open(os.path.join(output_dir, f"{experiment_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, model

def train_ann(model_type, train_loader, test_loader, data_info, config, 
             device, epochs, experiment_name, output_dir):
    """Train and evaluate an ANN model."""
    print(f"\nTraining ANN model ({model_type}) for {epochs} epochs...")
    
    # Create model
    model = create_model(
        model_arch='ann',
        model_type=model_type,
        n_neurons=data_info["n_neurons"],
        n_classes=data_info["n_classes"],
        length=data_info["length"],
        config=config,
        device=device
    )
    
    print(f"Created ANN {model_type} model with {model.count_parameters()} parameters")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config)
    
    # Metrics collector
    metrics_collector = TrainingMetricsCollector(
        experiment_name=experiment_name,
        model_type=f"ann_{model_type}",
        save_dir=output_dir,
        config=convert_to_serializable(config)  # Convert to serializable
    )
    
    # Set dataset information
    metrics_collector.set_dataset_info(
        n_samples=data_info["n_samples"],
        n_neurons=data_info["n_neurons"],
        length=data_info["length"],
        n_classes=data_info["n_classes"]
    )
    
    # Train model
    start_time = time.time()
    
    history = train_ann_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name=experiment_name,
        metrics_collector=metrics_collector,
        verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"ANN training completed in {train_time:.2f} seconds")
    
    # Final evaluation
    test_cm, test_acc, test_loss = compute_confusion_matrix(model, test_loader, device)
    
    results = {
        "model_type": f"ann_{model_type}",
        "parameters": model.count_parameters(),
        "train_time": train_time,
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_loss),
        "best_test_acc": float(max(history["test_acc"])),
        "history": {k: convert_to_serializable(v) for k, v in history.items() if k != 'epoch_metrics'}
    }
    
    # Save results summary
    with open(os.path.join(output_dir, f"{experiment_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, model

def generate_comparison_visualizations(snn_results, ann_results, data_info, output_dir):
    """Generate visualizations comparing SNN and ANN performance."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract histories
    snn_history = snn_results["history"]
    ann_history = ann_results["history"]
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracies
    plt.subplot(1, 2, 1)
    plt.plot(snn_history['train_acc'], 'r-', label='SNN Training')
    plt.plot(ann_history['train_acc'], 'b-', label='ANN Training')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test accuracies
    plt.subplot(1, 2, 2)
    plt.plot(snn_history['test_acc'], 'r-', label='SNN Testing')
    plt.plot(ann_history['test_acc'], 'b-', label='ANN Testing')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Loss Comparison
    plt.figure(figsize=(12, 6))
    
    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(snn_history['train_loss'], 'r-', label='SNN Training')
    plt.plot(ann_history['train_loss'], 'b-', label='ANN Training')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test losses
    plt.subplot(1, 2, 2)
    plt.plot(snn_history['test_loss'], 'r-', label='SNN Testing')
    plt.plot(ann_history['test_loss'], 'b-', label='ANN Testing')
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "loss_comparison.png"), dpi=150)
    plt.close()
    
    # 3. Convergence Speed Comparison
    plt.figure(figsize=(10, 6))
    
    # Define thresholds for convergence
    thresholds = [0.7, 0.8, 0.9, 0.95]
    ann_convergence = []
    snn_convergence = []
    
    for threshold in thresholds:
        # Find first epoch where test accuracy exceeds threshold
        ann_epoch = next((i+1 for i, acc in enumerate(ann_history['test_acc']) 
                         if acc >= threshold), len(ann_history['test_acc']))
        snn_epoch = next((i+1 for i, acc in enumerate(snn_history['test_acc']) 
                         if acc >= threshold), len(snn_history['test_acc']))
        
        ann_convergence.append(ann_epoch)
        snn_convergence.append(snn_epoch)
    
    # Plot as grouped bar chart
    x = np.arange(len(thresholds))
    width = 0.35
    
    plt.bar(x - width/2, ann_convergence, width, label='ANN', color='blue', alpha=0.7)
    plt.bar(x + width/2, snn_convergence, width, label='SNN', color='red', alpha=0.7)
    
    plt.xlabel('Accuracy Threshold')
    plt.ylabel('Epochs to Converge')
    plt.title('Convergence Speed Comparison')
    plt.xticks(x, [f'{t*100}%' for t in thresholds])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(ann_convergence):
        plt.text(i - width/2, v + 0.5, str(v), ha='center', va='bottom')
    for i, v in enumerate(snn_convergence):
        plt.text(i + width/2, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "convergence_comparison.png"), dpi=150)
    plt.close()
    
    # 4. Final performance comparison table
    final_metrics = pd.DataFrame([
        {"Model": "SNN", 
         "Final Train Acc": snn_history['train_acc'][-1], 
         "Final Test Acc": snn_history['test_acc'][-1], 
         "Max Test Acc": max(snn_history['test_acc']),
         "Training Time (s)": snn_results["train_time"]},
        {"Model": "ANN", 
         "Final Train Acc": ann_history['train_acc'][-1], 
         "Final Test Acc": ann_history['test_acc'][-1], 
         "Max Test Acc": max(ann_history['test_acc']),
         "Training Time (s)": ann_results["train_time"]}
    ])
    
    # Save metrics to CSV
    final_metrics.to_csv(os.path.join(vis_dir, "final_metrics.csv"), index=False)
    
    # Return visualization data for report generation
    return {
        "thresholds": thresholds,
        "ann_convergence": ann_convergence,
        "snn_convergence": snn_convergence,
        "final_metrics": final_metrics
    }

def generate_comparison_report(snn_results, ann_results, data_info, vis_data, output_dir):
    """Generate a detailed comparison report."""
    dataset_name = data_info["dataset_name"]
    report_file = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"ANN vs SNN COMPARISON REPORT - DATASET: {dataset_name}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write(f"Total samples: {data_info['n_samples']}\n")
        f.write(f"Input neurons: {data_info['n_neurons']}\n")
        f.write(f"Time steps: {data_info['length']}\n")
        f.write(f"Number of classes: {data_info['n_classes']}\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"SNN model type: {snn_results['model_type']}\n")
        f.write(f"SNN parameters: {snn_results['parameters']}\n")
        f.write(f"ANN model type: {ann_results['model_type']}\n")
        f.write(f"ANN parameters: {ann_results['parameters']}\n\n")
        
        f.write("FINAL METRICS:\n")
        f.write(vis_data["final_metrics"].to_string(index=False))
        f.write("\n\n")
        
        # Convergence comparison
        f.write("CONVERGENCE COMPARISON:\n")
        for i, threshold in enumerate(vis_data["thresholds"]):
            f.write(f"Epochs to {threshold*100}% accuracy: ")
            ann_epoch = vis_data["ann_convergence"][i]
            snn_epoch = vis_data["snn_convergence"][i]
            diff = ann_epoch - snn_epoch
            
            f.write(f"ANN={ann_epoch}, SNN={snn_epoch}, ")
            f.write(f"Diff={diff}\n")
        
        # Performance comparison
        f.write("\nPERFORMANCE ANALYSIS:\n")
        max_ann = max(ann_results["history"]['test_acc'])
        max_snn = max(snn_results["history"]['test_acc'])
        diff = max_ann - max_snn
        
        f.write(f"Best ANN Accuracy: {max_ann:.4f}\n")
        f.write(f"Best SNN Accuracy: {max_snn:.4f}\n")
        f.write(f"Difference: {diff:.4f}\n")
        
        if diff > 0.05:
            f.write("ANN significantly outperforms SNN\n")
        elif diff > 0.01:
            f.write("ANN slightly outperforms SNN\n")
        elif diff < -0.05:
            f.write("SNN significantly outperforms ANN\n") 
        elif diff < -0.01:
            f.write("SNN slightly outperforms ANN\n")
        else:
            f.write("ANN and SNN have comparable performance\n")
        
        # Training efficiency
        f.write("\nTRAINING EFFICIENCY:\n")
        ann_time = ann_results["train_time"]
        snn_time = snn_results["train_time"]
        time_ratio = ann_time / snn_time if snn_time > 0 else float('inf')
        
        f.write(f"ANN training time: {ann_time:.2f} seconds\n")
        f.write(f"SNN training time: {snn_time:.2f} seconds\n")
        f.write(f"Time ratio (ANN/SNN): {time_ratio:.2f}\n")
        
        if time_ratio > 2:
            f.write("SNN training is significantly faster\n")
        elif time_ratio > 1.2:
            f.write("SNN training is slightly faster\n")
        elif time_ratio < 0.5:
            f.write("ANN training is significantly faster\n")
        elif time_ratio < 0.8:
            f.write("ANN training is slightly faster\n")
        else:
            f.write("ANN and SNN have comparable training efficiency\n")
    
    print(f"Comparison report saved to {report_file}")
    
    # Create summary for combined report
    summary = {
        "dataset": dataset_name,
        "ann_best_acc": float(max_ann),
        "snn_best_acc": float(max_snn),
        "diff": float(diff),
        "ann_train_time": float(ann_time),
        "snn_train_time": float(snn_time),
        "time_ratio": float(time_ratio)
    }
    
    return summary, report_file

def create_combined_report(dataset_summaries, output_dir):
    """Create a combined report summarizing all dataset comparisons."""
    report_file = os.path.join(output_dir, "combined_comparison_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("ANN vs SNN COMPARATIVE ANALYSIS\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=============================================\n\n")
        
        f.write("SUMMARY TABLE:\n")
        f.write("------------------------------------------------------------------\n")
        f.write("%-25s %-10s %-10s %-10s %-10s %-10s\n" % 
              ("Dataset", "ANN Acc", "SNN Acc", "Diff", "Time Ratio", "Faster"))
        f.write("------------------------------------------------------------------\n")
        
        for summary in dataset_summaries:
            faster = "SNN" if summary["time_ratio"] > 1 else "ANN"
            f.write("%-25s %-10.4f %-10.4f %-+10.4f %-10.2f %-10s\n" % (
                summary["dataset"],
                summary["ann_best_acc"],
                summary["snn_best_acc"],
                summary["diff"],
                summary["time_ratio"],
                faster
            ))
        
        f.write("------------------------------------------------------------------\n\n")
        # Average performance metrics
        ann_accs = [s["ann_best_acc"] for s in dataset_summaries]
        snn_accs = [s["snn_best_acc"] for s in dataset_summaries]
        diffs = [s["diff"] for s in dataset_summaries]
        time_ratios = [s["time_ratio"] for s in dataset_summaries]
        
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write(f"Average ANN Accuracy: {np.mean(ann_accs):.4f}\n")
        f.write(f"Average SNN Accuracy: {np.mean(snn_accs):.4f}\n")
        f.write(f"Average Accuracy Difference: {np.mean(diffs):.4f}\n")
        f.write(f"Average Time Ratio (ANN/SNN): {np.mean(time_ratios):.2f}\n\n")
        
        # Count which model performed better
        ann_better = sum(1 for d in diffs if d > 0.01)
        snn_better = sum(1 for d in diffs if d < -0.01)
        comparable = sum(1 for d in diffs if abs(d) <= 0.01)
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(f"ANN better: {ann_better}/{len(diffs)} datasets\n")
        f.write(f"SNN better: {snn_better}/{len(diffs)} datasets\n")
        f.write(f"Comparable: {comparable}/{len(diffs)} datasets\n\n")
        
        # Count which model was faster
        snn_faster = sum(1 for r in time_ratios if r > 1.2)
        ann_faster = sum(1 for r in time_ratios if r < 0.8)
        similar_speed = sum(1 for r in time_ratios if 0.8 <= r <= 1.2)
        
        f.write("TRAINING EFFICIENCY COMPARISON:\n")
        f.write(f"SNN faster: {snn_faster}/{len(time_ratios)} datasets\n")
        f.write(f"ANN faster: {ann_faster}/{len(time_ratios)} datasets\n")
        f.write(f"Similar speed: {similar_speed}/{len(time_ratios)} datasets\n\n")
        
        # Generate detailed analysis
        f.write("DATASET-SPECIFIC ANALYSIS:\n")
        for summary in dataset_summaries:
            f.write(f"\n{summary['dataset']}:\n")
            f.write(f"  Accuracy: ANN={summary['ann_best_acc']:.4f}, SNN={summary['snn_best_acc']:.4f}, Diff={summary['diff']:.4f}\n")
            f.write(f"  Training Time: ANN={summary['ann_train_time']:.2f}s, SNN={summary['snn_train_time']:.2f}s, Ratio={summary['time_ratio']:.2f}\n")
            
            # Performance assessment
            if summary['diff'] > 0.05:
                f.write("  Performance: ANN significantly outperforms SNN\n")
            elif summary['diff'] > 0.01:
                f.write("  Performance: ANN slightly outperforms SNN\n")
            elif summary['diff'] < -0.05:
                f.write("  Performance: SNN significantly outperforms ANN\n") 
            elif summary['diff'] < -0.01:
                f.write("  Performance: SNN slightly outperforms ANN\n")
            else:
                f.write("  Performance: ANN and SNN have comparable performance\n")
            
            # Efficiency assessment
            if summary['time_ratio'] > 2:
                f.write("  Training Efficiency: SNN training is significantly faster\n")
            elif summary['time_ratio'] > 1.2:
                f.write("  Training Efficiency: SNN training is slightly faster\n")
            elif summary['time_ratio'] < 0.5:
                f.write("  Training Efficiency: ANN training is significantly faster\n")
            elif summary['time_ratio'] < 0.8:
                f.write("  Training Efficiency: ANN training is slightly faster\n")
            else:
                f.write("  Training Efficiency: ANN and SNN have comparable training efficiency\n")
    
    print(f"Combined report saved to {report_file}")
    
    # Create a visualization of overall comparison
    create_comparative_visualizations(dataset_summaries, output_dir)

def create_comparative_visualizations(dataset_summaries, output_dir):
    """Create visualizations comparing performance across datasets."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract data
    datasets = [s["dataset"].replace(".npz", "") for s in dataset_summaries]
    ann_accs = [s["ann_best_acc"] for s in dataset_summaries]
    snn_accs = [s["snn_best_acc"] for s in dataset_summaries]
    diffs = [s["diff"] for s in dataset_summaries]
    time_ratios = [s["time_ratio"] for s in dataset_summaries]
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, ann_accs, width, label='ANN', color='blue', alpha=0.7)
    plt.bar(x + width/2, snn_accs, width, label='SNN', color='red', alpha=0.7)
    
    plt.xlabel('Dataset')
    plt.ylabel('Best Test Accuracy')
    plt.title('ANN vs SNN Accuracy Comparison Across Datasets')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(vis_dir, "overall_accuracy_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Accuracy difference chart
    plt.figure(figsize=(10, 6))
    
    # Color bars based on which model performed better
    colors = ['green' if d < 0 else 'blue' for d in diffs]
    
    plt.bar(datasets, diffs, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Dataset')
    plt.ylabel('ANN - SNN Accuracy')
    plt.title('Accuracy Difference (ANN - SNN)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(vis_dir, "accuracy_difference.png"), dpi=150)
    plt.close()
    
    # 3. Training time ratio
    plt.figure(figsize=(10, 6))
    
    # Color bars based on which model was faster
    colors = ['green' if r > 1 else 'blue' for r in time_ratios]
    
    plt.bar(datasets, time_ratios, color=colors, alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Dataset')
    plt.ylabel('Time Ratio (ANN/SNN)')
    plt.title('Training Time Ratio (ANN/SNN)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    for i, v in enumerate(time_ratios):
        plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.savefig(os.path.join(vis_dir, "training_time_ratio.png"), dpi=150)
    plt.close()

def process_dataset(dataset_path, args, conf, device):
    """Process a single dataset with both ANN and SNN models."""
    # Extract dataset name
    dataset_name = os.path.basename(dataset_path).replace('.npz', '')
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory for this dataset
    dataset_dir = os.path.join(args.outdir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "snn"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "ann"), exist_ok=True)
    
    # Load and prepare data
    batch_size = conf.hyperparameters.batch_size
    train_loader, test_loader, data_info = load_and_prepare_data(dataset_path, batch_size)
    
    # Use provided epochs or from config
    epochs = args.epochs if args.epochs is not None else conf.hyperparameters.epoch
    
    # Determine model type based on dataset name
    model_type = determine_model_type(dataset_name)
    
    # Train SNN model if not skipped
    snn_results = None
    if not args.skip_snn:
        snn_results, snn_model = train_snn(
            model_type=model_type,
            train_loader=train_loader,
            test_loader=test_loader,
            data_info=data_info,
            config=conf,
            device=device,
            epochs=epochs,
            experiment_name=f"{dataset_name}_snn",
            output_dir=os.path.join(dataset_dir, "snn")
        )
    else:
        # Try to load SNN results from file
        snn_results_file = os.path.join(dataset_dir, "snn", f"{dataset_name}_snn_results.json")
        if os.path.exists(snn_results_file):
            with open(snn_results_file, 'r') as f:
                snn_results = json.load(f)
                print(f"Loaded SNN results from {snn_results_file}")
        else:
            print("Warning: SNN training skipped and no previous results found")
            return None
    
    # Train ANN model if not skipped
    ann_results = None
    if not args.skip_ann:
        ann_results, ann_model = train_ann(
            model_type=model_type,
            train_loader=train_loader,
            test_loader=test_loader,
            data_info=data_info,
            config=conf,
            device=device,
            epochs=epochs,
            experiment_name=f"{dataset_name}_ann",
            output_dir=os.path.join(dataset_dir, "ann")
        )
    else:
        # Try to load ANN results from file
        ann_results_file = os.path.join(dataset_dir, "ann", f"{dataset_name}_ann_results.json")
        if os.path.exists(ann_results_file):
            with open(ann_results_file, 'r') as f:
                ann_results = json.load(f)
                print(f"Loaded ANN results from {ann_results_file}")
        else:
            print("Warning: ANN training skipped and no previous results found")
            return None
    
    # Generate visualizations and report if both models were trained
    if snn_results and ann_results:
        print("\nGenerating comparison visualizations and report...")
        vis_data = generate_comparison_visualizations(
            snn_results=snn_results,
            ann_results=ann_results,
            data_info=data_info,
            output_dir=dataset_dir
        )
        
        summary, _ = generate_comparison_report(
            snn_results=snn_results,
            ann_results=ann_results,
            data_info=data_info,
            vis_data=vis_data,
            output_dir=dataset_dir
        )
        
        print(f"Successfully processed dataset {dataset_name}")
        print(f"Results saved to {dataset_dir}")
        
        return summary
    else:
        print(f"Skipping comparison for {dataset_name} due to missing results")
        return None

def main():
    """Main function to run the batch comparison."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    conf = OmegaConf.load(args.config)
    
    # Set up environment
    device, _ = setup_environment(args, conf)
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Get dataset files
    dataset_files = get_dataset_files(args.datadir, args.filter)
    
    if not dataset_files:
        print(f"No dataset files found in {args.datadir}")
        return
    
    print(f"Found {len(dataset_files)} dataset files")
    for i, f in enumerate(dataset_files):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    # Process each dataset
    summaries = []
    successful = []
    failed = []
    
    for dataset_path in dataset_files:
        try:
            summary = process_dataset(dataset_path, args, conf, device)
            if summary:
                summaries.append(summary)
                successful.append(os.path.basename(dataset_path))
            else:
                failed.append(os.path.basename(dataset_path))
        except Exception as e:
            print(f"Error processing {os.path.basename(dataset_path)}: {e}")
            failed.append(os.path.basename(dataset_path))
    
    # Create combined report if multiple datasets were processed
    if len(summaries) > 1:
        print("\nCreating combined report...")
        create_combined_report(summaries, args.outdir)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Comparison Summary:")
    print(f"  Total datasets: {len(dataset_files)}")
    print(f"  Successfully processed: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print("\nFailed datasets:")
        for f in failed:
            print(f"  - {f}")
    
    print("\nComparison process completed!")

if __name__ == "__main__":
    main()