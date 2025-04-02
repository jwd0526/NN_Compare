#!/usr/bin/env python3
"""
Fix datasets with dimension mismatches between patterns and labels.
"""

import numpy as np
import os

def fix_dataset(dataset_path, output_path=None):
    """
    Fix a dataset by ensuring patterns and labels have matching first dimensions.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the fixed dataset (if None, will use dataset_path)
    """
    try:
        # Load the dataset
        print(f'Processing {dataset_path}')
        dataset = np.load(dataset_path)
        
        # Extract patterns and labels
        keys = list(dataset.keys())
        if len(keys) >= 2:
            patterns = dataset[keys[0]]
            labels = dataset[keys[1]]
            
            # Print shapes to diagnose
            print(f'Patterns shape: {patterns.shape}')
            print(f'Labels shape: {labels.shape}')
            
            # Check if there's a mismatch
            if len(patterns) != len(labels):
                print(f'Error: Shape mismatch - {len(patterns)} patterns vs {len(labels)} labels')
                
                # Keep the minimum length between patterns and labels
                min_len = min(len(patterns), len(labels))
                patterns = patterns[:min_len]
                labels = labels[:min_len]
                
                print(f'Fixed by truncating to {min_len} samples')
                
                # Determine output path
                if output_path is None:
                    output_path = dataset_path
                
                # Save fixed dataset
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                np.savez(output_path, patterns=patterns, labels=labels)
                print(f'Saved fixed dataset to {output_path}')
                return True
            else:
                print('Dataset looks good, no fix needed')
                return False
        else:
            print(f'Warning: Could not identify patterns and labels in {dataset_path}')
            return False
            
    except Exception as e:
        print(f'Error processing {dataset_path}: {str(e)}')
        return False

def main():
    # Create output directory
    os.makedirs("data/temporal/fix", exist_ok=True)
    
    # Fix all temporal datasets
    datasets = [
        'data/temporal/datasets/temporal_tier1_precise.npz',
        'data/temporal/datasets/temporal_tier2_correlation.npz',
        'data/temporal/datasets/temporal_tier3_complex.npz'
    ]
    
    fixed_datasets = []
    for dataset_path in datasets:
        fixed_path = dataset_path.replace('datasets', 'fix')
        if fix_dataset(dataset_path, fixed_path):
            fixed_datasets.append(fixed_path)
    
    # Copy fixed datasets back if needed
    if fixed_datasets:
        print("\nFixed datasets:")
        for path in fixed_datasets:
            print(f"  - {path}")
            
        print("\nTo use the fixed datasets, either:")
        print("1. Copy them back to the original location:")
        for fixed_path in fixed_datasets:
            original = fixed_path.replace('fix', 'datasets')
            print(f"   cp {fixed_path} {original}")
        
        print("\n2. Or update the run_benchmarks.sh script to use the fixed directory")
        print("   Change --datadir \"$DATA_DIR/temporal/datasets\" to --datadir \"$DATA_DIR/temporal/fix\"")

if __name__ == "__main__":
    main()