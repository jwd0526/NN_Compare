#!/bin/bash

# Neural Network Comparison Benchmark Runner
# This script runs a comprehensive comparison between ANN and SNN models
# on both spatial and synthetic datasets, generating visualizations and reports.
#
# Usage: ./run_benchmarks.sh [--quick]
#   --quick     Run with reduced epochs for quicker testing

set -e  # Exit on error

# Configuration
OUTPUT_DIR="./results"
DATA_DIR="./data"
CONFIG_DIR="./configs"

# Parse command line arguments
QUICK_MODE=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./run_benchmarks.sh [--quick]"
            exit 1
            ;;
    esac
done

# Set epochs based on quick mode
if [ "$QUICK_MODE" = true ]; then
    EPOCHS="--epochs 3"
    echo "Running in quick mode with 3 epochs"
else
    EPOCHS=""
fi

# Create required directories
mkdir -p "$DATA_DIR/synthetic"
mkdir -p "$DATA_DIR/spatial"
mkdir -p "$OUTPUT_DIR"

echo "===== Neural Network Comparison Framework ====="
echo "Starting comprehensive benchmark tests"

# Step 1: Generate synthetic datasets if they don't exist
echo "
Step 1: Generating datasets..."
if [ ! -f "$DATA_DIR/synthetic/temporal_tier1_easy.npz" ] || \
   [ ! -f "$DATA_DIR/spatial/spatial_tier1_easy.npz" ]; then
    echo "Generating synthetic datasets..."
    python3 experiments/run_synthetic_experiments.py --generate
    
    # Move files to their proper directories
    echo "Organizing dataset files..."
    mv -f $DATA_DIR/temporal_*.npz $DATA_DIR/synthetic/ 2>/dev/null || true
    mv -f $DATA_DIR/spatial_*.npz $DATA_DIR/spatial/ 2>/dev/null || true
    
    # Create expected dataset names for config compatibility
    cp -f "$DATA_DIR/synthetic/temporal_tier1_easy.npz" "$DATA_DIR/synthetic/simple_synthetic_5class.npz" 2>/dev/null || true
    cp -f "$DATA_DIR/synthetic/temporal_tier2_medium.npz" "$DATA_DIR/synthetic/medium_synthetic_5class.npz" 2>/dev/null || true
    cp -f "$DATA_DIR/synthetic/temporal_tier3_hard.npz" "$DATA_DIR/synthetic/complex_synthetic_5class.npz" 2>/dev/null || true
    
    cp -f "$DATA_DIR/spatial/spatial_tier1_easy.npz" "$DATA_DIR/spatial/small_spatial_10class.npz" 2>/dev/null || true
    cp -f "$DATA_DIR/spatial/spatial_tier2_medium.npz" "$DATA_DIR/spatial/medium_spatial_10class.npz" 2>/dev/null || true
    cp -f "$DATA_DIR/spatial/spatial_tier3_hard.npz" "$DATA_DIR/spatial/large_spatial_10class.npz" 2>/dev/null || true
else
    echo "Datasets already exist, skipping generation."
fi

# Step 2: Run temporal/synthetic pattern benchmarks
echo "
Step 2: Running temporal pattern benchmarks..."
echo "Training and comparing ANN and SNN models on synthetic temporal data..."
python3 batch_compare.py \
    --datadir "$DATA_DIR/synthetic" \
    --outdir "$OUTPUT_DIR/temporal_benchmark" \
    --config "$CONFIG_DIR/experiments/temporal_benchmark.yaml" \
    $EPOCHS

# Step 3: Run spatial pattern benchmarks
echo "
Step 3: Running spatial pattern benchmarks..."
echo "Training and comparing ANN and SNN models on synthetic spatial data..."
python3 batch_compare.py \
    --datadir "$DATA_DIR/spatial" \
    --outdir "$OUTPUT_DIR/spatial_benchmark" \
    --config "$CONFIG_DIR/experiments/spatial_benchmark.yaml" \
    $EPOCHS

# Step 4: Generate combined comparison report
echo "
Step 4: Generating final comparison reports..."

# Create timestamp for this run
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "Benchmark run completed at: $TIMESTAMP"

# Create a master report directory
MASTER_REPORT_DIR="$OUTPUT_DIR/master_report"
mkdir -p "$MASTER_REPORT_DIR"

# Copy key visualizations to the master report directory
cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/temporal_accuracy_comparison.png" 2>/dev/null || true
cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/temporal_training_time.png" 2>/dev/null || true
cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/spatial_accuracy_comparison.png" 2>/dev/null || true
cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/spatial_training_time.png" 2>/dev/null || true

# Create master report summary
cat > "$MASTER_REPORT_DIR/master_summary.txt" << EOL
NEURAL NETWORK COMPARISON - MASTER SUMMARY
=========================================
Generated: $(date)
Run timestamp: $TIMESTAMP

This report summarizes the comprehensive comparison between
Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)
across multiple datasets and pattern recognition tasks.

SUMMARY OF FINDINGS:

1. Performance Comparison:
   - Review the accuracy charts in this directory to see relative
     performance between ANN and SNN models on each dataset type

2. Training Efficiency:
   - Review the training time ratio charts to see relative
     training efficiency between models

3. Detailed Results:
   - Full temporal benchmark results: $OUTPUT_DIR/temporal_benchmark
   - Full spatial benchmark results: $OUTPUT_DIR/spatial_benchmark

KEY OBSERVATIONS:
   - (Add key observations after viewing the results)

RECOMMENDATIONS:
   - (Add recommendations after viewing the results)

For a more detailed analysis, see the individual comparison reports in
the respective benchmark directories.
EOL

echo "
Benchmarks completed successfully! Results are available in:"
echo "  $OUTPUT_DIR"
echo "
Master report summary is available at:"
echo "  $MASTER_REPORT_DIR/master_summary.txt"
echo ""
