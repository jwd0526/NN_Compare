#!/bin/bash

# Neural Network Comparison Benchmark Runner
# This script runs a comprehensive comparison between ANN and SNN models
# on both spatial and synthetic datasets, generating visualizations and reports.
#
# Usage: ./run_benchmarks.sh [--quick] [--spatial-only] [--temporal-only] [--detailed-metrics] [--ann-optimized]
#   --quick              Run with reduced epochs for quicker testing
#   --spatial-only       Run only the spatial pattern benchmarks
#   --temporal-only      Run only the temporal pattern benchmarks
#   --detailed-metrics   Generate enhanced visualizations for temporal data
#   --ann-optimized      Use ANN-optimized datasets instead of standard spatial datasets

set -e  # Exit on error

# Configuration
OUTPUT_DIR="./results"
DATA_DIR="./data"
CONFIG_DIR="./configs"

# Default settings
QUICK_MODE=false
RUN_SPATIAL=true
RUN_TEMPORAL=true
DETAILED_METRICS=false
USE_ANN_OPTIMIZED=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --spatial-only)
            RUN_SPATIAL=true
            RUN_TEMPORAL=false
            ;;
        --temporal-only)
            RUN_SPATIAL=false
            RUN_TEMPORAL=true
            ;;
        --detailed-metrics)
            DETAILED_METRICS=true
            ;;
        --ann-optimized)
            USE_ANN_OPTIMIZED=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./run_benchmarks.sh [--quick] [--spatial-only] [--temporal-only] [--detailed-metrics] [--ann-optimized]"
            exit 1
            ;;
    esac
done

# Set epochs based on quick mode
if [ "$QUICK_MODE" = true ]; then
    EPOCHS="--epochs 3"
    echo "Running in quick mode with 3 epochs"
else
    EPOCHS="--epochs 40"  # Increased to allow for longer convergence time
    echo "Running in full mode with 40 epochs (increased for longer convergence)"
fi

# Display what benchmarks will run
if [ "$RUN_SPATIAL" = true ] && [ "$RUN_TEMPORAL" = true ]; then
    echo "Running both spatial and temporal benchmarks"
elif [ "$RUN_SPATIAL" = true ]; then
    echo "Running spatial benchmarks only"
elif [ "$RUN_TEMPORAL" = true ]; then
    echo "Running temporal benchmarks only"
fi

if [ "$DETAILED_METRICS" = true ]; then
    echo "Enhanced visualization metrics enabled"
fi

if [ "$USE_ANN_OPTIMIZED" = true ]; then
    echo "Using ANN-optimized datasets for spatial benchmarks"
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

# Check if we need temporal data
NEED_TEMPORAL_DATA=false
if [ "$RUN_TEMPORAL" = true ] && [ ! -f "$DATA_DIR/temporal/datasets/temporal_tier1_precise.npz" ]; then
    NEED_TEMPORAL_DATA=true
    # Ensure the temporal datasets directory exists
    mkdir -p "$DATA_DIR/temporal/datasets"
fi

# Check if we need spatial data
NEED_SPATIAL_DATA=false
if [ "$RUN_SPATIAL" = true ] && [ ! -f "$DATA_DIR/spatial/datasets/spatial_tier1_static.npz" ]; then
    NEED_SPATIAL_DATA=true
fi

# Generate datasets if needed
if [ "$NEED_TEMPORAL_DATA" = true ] || [ "$NEED_SPATIAL_DATA" = true ]; then
    echo "Generating synthetic datasets..."
    python3 experiments/run_synthetic_experiments.py --generate
    
    # The dataset generation script now handles file organization and creates compatibility links
    echo "Dataset generation complete with proper organization."
else
    echo "Datasets already exist, skipping generation."
fi

# Step 2: Run temporal/synthetic pattern benchmarks
if [ "$RUN_TEMPORAL" = true ]; then
    echo "
Step 2: Running temporal pattern benchmarks..."
    echo "Training and comparing ANN and SNN models on synthetic temporal data..."
    
    # Additional flags for detailed metrics
    EXTRA_FLAGS=""
    if [ "$DETAILED_METRICS" = true ]; then
        EXTRA_FLAGS="--detailed-metrics"
    fi
    
    # Check if the required temporal datasets exist - directly check the npz files
    if [ ! -f "$DATA_DIR/temporal/datasets/temporal_tier1_precise.npz" ] || \
       [ ! -f "$DATA_DIR/temporal/datasets/temporal_tier2_correlation.npz" ] || \
       [ ! -f "$DATA_DIR/temporal/datasets/temporal_tier3_complex.npz" ]; then
        echo "Warning: One or more required temporal datasets are missing!"
        echo "Checking for alternate dataset formats..."
        
        # Check if alternate format (synthetic datasets) exists
        if [ -f "$DATA_DIR/synthetic/simple_synthetic_5class.npz" ] && \
           [ -f "$DATA_DIR/synthetic/medium_synthetic_5class.npz" ] && \
           [ -f "$DATA_DIR/synthetic/complex_synthetic_5class.npz" ]; then
            
            echo "Found synthetic datasets. Copying to temporal datasets directory..."
            mkdir -p "$DATA_DIR/temporal/datasets"
            
            # Create compatibility links/copies
            cp -f "$DATA_DIR/synthetic/simple_synthetic_5class.npz" "$DATA_DIR/temporal/datasets/temporal_tier1_precise.npz"
            cp -f "$DATA_DIR/synthetic/medium_synthetic_5class.npz" "$DATA_DIR/temporal/datasets/temporal_tier2_correlation.npz"
            cp -f "$DATA_DIR/synthetic/complex_synthetic_5class.npz" "$DATA_DIR/temporal/datasets/temporal_tier3_complex.npz"
            
            echo "Dataset compatibility links created successfully."
        else
            echo "Error: Required datasets not found. Please generate them first using run_synthetic_experiments.py"
            echo "Continuing with available datasets, but results may be incomplete."
        fi
    fi
    
    # Run the dataset fix script to ensure pattern and label dimensions match
    echo "Checking and fixing dataset dimensions if needed..."
    python3 ./fix_datasets.py
    
    # Now run the benchmark with the prepared datasets
    python3 batch_compare.py \
        --datadir "$DATA_DIR/temporal/datasets" \
        --outdir "$OUTPUT_DIR/temporal_benchmark" \
        --config "$CONFIG_DIR/experiments/temporal_benchmark.yaml" \
        $EPOCHS $EXTRA_FLAGS
        
    # Copy example visualizations to the results directory
    mkdir -p "$OUTPUT_DIR/temporal_benchmark/examples"
    cp -f "$DATA_DIR/temporal/examples/"*.png "$OUTPUT_DIR/temporal_benchmark/examples/" 2>/dev/null || true
fi

# Step 3: Run spatial pattern benchmarks
if [ "$RUN_SPATIAL" = true ]; then
    if [ "$USE_ANN_OPTIMIZED" = true ]; then
        echo "
Step 3: Running ANN-optimized spatial pattern benchmarks..."
        echo "Training and comparing ANN and SNN models on ANN-optimized spatial data..."
        python3 batch_compare.py \
            --datadir "$DATA_DIR/ann_optimized/datasets" \
            --outdir "$OUTPUT_DIR/ann_optimized_benchmark" \
            --config "$CONFIG_DIR/experiments/spatial_benchmark.yaml" \
            $EPOCHS
            
        # Copy example visualizations to the results directory
        mkdir -p "$OUTPUT_DIR/ann_optimized_benchmark/examples"
        cp -f "$DATA_DIR/ann_optimized/examples/"*.png "$OUTPUT_DIR/ann_optimized_benchmark/examples/" 2>/dev/null || true
    else
        echo "
Step 3: Running spatial pattern benchmarks..."
        echo "Training and comparing ANN and SNN models on synthetic spatial data..."
        python3 batch_compare.py \
            --datadir "$DATA_DIR/spatial" \
            --outdir "$OUTPUT_DIR/spatial_benchmark" \
            --config "$CONFIG_DIR/experiments/spatial_benchmark.yaml" \
            $EPOCHS
            
        # Copy example visualizations to the results directory
        mkdir -p "$OUTPUT_DIR/spatial_benchmark/examples"
        cp -f "$DATA_DIR/spatial/examples/"*.png "$OUTPUT_DIR/spatial_benchmark/examples/" 2>/dev/null || true
    fi
fi

# Step 4: Generate combined comparison report
echo "
Step 4: Generating final comparison reports..."

# Create timestamp for this run
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "Benchmark run completed at: $TIMESTAMP"

# Create a master report directory
MASTER_REPORT_DIR="$OUTPUT_DIR/master_report"
mkdir -p "$MASTER_REPORT_DIR"

# Set output directory for spatial benchmark results
if [ "$USE_ANN_OPTIMIZED" = true ]; then
    SPATIAL_OUTPUT_DIR="$OUTPUT_DIR/ann_optimized_benchmark"
else
    SPATIAL_OUTPUT_DIR="$OUTPUT_DIR/spatial_benchmark"
fi

# Create report summary based on what was run
if [ "$RUN_TEMPORAL" = true ] && [ "$RUN_SPATIAL" = true ]; then
    # Both benchmarks were run - copy all visualizations
    cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/temporal_accuracy_comparison.png" 2>/dev/null || true
    cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/temporal_training_time.png" 2>/dev/null || true
    
    # Copy from the appropriate spatial benchmark directory
    if [ "$USE_ANN_OPTIMIZED" = true ]; then
        cp -f "$OUTPUT_DIR/ann_optimized_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/spatial_accuracy_comparison.png" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/ann_optimized_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/spatial_training_time.png" 2>/dev/null || true
    else
        cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/spatial_accuracy_comparison.png" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/spatial_training_time.png" 2>/dev/null || true
    fi
    
    # Copy enhanced temporal visualizations if they exist
    if [ "$DETAILED_METRICS" = true ]; then
        mkdir -p "$MASTER_REPORT_DIR/temporal_detailed_metrics"
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/temporal_comparison_dashboard.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/learning_dynamics.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/temporal_feature_importance.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/error_comparison_log.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/accuracy_gain.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
    fi
    
    # Create comprehensive master report
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
   - Full spatial benchmark results: $SPATIAL_OUTPUT_DIR
EOL

    if [ "$DETAILED_METRICS" = true ]; then
        cat >> "$MASTER_REPORT_DIR/master_summary.txt" << EOL

4. Enhanced Temporal Analytics:
   - Temporal Comparison Dashboard: View in temporal_detailed_metrics/
   - Learning Dynamics: See detailed learning behavior analysis
   - Temporal Feature Importance: Analysis of which temporal features
     are better handled by each model type
   - Error Analysis: Log-scale visualization of error rates
   - Accuracy Gain: First-derivative analysis of learning progress
EOL
    fi

    cat >> "$MASTER_REPORT_DIR/master_summary.txt" << EOL

KEY OBSERVATIONS:
   - (Add key observations after viewing the results)

RECOMMENDATIONS:
   - (Add recommendations after viewing the results)

For a more detailed analysis, see the individual comparison reports in
the respective benchmark directories.
EOL

elif [ "$RUN_TEMPORAL" = true ]; then
    # Only temporal benchmark was run
    cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/temporal_accuracy_comparison.png" 2>/dev/null || true
    cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/temporal_training_time.png" 2>/dev/null || true
    
    # Copy enhanced temporal visualizations if they exist
    if [ "$DETAILED_METRICS" = true ]; then
        mkdir -p "$MASTER_REPORT_DIR/temporal_detailed_metrics"
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/temporal_comparison_dashboard.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/learning_dynamics.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/temporal_feature_importance.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/error_comparison_log.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
        cp -f "$OUTPUT_DIR/temporal_benchmark/visualizations/accuracy_gain.png" "$MASTER_REPORT_DIR/temporal_detailed_metrics/" 2>/dev/null || true
    fi
    
    # Create temporal-only report
    cat > "$MASTER_REPORT_DIR/master_summary.txt" << EOL
NEURAL NETWORK COMPARISON - TEMPORAL BENCHMARK SUMMARY
=====================================================
Generated: $(date)
Run timestamp: $TIMESTAMP

This report summarizes the comparison between
Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)
on synthetic temporal pattern recognition tasks.

SUMMARY OF FINDINGS:

1. Performance Comparison:
   - Review the accuracy charts in this directory to see relative
     performance between ANN and SNN models on temporal datasets

2. Training Efficiency:
   - Review the training time ratio charts to see relative
     training efficiency between models

3. Detailed Results:
   - Full temporal benchmark results: $OUTPUT_DIR/temporal_benchmark
   - Full spatial benchmark results: $SPATIAL_OUTPUT_DIR
EOL

    if [ "$DETAILED_METRICS" = true ]; then
        cat >> "$MASTER_REPORT_DIR/master_summary.txt" << EOL

4. Enhanced Temporal Analytics:
   - Temporal Comparison Dashboard: View in temporal_detailed_metrics/
   - Learning Dynamics: See detailed learning behavior analysis
   - Temporal Feature Importance: Analysis of which temporal features
     are better handled by each model type
   - Error Analysis: Log-scale visualization of error rates
   - Accuracy Gain: First-derivative analysis of learning progress
EOL
    fi

    cat >> "$MASTER_REPORT_DIR/master_summary.txt" << EOL

KEY OBSERVATIONS:
   - (Add key observations after viewing the results)

RECOMMENDATIONS:
   - (Add recommendations after viewing the results)

For a more detailed analysis, see the individual comparison reports in
the temporal benchmark directory.
EOL

elif [ "$RUN_SPATIAL" = true ]; then
    # Only spatial benchmark was run
    cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/overall_accuracy_comparison.png" "$MASTER_REPORT_DIR/spatial_accuracy_comparison.png" 2>/dev/null || true
    cp -f "$OUTPUT_DIR/spatial_benchmark/visualizations/training_time_ratio.png" "$MASTER_REPORT_DIR/spatial_training_time.png" 2>/dev/null || true
    
    # Create spatial-only report
    cat > "$MASTER_REPORT_DIR/master_summary.txt" << EOL
NEURAL NETWORK COMPARISON - SPATIAL BENCHMARK SUMMARY
====================================================
Generated: $(date)
Run timestamp: $TIMESTAMP

This report summarizes the comparison between
Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)
on synthetic spatial pattern recognition tasks.

SUMMARY OF FINDINGS:

1. Performance Comparison:
   - Review the accuracy charts in this directory to see relative
     performance between ANN and SNN models on spatial datasets

2. Training Efficiency:
   - Review the training time ratio charts to see relative
     training efficiency between models

3. Detailed Results:
   - Full spatial benchmark results: $OUTPUT_DIR/spatial_benchmark

KEY OBSERVATIONS:
   - (Add key observations after viewing the results)

RECOMMENDATIONS:
   - (Add recommendations after viewing the results)

For a more detailed analysis, see the individual comparison reports in
the spatial benchmark directory.
EOL

fi

echo "
Benchmarks completed successfully! Results are available in:"
echo "  $OUTPUT_DIR"
echo "
Master report summary is available at:"
echo "  $MASTER_REPORT_DIR/master_summary.txt"
echo ""
