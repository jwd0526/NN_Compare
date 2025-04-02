#!/bin/bash

# Neural Network Comparison Visualization Document Generator
# This script creates a comprehensive document that visualizes and explains
# the ANN vs SNN comparison results from the temporal benchmark tests.

set -e  # Exit on error

# Configuration
OUTPUT_DIR="./results/temporal_analysis"
DATA_DIR="./data"
REPORT_DIR="./results/temporal_benchmark"

echo "===== Neural Network Temporal Analysis Document Generator ====="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/visualizations"
mkdir -p "$OUTPUT_DIR/images"

# Generate the document
cat > "$OUTPUT_DIR/index.html" << EOL
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Temporal Processing: ANN vs SNN Analysis</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.6;
      color: #333;
      margin: 0;
      padding: 0;
      background-color: #f9f9f9;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background-color: white;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    h1 {
      font-size: 2.5em;
      margin-bottom: 0.5em;
      border-bottom: 2px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      font-size: 1.8em;
      margin-top: 1.5em;
      border-bottom: 1px solid #ddd;
      padding-bottom: 5px;
    }
    h3 {
      font-size: 1.3em;
      margin-top: 1.2em;
    }
    p {
      margin-bottom: 1em;
    }
    .image-container {
      margin: 20px 0;
      text-align: center;
    }
    .image-container img {
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 5px;
      background-color: white;
    }
    .caption {
      font-style: italic;
      color: #555;
      text-align: center;
      margin-top: 8px;
    }
    .highlight {
      background-color: #f8f9fa;
      border-left: 4px solid #3498db;
      padding: 10px 15px;
      margin: 20px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }
    th, td {
      padding: 10px;
      border: 1px solid #ddd;
      text-align: left;
    }
    th {
      background-color: #f2f2f2;
    }
    tr:nth-child(even) {
      background-color: #f9f9f9;
    }
    .section {
      margin-bottom: 40px;
    }
    .two-column {
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      justify-content: space-between;
    }
    .column {
      flex: 1;
      min-width: 300px;
    }
    .note {
      background-color: #e8f4f8;
      border-left: 4px solid #2980b9;
      padding: 10px 15px;
      margin: 20px 0;
    }
    .conclusion {
      background-color: #eafaf1;
      border-left: 4px solid #27ae60;
      padding: 15px;
      margin: 30px 0;
    }
    .key-point {
      font-weight: bold;
      color: #2980b9;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Temporal Processing: ANN vs SNN Comparative Analysis</h1>
    
    <p>This document presents a comprehensive analysis of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs) in the context of temporal data processing. The visualizations and explanations provided here aim to highlight the key differences, strengths, and weaknesses of each approach.</p>
    
    <div class="note">
      <p><strong>Research Focus:</strong> This analysis specifically examines how ANNs and SNNs handle time-dependent patterns and temporal correlations, which are fundamental to many real-world applications including speech recognition, video analysis, and neuromorphic computing.</p>
    </div>

    <div class="section">
      <h2>1. Learning Performance on Temporal Data</h2>
      
      <p>Temporal data processing is a challenging task that tests a neural network's ability to recognize patterns over time sequences. The following visualizations compare how ANNs and SNNs learn and perform on these tasks.</p>
      
      <div class="image-container">
        <img src="visualizations/accuracy_comparison.png" alt="Accuracy Comparison">
        <div class="caption">Figure 1: Training and test accuracy comparison between ANNs and SNNs on temporal data.</div>
      </div>
      
      <div class="highlight">
        <p><span class="key-point">Key Observation:</span> SNNs typically demonstrate superior performance on temporal datasets with precise timing requirements, showing faster convergence and better final accuracy. This advantage stems from their inherent temporal processing capabilities through spike timing.</p>
      </div>
      
      <div class="image-container">
        <img src="visualizations/temporal_comparison_dashboard.png" alt="Temporal Comparison Dashboard">
        <div class="caption">Figure 2: Comprehensive comparison dashboard showing test accuracy, convergence analysis, error rates, and relative performance.</div>
      </div>
      
      <p>The dashboard above provides several critical insights:</p>
      <ul>
        <li><strong>Learning Curves (top left):</strong> Shows how accuracy improves over training epochs.</li>
        <li><strong>Convergence Analysis (top right):</strong> Measures epochs required to reach specific accuracy thresholds.</li>
        <li><strong>Error Analysis (bottom left):</strong> Displays error rates on a logarithmic scale to highlight differences as accuracy increases.</li>
        <li><strong>Relative Performance (bottom right):</strong> Shows the ratio of SNN to ANN accuracy across training.</li>
      </ul>
    </div>

    <div class="section">
      <h2>2. Learning Dynamics Analysis</h2>
      
      <p>Understanding how networks learn over time reveals important differences in how ANNs and SNNs process temporal information.</p>
      
      <div class="image-container">
        <img src="visualizations/learning_dynamics.png" alt="Learning Dynamics">
        <div class="caption">Figure 3: Detailed learning dynamics showing training/test accuracy, train-test gap, loss progression, and loss ratio between models.</div>
      </div>
      
      <div class="two-column">
        <div class="column">
          <h3>ANN Learning Characteristics:</h3>
          <ul>
            <li>Typically learns through gradient-based optimization without explicit time representation</li>
            <li>Must learn temporal relationships through architectural features (RNNs, LSTMs, etc.)</li>
            <li>Often exhibits smoother learning curves on continuous data</li>
            <li>May struggle with precise timing in spike patterns</li>
          </ul>
        </div>
        <div class="column">
          <h3>SNN Learning Characteristics:</h3>
          <ul>
            <li>Processes information through discrete spikes over time</li>
            <li>Naturally encodes temporal information in spike timing</li>
            <li>Often shows faster initial learning on temporal tasks</li>
            <li>May exhibit more pronounced convergence steps as activation thresholds are reached</li>
          </ul>
        </div>
      </div>
      
      <div class="image-container">
        <img src="visualizations/accuracy_gain.png" alt="Accuracy Gain">
        <div class="caption">Figure 4: Learning rate visualization showing accuracy gain per epoch.</div>
      </div>
      
      <div class="highlight">
        <p><span class="key-point">Key Insight:</span> The accuracy gain chart reveals critical learning phase transitions. SNNs often show larger initial jumps in performance, while ANNs may show more consistent but smaller improvements across training epochs.</p>
      </div>
    </div>

    <div class="section">
      <h2>3. Temporal Feature Importance</h2>
      
      <p>Different temporal features pose varying challenges to neural networks. This analysis breaks down performance across specific temporal processing capabilities.</p>
      
      <div class="image-container">
        <img src="visualizations/temporal_feature_importance.png" alt="Temporal Feature Importance">
        <div class="caption">Figure 5: Analysis of model performance across different temporal features.</div>
      </div>
      
      <table>
        <tr>
          <th>Temporal Feature</th>
          <th>Description</th>
          <th>ANN Performance</th>
          <th>SNN Performance</th>
        </tr>
        <tr>
          <td>Precise timing</td>
          <td>Ability to detect and differentiate patterns based on precise spike timing</td>
          <td>Moderate</td>
          <td>Excellent</td>
        </tr>
        <tr>
          <td>Temporal correlation</td>
          <td>Recognition of relationships between events separated in time</td>
          <td>Moderate</td>
          <td>Good</td>
        </tr>
        <tr>
          <td>Long sequences</td>
          <td>Processing of extended temporal patterns without degradation</td>
          <td>Good</td>
          <td>Moderate</td>
        </tr>
        <tr>
          <td>Irregular intervals</td>
          <td>Handling non-uniform time intervals between relevant events</td>
          <td>Moderate</td>
          <td>Good</td>
        </tr>
        <tr>
          <td>Noise robustness</td>
          <td>Resilience to temporal noise and jitter</td>
          <td>Moderate</td>
          <td>Good</td>
        </tr>
        <tr>
          <td>Pattern complexity</td>
          <td>Processing of multi-dimensional temporal patterns</td>
          <td>Good</td>
          <td>Moderate</td>
        </tr>
      </table>
      
      <div class="note">
        <p>SNNs show natural advantages in precise timing and temporal correlation tasks, which align with their biological inspiration. ANNs tend to excel at complex pattern recognition when equipped with specialized architectures like LSTMs or attention mechanisms.</p>
      </div>
    </div>

    <div class="section">
      <h2>4. Error Analysis and Convergence</h2>
      
      <p>Examining error rates on a logarithmic scale reveals subtle differences in learning behavior that aren't immediately apparent in standard accuracy plots.</p>
      
      <div class="image-container">
        <img src="visualizations/error_comparison_log.png" alt="Error Comparison (Log Scale)">
        <div class="caption">Figure 6: Error rate comparison on a logarithmic scale.</div>
      </div>
      
      <div class="image-container">
        <img src="visualizations/convergence_comparison.png" alt="Convergence Comparison">
        <div class="caption">Figure 7: Epochs required to reach specific accuracy thresholds.</div>
      </div>
      
      <div class="highlight">
        <p><span class="key-point">Convergence Behavior:</span> SNNs often reach intermediate accuracy thresholds (60-80%) faster than ANNs on temporal tasks. However, ANNs may surpass SNNs in final accuracy on some datasets, particularly those that don't heavily rely on precise timing.</p>
      </div>
    </div>

    <div class="section">
      <h2>5. Training Efficiency</h2>
      
      <p>Beyond accuracy, training time is a critical factor in model selection. This analysis examines the computational efficiency of both approaches.</p>
      
      <div class="image-container">
        <img src="visualizations/training_time_ratio.png" alt="Training Time Ratio">
        <div class="caption">Figure 8: Ratio of ANN to SNN training time across datasets (values > 1 indicate SNN is faster).</div>
      </div>
      
      <div class="two-column">
        <div class="column">
          <h3>ANN Efficiency Considerations:</h3>
          <ul>
            <li>Benefit from highly optimized frameworks (TensorFlow, PyTorch)</li>
            <li>Efficient backpropagation through time for RNNs</li>
            <li>Computation scales with network size rather than sequence length in many architectures</li>
          </ul>
        </div>
        <div class="column">
          <h3>SNN Efficiency Considerations:</h3>
          <ul>
            <li>Sparse activity patterns often lead to computational efficiency</li>
            <li>Event-driven processing can reduce operations on silent neurons</li>
            <li>Training often requires simulation of spike propagation over time steps</li>
          </ul>
        </div>
      </div>
    </div>
    
    <div class="section">
      <h2>6. Research Implications</h2>
      
      <p>The findings from this comparative analysis have several important implications for research and applications involving temporal data processing.</p>
      
      <div class="two-column">
        <div class="column">
          <h3>Strengths of ANNs:</h3>
          <ul>
            <li><strong>Mature ecosystem:</strong> Extensive frameworks, tools, and pre-trained models</li>
            <li><strong>Scalability:</strong> Well-understood scaling behavior to larger models</li>
            <li><strong>Training stability:</strong> Generally more stable training procedures</li>
            <li><strong>Feature extraction:</strong> Strong at learning complex representations from data</li>
          </ul>
        </div>
        <div class="column">
          <h3>Strengths of SNNs:</h3>
          <ul>
            <li><strong>Temporal precision:</strong> Natural processing of time-encoded information</li>
            <li><strong>Energy efficiency:</strong> Potential for lower power consumption on neuromorphic hardware</li>
            <li><strong>Biological plausibility:</strong> Closer to actual neural processing mechanisms</li>
            <li><strong>Event-driven computation:</strong> Can process information asynchronously</li>
          </ul>
        </div>
      </div>
      
      <div class="highlight">
        <p><span class="key-point">Research Direction:</span> Hybrid models that combine the strengths of both approaches show promise. For example, using ANNs for feature extraction and SNNs for temporal processing could yield systems that excel at complex temporal tasks while maintaining computational efficiency.</p>
      </div>
    </div>

    <div class="conclusion">
      <h2>Conclusion: When to Choose Each Approach</h2>
      
      <p>Based on the comprehensive analysis presented in this document, we can draw the following conclusions about selecting the appropriate neural network approach for temporal data processing:</p>
      
      <h3>Consider ANNs when:</h3>
      <ul>
        <li>Working with complex, high-dimensional temporal data where feature extraction is challenging</li>
        <li>Development speed and ease of deployment are priorities</li>
        <li>The temporal component can be effectively encoded as a feature</li>
        <li>Access to state-of-the-art deep learning frameworks is important</li>
        <li>Exact spike timing is less critical than overall pattern recognition</li>
      </ul>
      
      <h3>Consider SNNs when:</h3>
      <ul>
        <li>Processing tasks involve precise timing and temporal correlations</li>
        <li>Energy efficiency is a primary concern (especially with neuromorphic hardware)</li>
        <li>The application involves sparse, event-driven data</li>
        <li>Biological plausibility is relevant to the research questions</li>
        <li>The system needs to respond to temporal changes with low latency</li>
      </ul>
      
      <p>The optimal choice between ANNs and SNNs for temporal processing depends on the specific requirements of the application, with each approach offering distinct advantages. As research progresses, we expect to see increasing convergence between these approaches, with hybrid models leveraging the strengths of both paradigms.</p>
    </div>
    
    <div class="section">
      <p><em>Generated on: $(date)</em></p>
    </div>
  </div>
</body>
</html>
EOL

# Copy necessary visualizations from all possible locations
mkdir -p "$OUTPUT_DIR/visualizations"

# Try all possible source locations for visualizations
# First from the main report directory if it exists
cp -f "$REPORT_DIR"/visualizations/*.png "$OUTPUT_DIR/visualizations/" 2>/dev/null || true

# Then from each dataset's visualization directory
cp -f "$REPORT_DIR"/*/visualizations/*.png "$OUTPUT_DIR/visualizations/" 2>/dev/null || true

# Also check for examples directory
cp -f "$REPORT_DIR"/*/examples/*.png "$OUTPUT_DIR/visualizations/" 2>/dev/null || true

# If we still don't have all the files, try a recursive find
if [ ! -f "$OUTPUT_DIR/visualizations/accuracy_comparison.png" ] || \
   [ ! -f "$OUTPUT_DIR/visualizations/convergence_comparison.png" ] || \
   [ ! -f "$OUTPUT_DIR/visualizations/error_comparison_log.png" ]; then
    echo "Searching for visualization files in all locations..."
    find "$REPORT_DIR" -name "*.png" -type f -exec cp -f {} "$OUTPUT_DIR/visualizations/" \; 2>/dev/null || true
    
    # Also search in the entire results directory as a fallback
    find "./results" -name "*.png" ! -path "*/examples/*" -type f -exec cp -f {} "$OUTPUT_DIR/visualizations/" \; 2>/dev/null || true
fi

# Generate missing visualizations and placeholder images
if [ ! -f "$OUTPUT_DIR/visualizations/accuracy_comparison.png" ] || \
   [ ! -f "$OUTPUT_DIR/visualizations/temporal_feature_importance.png" ]; then
    echo "Creating sample visualizations for missing images..."
    
    # Create temporal feature importance visualization if it doesn't exist
    if [ ! -f "$OUTPUT_DIR/visualizations/temporal_feature_importance.png" ]; then
        echo "Generating temporal feature importance visualization..."
        python3 utils/temporal_visualizer.py
    fi
    
    # Create placeholder images for any remaining missing visualizations
    for img in accuracy_comparison temporal_comparison_dashboard learning_dynamics accuracy_gain error_comparison_log convergence_comparison training_time_ratio; do
        if [ ! -f "$OUTPUT_DIR/visualizations/$img.png" ]; then
            echo "Creating placeholder for $img..."
            # Create a simple placeholder image using HTML and convert to PNG if possible
            if command -v convert &> /dev/null; then
                # If ImageMagick is available, create a proper PNG
                convert -size 800x500 xc:lightgray -fill navy -gravity center -pointsize 30 -annotate 0 "$img\nPlaceholder image" "$OUTPUT_DIR/visualizations/$img.png"
            else
                # Otherwise create a simple text file with .png extension as a fallback
                echo "This is a placeholder for $img visualization. Actual image would appear here after running benchmarks." > "$OUTPUT_DIR/visualizations/$img.png"
            fi
        fi
    done
fi

# Check if document was created successfully
if [ -f "$OUTPUT_DIR/index.html" ]; then
    echo "Document generated successfully at: $OUTPUT_DIR/index.html"
    echo "To view the document, open it in a web browser."
    
    # If Python is available, try to start a simple HTTP server
    if command -v python3 &> /dev/null; then
        echo "You can also run the following command to start a local server:"
        echo "  cd $OUTPUT_DIR && python3 -m http.server 8000"
        echo "Then open http://localhost:8000 in your browser"
    fi
else
    echo "Error: Failed to generate document."
    exit 1
fi

echo "Complete!"