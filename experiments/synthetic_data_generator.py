import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

class SyntheticSpikeGenerator:
    """Generate synthetic spike patterns for SNN testing and evaluation with multiple difficulty tiers"""
    
    def __init__(self, output_dir='./data'):
        """
        Initialize the spike generator
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_regular_pattern(self, n_neurons=10, length=100, period=10, jitter=0, n_samples=100, phase_shift=0):
        """
        Generate regularly spaced spikes with optional jitter and phase shift
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            period: Regular spike interval
            jitter: Random jitter to add to spike timing (std dev in time steps)
            n_samples: Number of samples to generate
            phase_shift: Phase shift for the regular pattern (in time steps)
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 0)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Create base spike times with possible phase shift
                spike_times = np.arange(phase_shift, length, period)
                
                # Add jitter
                if jitter > 0:
                    spike_times = spike_times + np.random.normal(0, jitter, size=len(spike_times))
                    spike_times = np.clip(spike_times, 0, length-1).astype(int)
                
                # Set spikes at the specified times
                valid_times = spike_times[(spike_times >= 0) & (spike_times < length)]
                patterns[i, j, valid_times] = 1.0
                
        # All samples in this set are class 2
        labels = np.full(n_samples, 2, dtype=np.int32)
        
        return patterns, labels

    def generate_sequential_pattern(self, n_neurons=10, length=100, start_time=20, 
                                  interval=2, jitter=0.5, n_samples=100):
        """
        Generate patterns where neurons fire in sequence with fixed intervals
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            start_time: Time step when the first neuron fires
            interval: Time interval between consecutive neurons
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 3)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Calculate spike time for this neuron
                spike_time = start_time + j * interval
                
                # Add jitter
                if jitter > 0:
                    spike_time = int(spike_time + np.random.normal(0, jitter))
                    spike_time = max(0, min(length-1, spike_time))
                
                if spike_time < length:
                    patterns[i, j, spike_time] = 1.0
        
        # All samples in this set are class 3
        labels = np.full(n_samples, 3, dtype=np.int32)
        
        return patterns, labels
    
    def generate_frequency_modulated_pattern(self, n_neurons=10, length=100, base_period=10,
                                           modulation=0.2, jitter=0.5, n_samples=100):
        """
        Generate patterns with frequency modulation (periods that slowly change)
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            base_period: Base period for the spike pattern
            modulation: Amount of period modulation (factor)
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all specified)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Start with base period
                current_period = base_period
                current_time = 0
                
                # Generate spikes with gradually changing period
                while current_time < length:
                    # Set spike
                    if current_time >= 0:
                        actual_time = int(current_time)
                        if actual_time < length:
                            patterns[i, j, actual_time] = 1.0
                    
                    # Gradually modulate the period
                    current_period += base_period * modulation * (np.random.random() - 0.5) * 2
                    current_period = max(base_period * 0.5, min(base_period * 1.5, current_period))
                    
                    # Add jitter
                    current_time += current_period + np.random.normal(0, jitter)
        
        # Default label
        labels = np.full(n_samples, 4, dtype=np.int32)
        
        return patterns, labels
    
    def generate_spatial_patterns(self, width=10, height=10, length=100, n_patterns=4, n_samples=100):
        """
        Generate 2D spatial-temporal patterns (circles, lines, spirals, etc.)
        
        Args:
            width: Width of the 2D grid
            height: Height of the 2D grid
            length: Sequence length in time steps
            n_patterns: Number of different pattern types to generate
            n_samples: Total number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, width*height, length)
            labels: Array of shape (n_samples,) containing class labels
        """
        samples_per_pattern = n_samples // n_patterns
        patterns = np.zeros((n_samples, width*height, length), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int32)
        
        sample_idx = 0
        
        # Pattern 1: Horizontal line sweeping from top to bottom
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Time at which the line starts sweeping
            start_time = np.random.randint(10, 30)
            
            # Speed of the sweep (in time steps per row)
            speed = np.random.randint(2, 5)
            
            # Create the sweeping line
            for y in range(height):
                t = start_time + y * speed
                if t < length:
                    pattern[y, :, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 0
            sample_idx += 1
        
        # Pattern 2: Vertical line sweeping from left to right
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Time at which the line starts sweeping
            start_time = np.random.randint(10, 30)
            
            # Speed of the sweep (in time steps per column)
            speed = np.random.randint(2, 5)
            
            # Create the sweeping line
            for x in range(width):
                t = start_time + x * speed
                if t < length:
                    pattern[:, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 1
            sample_idx += 1
        
        # Pattern 3: Expanding circle
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Center coordinates
            center_y, center_x = height // 2, width // 2
            
            # Time at which the circle starts expanding
            start_time = np.random.randint(10, 30)
            
            # Speed of expansion (in time steps per unit radius)
            speed = np.random.randint(3, 6)
            
            # Maximum radius
            max_radius = min(height, width) // 2
            
            # Create expanding circle
            for r in range(1, max_radius + 1):
                t = start_time + r * speed
                if t < length:
                    # Draw a rough circle
                    for y in range(height):
                        for x in range(width):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if abs(dist - r) < 0.5:  # Close enough to this radius
                                pattern[y, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 2
            sample_idx += 1
        
        # Pattern 4: Diagonal movement
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Starting corner (randomly choose one of four corners)
            corner = np.random.randint(0, 4)
            if corner == 0:  # Top-left to bottom-right
                start_y, start_x = 0, 0
                dir_y, dir_x = 1, 1
            elif corner == 1:  # Top-right to bottom-left
                start_y, start_x = 0, width - 1
                dir_y, dir_x = 1, -1
            elif corner == 2:  # Bottom-left to top-right
                start_y, start_x = height - 1, 0
                dir_y, dir_x = -1, 1
            else:  # Bottom-right to top-left
                start_y, start_x = height - 1, width - 1
                dir_y, dir_x = -1, -1
            
            # Time at which the movement starts
            start_time = np.random.randint(10, 30)
            
            # Speed of movement (in time steps per diagonal step)
            speed = np.random.randint(2, 4)
            
            # Maximum number of steps
            max_steps = min(height, width)
            
            # Create diagonal movement
            for step in range(max_steps):
                t = start_time + step * speed
                if t < length:
                    y = start_y + step * dir_y
                    x = start_x + step * dir_x
                    
                    # Check if coordinates are valid
                    if 0 <= y < height and 0 <= x < width:
                        pattern[y, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 3
            sample_idx += 1
        
        return patterns[:sample_idx], labels[:sample_idx]
    
    def generate_subtle_spatial_patterns(self, width=10, height=10, length=100, pattern_type='diagonal',
                                       variations=4, n_samples_per_variation=100):
        """
        Generate subtle variations of a specific spatial pattern type
        
        Args:
            width: Width of the 2D grid
            height: Height of the 2D grid
            length: Sequence length in time steps
            pattern_type: Base pattern type ('diagonal', 'lines', 'circle')
            variations: Number of subtle variations to generate
            n_samples_per_variation: Number of samples per variation
            
        Returns:
            patterns: Array of shape (n_samples, width*height, length)
            labels: Array of shape (n_samples,) containing class labels
        """
        total_samples = variations * n_samples_per_variation
        patterns = np.zeros((total_samples, width*height, length), dtype=np.float32)
        labels = np.zeros(total_samples, dtype=np.int32)
        
        sample_idx = 0
        
        if pattern_type == 'diagonal':
            # Create base diagonal pattern
            base_pattern = np.zeros((height, width, length))
            
            # Basic diagonal timing
            for i in range(min(width, height)):
                base_pattern[i, i, 10+i*5:15+i*5] = 1
                
            for v in range(variations):
                # Create variation
                var_pattern = base_pattern.copy()
                
                if v == 0:
                    # Original pattern
                    pass
                elif v == 1:
                    # Shift timing slightly
                    new_pattern = np.zeros_like(var_pattern)
                    new_pattern[:, :, 2:] = var_pattern[:, :, :-2]
                    var_pattern = new_pattern
                elif v == 2:
                    # Small spatial shift
                    new_pattern = np.zeros_like(var_pattern)
                    for y in range(height-1):
                        for x in range(width-1):
                            new_pattern[y+1, x, :] = var_pattern[y, x, :]
                    var_pattern = new_pattern
                elif v == 3:
                    # Skip one position
                    for i in range(min(width, height)):
                        if i % 3 == 1:  # Skip every third position
                            var_pattern[i, i, :] = 0
                
                # Create samples for this variation
                for i in range(n_samples_per_variation):
                    # Add tiny random noise (1%)
                    sample = var_pattern.copy()
                    mask = np.random.random(sample.shape) < 0.01
                    sample[mask] = 1 - sample[mask]
                    
                    # Reshape and add to patterns
                    patterns[sample_idx] = sample.reshape(height*width, length)
                    labels[sample_idx] = v
                    sample_idx += 1
        
        elif pattern_type == 'lines':
            # Create base horizontal lines pattern
            base_pattern = np.zeros((height, width, length))
            
            # Add three horizontal lines with sequential activation
            for y in [height//4, height//2, 3*height//4]:
                for t in range(20, 80, 20):
                    base_pattern[y, :, t:t+5] = 1
            
            for v in range(variations):
                # Create variation
                var_pattern = base_pattern.copy()
                
                if v == 0:
                    # Original pattern
                    pass
                elif v == 1:
                    # Shift timing slightly
                    new_pattern = np.zeros_like(var_pattern)
                    new_pattern[:, :, 5:] = var_pattern[:, :, :-5]
                    var_pattern = new_pattern
                elif v == 2:
                    # Different spacing between lines
                    new_pattern = np.zeros_like(var_pattern)
                    for y in [height//5, 2*height//5, 4*height//5]:
                        for t in range(20, 80, 20):
                            new_pattern[y, :, t:t+5] = 1
                    var_pattern = new_pattern
                elif v == 3:
                    # Shorter lines
                    new_pattern = np.zeros_like(var_pattern)
                    for y in [height//4, height//2, 3*height//4]:
                        for t in range(20, 80, 20):
                            new_pattern[y, width//4:3*width//4, t:t+5] = 1
                    var_pattern = new_pattern
                
                # Create samples for this variation
                for i in range(n_samples_per_variation):
                    # Add tiny random noise (1%)
                    sample = var_pattern.copy()
                    mask = np.random.random(sample.shape) < 0.01
                    sample[mask] = 1 - sample[mask]
                    
                    # Reshape and add to patterns
                    patterns[sample_idx] = sample.reshape(height*width, length)
                    labels[sample_idx] = v
                    sample_idx += 1
        
        elif pattern_type == 'circle':
            # Create base expanding circle pattern
            base_pattern = np.zeros((height, width, length))
            
            # Center coordinates
            center_y, center_x = height // 2, width // 2
            
            # Create expanding circle
            for r in range(1, min(height, width) // 3):
                t = 20 + r * 10
                if t < length:
                    for y in range(height):
                        for x in range(width):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if abs(dist - r) < 0.5:
                                base_pattern[y, x, t:t+5] = 1
            
            for v in range(variations):
                # Create variation
                var_pattern = base_pattern.copy()
                
                if v == 0:
                    # Original pattern
                    pass
                elif v == 1:
                    # Shift timing slightly
                    new_pattern = np.zeros_like(var_pattern)
                    new_pattern[:, :, 3:] = var_pattern[:, :, :-3]
                    var_pattern = new_pattern
                elif v == 2:
                    # Slightly off-center
                    new_pattern = np.zeros_like(var_pattern)
                    off_center_y, off_center_x = center_y + 1, center_x + 1
                    
                    for r in range(1, min(height, width) // 3):
                        t = 20 + r * 10
                        if t < length:
                            for y in range(height):
                                for x in range(width):
                                    dist = np.sqrt((y - off_center_y)**2 + (x - off_center_x)**2)
                                    if abs(dist - r) < 0.5:
                                        new_pattern[y, x, t:t+5] = 1
                    var_pattern = new_pattern
                elif v == 3:
                    # Different expansion rate
                    new_pattern = np.zeros_like(var_pattern)
                    
                    for r in range(1, min(height, width) // 3):
                        t = 20 + r * 8  # Different timing
                        if t < length:
                            for y in range(height):
                                for x in range(width):
                                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                                    if abs(dist - r) < 0.5:
                                        new_pattern[y, x, t:t+5] = 1
                    var_pattern = new_pattern
                
                # Create samples for this variation
                for i in range(n_samples_per_variation):
                    # Add tiny random noise (1%)
                    sample = var_pattern.copy()
                    mask = np.random.random(sample.shape) < 0.01
                    sample[mask] = 1 - sample[mask]
                    
                    # Reshape and add to patterns
                    patterns[sample_idx] = sample.reshape(height*width, length)
                    labels[sample_idx] = v
                    sample_idx += 1
        
        return patterns, labels
    
    def save_dataset(self, patterns, labels, filename):
        """
        Save patterns and labels to .npz file
        
        Args:
            patterns: Array of spike patterns
            labels: Array of class labels
            filename: Base filename to save to
        """
        save_path = os.path.join(self.output_dir, filename + '.npz')
        np.savez(save_path, patterns=patterns, labels=labels)
        print(f"Saved dataset to {save_path}")
    
    def visualize_patterns(self, patterns, labels, n_samples=5, filename=None):
        """
        Visualize a few examples of each pattern class
        
        Args:
            patterns: Array of spike patterns
            labels: Array of class labels
            n_samples: Number of samples per class to visualize
            filename: If provided, save the figure to this file
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples*3, n_classes*3))
        
        for i, label in enumerate(unique_labels):
            # Get indices for this class
            indices = np.where(labels == label)[0]
            
            # Select a few samples
            selected_indices = indices[:n_samples]
            
            for j, idx in enumerate(selected_indices):
                if n_classes == 1:
                    ax = axes[j] if n_samples > 1 else axes
                else:
                    ax = axes[i, j] if n_samples > 1 else axes[i]
                
                # Display the spike raster plot
                spike_data = patterns[idx]
                
                if spike_data.ndim == 2:  # Standard (neuron, time) format
                    neurons, times = np.where(spike_data > 0)
                    ax.scatter(times, neurons, s=5, marker='|', c='black')
                    ax.set_ylim(-0.5, spike_data.shape[0] - 0.5)
                    ax.invert_yaxis()
                else:
                    # Handle 3D data - flatten the spatial dimensions
                    ax.imshow(spike_data.sum(axis=2), cmap='viridis')
                
                ax.set_title(f"Class {label}")
                
                # Remove unnecessary ticks
                if j == 0:
                    ax.set_ylabel("Neuron")
                else:
                    ax.set_yticks([])
                
                if i == n_classes - 1:
                    ax.set_xlabel("Time")
                else:
                    ax.set_xticks([])
        
        plt.tight_layout()
        
        if filename:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        
        # Close the figure instead of showing it
        plt.close(fig)
        labels = np.zeros(n_samples, dtype=np.int32)
        
        return patterns, labels
    
    def generate_burst_pattern(self, n_neurons=10, length=100, burst_start=20, 
                               burst_width=10, n_spikes=5, jitter=1, n_samples=100):
        """
        Generate spike bursts with a specified timing
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            burst_start: Time step where burst begins
            burst_width: Width of the burst window
            n_spikes: Number of spikes in the burst
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 1)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Generate random spike times within the burst window
                spike_times = burst_start + np.random.randint(0, burst_width, size=n_spikes)
                spike_times = np.clip(spike_times, 0, length-1)
                
                # Add jitter
                if jitter > 0:
                    spike_times = spike_times + np.random.normal(0, jitter, size=len(spike_times))
                    spike_times = np.clip(spike_times, 0, length-1).astype(int)
                
                patterns[i, j, spike_times] = 1.0
        
        # All samples in this set are class 1
        labels = np.ones(n_samples, dtype=np.int32)
        
        return patterns, labels
    
    def generate_sync_pattern(self, n_neurons=10, length=100, sync_times=[20, 50, 80], 
                             jitter=1, n_samples=100):
        """
        Generate patterns where neurons fire synchronously at specific times
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            sync_times: List of time points where synchronous firing occurs
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 2)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # For each synchronization time
                for sync_t in sync_times:
                    # Add jitter
                    if jitter > 0:
                        actual_t = int(sync_t + np.random.normal(0, jitter))
                        actual_t = max(0, min(length-1, actual_t))
                    else:
                        actual_t = sync_t
                    
                    patterns[i, j, actual_t] = 1.0
        
        # All samples in