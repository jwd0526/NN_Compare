import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add the parent directory to the path so we can import from snn_lib
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from snn_lib.snn_layers import *

class SyntheticSpikeDataset(Dataset):
    """Dataset class for synthetic spike patterns"""
    
    def __init__(self, data_path, length=100, transform=None):
        """
        Args:
            data_path: Path to the .npz file containing patterns and labels
            length: Length of the spike trains
            transform: Optional transform to apply to the data
        """
        # Load the data
        data = np.load(data_path)
        self.patterns = data['patterns']
        self.labels = data['labels']
        self.length = length
        self.transform = transform
        
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the pattern and label
        pattern = self.patterns[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Apply transform if specified
        if self.transform:
            pattern = self.transform(pattern)
        
        return pattern, label

class SyntheticSNNModel(torch.nn.Module):
    """Modified SNN model for synthetic spike patterns"""
    
    def __init__(self, input_size, hidden_size, output_size, length, batch_size, tau_m=4, tau_s=1):
        super().__init__()
        
        self.length = length
        self.batch_size = batch_size
        
        # Temporal encoding layer
        self.axon1 = dual_exp_iir_layer((input_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn1 = neuron_layer(input_size, hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Hidden layer
        self.axon2 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn2 = neuron_layer(hidden_size, hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Output layer
        self.axon3 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn3 = neuron_layer(hidden_size, output_size, self.length, self.batch_size, tau_m, True, False)
        
        # Regularization
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.3)
    
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [batch, input_size, t]
        
        Returns:
            Tensor of shape [batch, output_size, t]
        """
        # First layer
        axon1_out, _ = self.axon1(inputs)
        spike_l1, _ = self.snn1(axon1_out)
        spike_l1 = self.dropout1(spike_l1)
        
        # Second layer
        axon2_out, _ = self.axon2(spike_l1)
        spike_l2, _ = self.snn2(axon2_out)
        spike_l2 = self.dropout2(spike_l2)
        
        # Output layer
        axon3_out, _ = self.axon3(spike_l2)
        spike_l3, _ = self.snn3(axon3_out)
        
        return spike_l3

def train_on_synthetic_data(model, train_loader, test_loader, device, optimizer, scheduler, 
                          epochs=100, save_path='./checkpoint'):
    """
    Train the SNN model on synthetic data
    
    Args:
        model: SNN model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to run the model on
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        epochs: Number of epochs to train for
        save_path: Path to save checkpoints
    
    Returns:
        train_acc_list: List of training accuracies
        test_acc_list: List of test accuracies
    """
    os.makedirs(save_path, exist_ok=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        correct_total = 0
        eval_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            spike_count = torch.sum(output, dim=2)
            
            # Calculate loss
            loss = criterion(spike_count, target.long())
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(spike_count.data, 1)
            eval_total += target.size(0)
            correct_total += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Calculate training accuracy
        train_acc = correct_total / eval_total
        train_acc_list.append(train_acc)
        print(f'Training accuracy: {train_acc:.4f}')
        
        # Testing
        model.eval()
        correct_total = 0
        eval_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                spike_count = torch.sum(output, dim=2)
                
                # Calculate accuracy
                _, predicted = torch.max(spike_count.data, 1)
                eval_total += target.size(0)
                correct_total += (predicted == target).sum().item()
        
        # Calculate test accuracy
        test_acc = correct_total / eval_total
        test_acc_list.append(test_acc)
        print(f'Test accuracy: {test_acc:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_path, f'synthetic_snn_checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
        }, checkpoint_path)
    
    return train_acc_list, test_acc_list

def analyze_model_predictions(model, data_loader, device):
    """
    Analyze model predictions on the test data
    
    Args:
        model: Trained SNN model
        data_loader: DataLoader for test data
        device: Device to run the model on
    """
    model.eval()
    
    # Store predictions and ground truth
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            spike_count = torch.sum(output, dim=2)
            
            # Get predictions
            _, predicted = torch.max(spike_count.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions))
    
    # Compute and print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    return all_predictions, all_labels

if __name__ == "__main__":
    import argparse
    import json
    from snn_lib.optimizers import get_optimizer
    from snn_lib.schedulers import get_scheduler
    import omegaconf
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='Train SNN on synthetic data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the synthetic data .npz file')
    parser.add_argument('--config_file', type=str, default='snn_synthetic.yaml', help='Path to the configuration file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for testing')
    args = parser.parse_args()
    
    # Load configuration
    conf = OmegaConf.load(args.config_file)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # Set seed for reproducibility
    torch.manual_seed(conf['pytorch_seed'])
    np.random.seed(conf['pytorch_seed'])
    
    # Load the data
    data = np.load(args.data_path)
    patterns = data['patterns']
    labels = data['labels']
    
    # Get dimensions
    n_samples, n_neurons, length = patterns.shape
    n_classes = len(np.unique(labels))
    
    print(f"Loaded {n_samples} samples with {n_neurons} neurons and {length} time steps")
    print(f"Number of classes: {n_classes}")
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        patterns, labels, test_size=0.2, random_state=conf['pytorch_seed'], stratify=labels)
    
    # Create datasets
    class TensorDataset(Dataset):
        def __init__(self, patterns, labels):
            self.patterns = torch.tensor(patterns, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            
        def __len__(self):
            return len(self.patterns)
            
        def __getitem__(self, idx):
            return self.patterns[idx], self.labels[idx]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=conf['hyperparameters']['batch_size'], 
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=conf['hyperparameters']['batch_size'], 
        shuffle=False,
        drop_last=True
    )
    
    # Create model
    model = SyntheticSNNModel(
        input_size=n_neurons,
        hidden_size=conf['hyperparameters']['hidden_size'],
        output_size=n_classes,
        length=length,
        batch_size=conf['hyperparameters']['batch_size'],
        tau_m=conf['hyperparameters']['tau_m'],
        tau_s=conf['hyperparameters']['tau_s']
    ).to(device)
    
    # Training
    if args.train:
        # Create optimizer and scheduler
        optimizer = get_optimizer(model.parameters(), conf)
        scheduler = get_scheduler(optimizer, conf)
        
        # Train the model
        print("Starting training...")
        train_acc_list, test_acc_list = train_on_synthetic_data(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=conf['hyperparameters']['epoch'],
            save_path=conf['checkpoint_base_path']
        )
        
        # Save results
        results = {
            'train_acc': train_acc_list,
            'test_acc': test_acc_list
        }
        
        with open(os.path.join(conf['checkpoint_base_path'], 'training_results.json'), 'w') as f:
            json.dump(results, f)
        
        # Find best model
        best_epoch = np.argmax(test_acc_list)
        print(f"Best model at epoch {best_epoch} with test accuracy {test_acc_list[best_epoch]:.4f}")
        
    # Testing
    if args.test:
        if args.checkpoint:
            # Load model from checkpoint
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.checkpoint}")
            print(f"Model was trained for {checkpoint['epoch']} epochs")
            print(f"Training accuracy: {checkpoint['train_acc']:.4f}")
            print(f"Test accuracy: {checkpoint['test_acc']:.4f}")
        
        # Analyze model predictions
        print("Analyzing model predictions...")
        predictions, labels = analyze_model_predictions(
            model=model,
            data_loader=test_loader,
            device=device
        )