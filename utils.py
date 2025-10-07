import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def compute_mean_std():
    # Compute mean and std of training data
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    
    images, _ = next(iter(loader))
    
    # For CIFAR-10 (RGB images), compute mean and std for each channel
    train_mean = images.mean(dim=(0, 2, 3))  # Mean across batch, height, width dimensions
    train_std = images.std(dim=(0, 2, 3))    # Std across batch, height, width dimensions
    
    return train_mean, train_std

import numpy as np

class AlbumentationsTransform:
    """Wrapper to make Albumentations transforms work with PyTorch datasets"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        # Convert PIL image to numpy array
        if hasattr(image, 'convert'):  # PIL Image
            image = np.array(image)
        
        # Apply Albumentations transform
        transformed = self.transform(image=image)
        return transformed['image']

def data_loader(train_mean, train_std, batch_size_train=256, batch_size_test=5000, train_transform=None, test_transform=None):
    # Handle Albumentations vs torchvision transforms
    if train_transform is not None:
        # If it's an Albumentations transform, wrap it
        if hasattr(train_transform, 'transforms'):  # Albumentations Compose object
            train_transform_final = AlbumentationsTransform(train_transform)
        else:
            train_transform_final = train_transform
    else:
        # Default transform for training
        train_transform_final = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])
    
    if test_transform is not None:
        # If it's an Albumentations transform, wrap it
        if hasattr(test_transform, 'transforms'):  # Albumentations Compose object
            test_transform_final = AlbumentationsTransform(test_transform)
        else:
            test_transform_final = test_transform
    else:
        # Default transform for testing
        test_transform_final = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])
    
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform_final)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform_final)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
    return trainloader, testloader

def evaluate_model(model, device, dataloader, criterion):
    """
    Evaluate the model on a given dataset and return loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def train_model(model, device, trainloader, testloader, optimizer, criterion, epochs=20, scheduler=None):
    """
    Train the model for specified epochs and track training/test metrics.
    """
    # Lists to store training statistics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Training loop for the specified number of epochs
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ONLY step schedulers that require per-batch stepping
            if scheduler is not None and isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        
        # Evaluate on test set using the new function
        test_loss, test_accuracy = evaluate_model(model, device, testloader, criterion)
        
        # Store statistics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{epochs}] - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Step schedulers that require per-epoch stepping (AFTER the epoch)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            elif not isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
                # CosineAnnealingLR, StepLR, etc. - step once per epoch
                scheduler.step()
    
    return train_losses, train_accuracies, test_losses, test_accuracies

def initialize_model(cnn_model, optimizer_func, **optimizer_params):
    # Initialize model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_func(model.parameters(), **optimizer_params)
    return model, criterion, optimizer, device

def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # Plot training and test accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_train_test_metrics(train_losses, train_accuracies, test_losses, test_accuracies, model_name, filename='training_results.pth'):
    # Save the train and test losses and accuracies for each epoch
    torch.save({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, "output/" + model_name + "_" + filename)

def save_model_checkpoint(model, optimizer, epoch, train_losses, train_accuracies, test_losses, test_accuracies, model_name, filename='model_checkpoint.pth'):
    """
    Save complete model checkpoint including model weights, optimizer state, and training history
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_losses: List of test losses
        test_accuracies: List of test accuracies
        model_name: Name of the model
        filename: Filename for the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'final_train_acc': train_accuracies[-1] if train_accuracies else 0,
        'final_test_acc': test_accuracies[-1] if test_accuracies else 0,
    }
    
    save_path = "output/" + model_name + "_" + filename
    torch.save(checkpoint, save_path)
    print(f"✅ Model checkpoint saved to {save_path}")
    
def save_model_for_inference(model, model_name, filename='model_best.pth'):
    """
    Save only the model state dict for inference/deployment
    This creates a lighter weight file suitable for deployment
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        filename: Filename for the model weights
    """
    save_path = "output/" + model_name + "_" + filename
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path} (for deployment)")

def save_plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, model_name, filename='training_plots.png'):
    # Plot training and test losses
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)
    plt.savefig("output/" + model_name + "_" + filename)
    plt.close()

def get_cifar100_transforms():
    """
    Convenience function to get both train and test transforms with computed mean/std
    Returns: (train_transform, test_transform, mean, std)
    """
    from model import get_train_transform, get_test_transform
    
    # Compute dynamic mean and std
    mean, std = compute_mean_std()
    
    # Get transforms with computed values
    train_transform = get_train_transform(mean, std)
    test_transform = get_test_transform(mean, std)
    
    return train_transform, test_transform, mean, std

def find_lr(model, device, trainloader, optimizer, criterion, start_lr=1e-7, end_lr=1, num_iter=100, smooth_factor=0.05):
    """
    Learning Rate Finder using torch-lr-finder library
    
    Args:
        model: PyTorch model
        device: Device to run on (cuda/cpu)
        trainloader: Training data loader
        optimizer: Optimizer (with any initial lr)
        criterion: Loss function
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 1)
        num_iter: Number of iterations to run (default: 100)
        smooth_factor: Smoothing factor for loss (default: 0.05)
    
    Returns:
        lrs: List of learning rates tried
        losses: List of corresponding losses
        best_lr: Suggested best learning rate
    """
    try:
        from torch_lr_finder import LRFinder
        
        print(f"\nRunning LR Finder from {start_lr} to {end_lr} over {num_iter} iterations...")
        
        # Create LR Finder
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        
        # Run range test
        lr_finder.range_test(trainloader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
        
        # Get the results
        lrs = lr_finder.history['lr']
        losses = lr_finder.history['loss']
        
        # Get suggested learning rate (steepest gradient)
        try:
            best_lr = lr_finder.suggestion()
            print(f"\nLR Finder completed!")
            print(f"Suggested LR: {best_lr:.2e}")
        except:
            # Fallback: use lr at minimum loss divided by 10
            min_loss_idx = losses.index(min(losses))
            best_lr = lrs[min_loss_idx] / 10
            print(f"\nLR Finder completed!")
            print(f"Suggested LR (fallback method): {best_lr:.2e}")
        
        # Reset the model and optimizer to initial state
        lr_finder.reset()
        
        return lrs, losses, best_lr
        
    except ImportError:
        print("\n⚠️  torch-lr-finder not installed. Installing now...")
        print("Please run: pip install torch-lr-finder")
        print("Then restart the training script.\n")
        raise ImportError("torch-lr-finder library is required. Install with: pip install torch-lr-finder")

def plot_lr_finder(lrs, losses, best_lr=None, save_path=None):
    """
    Plot the learning rate finder results
    
    Args:
        lrs: List of learning rates
        losses: List of losses
        best_lr: Best learning rate to highlight (optional)
        save_path: If provided, saves the plot to this path instead of showing it
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)
    
    if best_lr is not None:
        plt.axvline(x=best_lr, color='r', linestyle='--', label=f'Suggested LR: {best_lr:.2e}')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"LR Finder plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def save_lr_finder_plot(lrs, losses, best_lr=None, filename='lr_finder.png'):
    """
    Save the learning rate finder plot
    
    Args:
        lrs: List of learning rates
        losses: List of losses
        best_lr: Best learning rate to highlight (optional)
        filename: Filename to save the plot
    """
    save_path = "output/" + filename
    plot_lr_finder(lrs, losses, best_lr, save_path=save_path)
