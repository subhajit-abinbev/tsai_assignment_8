import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary
import os

from model import CNN_Model, get_optimizer, get_optimizer_params, get_train_transform, get_scheduler, get_scheduler_params

from utils import compute_mean_std, data_loader, initialize_model, save_plot_metrics, save_train_test_metrics, train_model, find_lr, plot_lr_finder, save_lr_finder_plot

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

print("Starting training script...")

selected_model_class = CNN_Model
get_optimizer_func = get_optimizer
get_optimizer_params_func = get_optimizer_params
get_train_transform_func = get_train_transform
get_scheduler_func = get_scheduler
get_scheduler_params_func = get_scheduler_params
print("="*50)

# Check if CUDA is available
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# Compute mean and std of training data
print("Computing mean and std of training data...")
train_mean, train_std = compute_mean_std()
print("Computation done.")
print(f"Mean: {train_mean}, Std: {train_std}")

# Data loaders
print("Preparing data loaders...")
train_transform = get_train_transform_func(train_mean, train_std)
train_loader, test_loader = data_loader(train_mean=train_mean, train_std=train_std, batch_size_train=128, batch_size_test=5000, train_transform=train_transform)
print("Data loaders ready.")

# Initialize model, loss function, optimizer
print("Initializing model...")
optimizer_func = get_optimizer_func()
optimizer_params = get_optimizer_params_func()
model, criterion, optimizer, device = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
print("Model initialized successfully.")

# Print model summary
print("Model Summary:")
summary(model, input_size=(3, 32, 32)) # Since input size for CIFAR-100 is 32x32 with 3 channels

# Learning Rate Finder
use_lr_finder = input("\nDo you want to run LR Finder? (y/n, default: y): ").lower() or 'y'

if use_lr_finder == 'y':
    print("\n" + "="*50)
    print("RUNNING LEARNING RATE FINDER")
    print("="*50)
    
    # Create a fresh model and optimizer for LR finding
    # Note: LR finder will override the optimizer's LR and test from start_lr to end_lr
    temp_model, temp_criterion, temp_optimizer, _ = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
    
    # Run LR finder
    lrs, losses, suggested_lr = find_lr(
        temp_model, 
        device, 
        train_loader, 
        temp_optimizer, 
        temp_criterion,
        start_lr=1e-4,  # Start higher for SGD (1e-4 instead of 1e-7)
        end_lr=1,  # Search up to 1 for SGD (typical range: 0.01-1.0)
        num_iter=200  # More iterations for better curve
    )
    
    # Display the plot interactively
    plot_lr_finder(lrs, losses, suggested_lr)
    
    # Save the plot to file
    save_lr_finder_plot(lrs, losses, suggested_lr, filename=f'{selected_model_class.__name__}_lr_finder.png')
    
    # Ask user if they want to use the suggested LR
    use_suggested = input(f"\nUse suggested LR {suggested_lr:.2e}? (y/n, default: y): ").lower() or 'y'
    
    if use_suggested == 'y':
        optimizer_params['lr'] = suggested_lr
        print(f"Using learning rate: {suggested_lr:.2e}")
        
        # Reinitialize model with new learning rate
        model, criterion, optimizer, device = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
        print("Model reinitialized with new learning rate.")
    else:
        custom_lr = float(input("Enter custom learning rate: "))
        optimizer_params['lr'] = custom_lr
        print(f"Using custom learning rate: {custom_lr:.2e}")
        
        # Reinitialize model with custom learning rate
        model, criterion, optimizer, device = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
        print("Model reinitialized with custom learning rate.")
    
    print("="*50)
    print()

# Model Training
epochs = int(input("Select number of epochs for training (default 20): ") or 20)
print(f"\nStarting training for {selected_model_class.__name__}...")

# Initialize scheduler for model training
scheduler = None
scheduler_func = get_scheduler_func()
scheduler_params = get_scheduler_params_func(len(train_loader), epochs=epochs)

# For OneCycleLR, override max_lr with the found LR value
if scheduler_func == optim.lr_scheduler.OneCycleLR:
    # Use the LR from optimizer (which was set by LR finder) as max_lr
    scheduler_params['max_lr'] = optimizer.param_groups[0]['lr']
    print(f"OneCycleLR configured: max_lr={scheduler_params['max_lr']:.4f}, initial_lr={scheduler_params['max_lr']/scheduler_params.get('div_factor', 10):.4f}")

scheduler = scheduler_func(optimizer, **scheduler_params)
print("Scheduler initialized for model training.")

# Actual model training
train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, device, train_loader, test_loader, optimizer, 
                                                                           criterion, epochs=epochs, scheduler=scheduler)
print("Training completed!")

# Save the train and test losses and accuracies for future reference
print("Saving training and testing metrics...")
save_train_test_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_class.__name__)
print("Metrics saved successfully.")

# Save plot metrics for visualization
print("Saving plot metrics...")
save_plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_class.__name__)
print("Plot metrics saved successfully.")

# Print final statistics
print(f"\nFinal Results:")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Test Loss: {test_losses[-1]:.4f}")
print(f"Test Accuracy: {test_accuracies[-1]:.2f}%")
