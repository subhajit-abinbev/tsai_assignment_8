import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Optimizer configuration
def get_optimizer():
    # return optim.AdamW  # Return the actual optimizer class
    return optim.SGD

def get_optimizer_params():
    # return {'lr': 0.01, "weight_decay": 1e-4}
    # For SGD, typical LR is much higher than AdamW (0.1 vs 0.001)
    # LR Finder will determine the optimal value
    return {'lr': 0.15, 'momentum': 0.9, "weight_decay": 5e-4, 'nesterov': True}

# Data augmentation transform
def get_train_transform(mean, std):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.1,  # ±10% translation
            scale=(0.9, 1.1),       # ±10% scaling
            rotate=(-15, 15),       # ±15° rotation
            p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(1, 1),    # Exactly 1 hole (min=1, max=1)
            hole_height_range=(16, 16), # Exactly 16px height
            hole_width_range=(16, 16),  # Exactly 16px width
            p=0.5
        ),

        # 🔥 Color augmentation - CRITICAL for CIFAR-100
        A.ColorJitter(
            brightness=0.2,       # ±20% brightness
            contrast=0.2,         # ±20% contrast
            saturation=0.2,       # ±20% saturation
            hue=0.1,              # ±10% hue
            p=0.5
        ),

        # Normalization (always last before ToTensor)
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])

# Test transform without augmentation
def get_test_transform(mean, std):
    return A.Compose([
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])

# Scheduler configuration for model
def get_scheduler():
    return optim.lr_scheduler.MultiStepLR

def get_scheduler_params(steps_per_epoch=None, epochs=100):
    """
    MultiStepLR: Drops learning rate at specific epochs
    For 150 epochs: drop at 45, 90, 120 (30%, 60%, 80%)
    This is the proven approach from ResNet paper
    """
    milestones = [40, 70, 85]
    
    return {
        'milestones': milestones,
        'gamma': 0.2  # Multiply LR by 0.2 at each milestone
    }

class BasicBlock(nn.Module):
    """Basic ResNet block for CIFAR (two 3x3 convolutions)"""
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN_Model(nn.Module):
    """ResNet-56 for CIFAR-100 (9 blocks per stage, 3 stages, 2 layers per block = 54 + 2 = 56 layers)"""
    def __init__(self, num_classes=100):
        super(CNN_Model, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet stages - each stage has 9 blocks for ResNet-56
        self.layer1 = self._make_layer(16, 16, num_blocks=9, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=9, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=9, stride=2)
        
        # Global Average Pooling + Fully Connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global Average Pooling + FC
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out