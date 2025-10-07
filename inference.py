"""
Inference module for ResNet-56 CIFAR-100 model
Handles model loading, preprocessing, and prediction
"""

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os

# Import model architecture from model.py
from model import CNN_Model

# CIFAR-100 class names (100 fine-grained classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# CIFAR-100 mean and std (computed from training set)
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


class CIFAR100Predictor:
    """Handles model loading and inference for CIFAR-100 classification"""
    
    def __init__(self, model_path='output/CNN_Model_model_best.pth', device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.classes = CIFAR100_CLASSES
        
    def _load_model(self, model_path):
        """Load the trained ResNet-56 model"""
        # Initialize model architecture
        model = CNN_Model(num_classes=100)
        
        # Load weights if file exists
        if os.path.exists(model_path):
            try:
                # Load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Handle different save formats
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    # Checkpoint format
                    model.load_state_dict(state_dict['model_state_dict'])
                    print(f"✅ Model loaded from checkpoint: {model_path}")
                    if 'final_test_acc' in state_dict:
                        print(f"   Model Test Accuracy: {state_dict['final_test_acc']:.2f}%")
                else:
                    # Direct state dict format
                    model.load_state_dict(state_dict)
                    print(f"✅ Model weights loaded from {model_path}")
                    
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                print("Using randomly initialized model (for testing only)")
        else:
            print(f"⚠️ Model file not found at {model_path}")
            print("Using randomly initialized model (for testing only)")
            print(f"Please train the model first and ensure weights are saved.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Get the preprocessing transform for inference"""
        return A.Compose([
            A.Resize(32, 32),  # Ensure image is 32x32
            A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
            ToTensorV2()
        ])
    
    def preprocess(self, image):
        """
        Preprocess an image for model input
        
        Args:
            image: PIL Image or numpy array (H, W, C) in RGB format
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed['image']
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor
    
    def predict(self, image, top_k=5):
        """
        Predict the class of an image
        
        Args:
            image: PIL Image or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing:
                - predictions: List of (class_name, probability) tuples
                - class_idx: Index of the predicted class
                - class_name: Name of the predicted class
                - confidence: Confidence score of the top prediction
        """
        # Preprocess image
        tensor = self.preprocess(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, 100))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predictions = [
            (self.classes[idx], float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        result = {
            'predictions': predictions,
            'class_idx': int(top_indices[0]),
            'class_name': self.classes[top_indices[0]],
            'confidence': float(top_probs[0])
        }
        
        return result
    
    def predict_batch(self, images, top_k=5):
        """
        Predict classes for a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, top_k=top_k)
            results.append(result)
        return results


def main():
    """Test the inference module"""
    print("🚀 Testing CIFAR-100 ResNet-56 Inference")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CIFAR100Predictor()
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {predictor.device}")
    print(f"   Model: ResNet-56")
    print(f"   Number of classes: {len(predictor.classes)}")
    print()
    
    # Create a dummy test image (32x32 RGB)
    print("📸 Creating dummy test image (32x32 RGB)...")
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Run prediction
    print("🔮 Running inference...")
    result = predictor.predict(test_image, top_k=5)
    
    print(f"\n✅ Top Prediction: {result['class_name']} ({result['confidence']:.2%})")
    print(f"\n📊 Top 5 Predictions:")
    for i, (class_name, prob) in enumerate(result['predictions'], 1):
        bar = '█' * int(prob * 50)
        print(f"   {i}. {class_name:20s} {prob:6.2%} {bar}")
    
    print("\n" + "=" * 60)
    print("✨ Inference module ready for deployment!")
    print("\nTo use in your code:")
    print("  from inference import CIFAR100Predictor")
    print("  predictor = CIFAR100Predictor()")
    print("  result = predictor.predict(your_image)")


if __name__ == "__main__":
    main()
