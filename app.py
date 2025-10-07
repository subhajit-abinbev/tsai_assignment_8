"""
Gradio Web App for ResNet-56 CIFAR-100 Image Classification
Deploy on Hugging Face Spaces: https://huggingface.co/spaces
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
from inference import CIFAR100Predictor, CIFAR100_CLASSES

# Initialize the model predictor
print("🚀 Initializing ResNet-56 model...")
try:
    predictor = CIFAR100Predictor()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    predictor = None

# Custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 1200px;
    margin: auto;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    margin: 10px 0;
}
.confidence-high { color: #10b981; font-weight: bold; }
.confidence-medium { color: #f59e0b; font-weight: bold; }
.confidence-low { color: #ef4444; font-weight: bold; }
"""

def predict_image(image):
    """
    Predict the class of an uploaded image
    
    Args:
        image: PIL Image or numpy array from Gradio
        
    Returns:
        Tuple of (formatted_html, confidences_dict)
    """
    if image is None:
        return "<p style='text-align: center; color: #ef4444;'>⚠️ Please upload an image</p>", {}
    
    if predictor is None:
        return "<p style='text-align: center; color: #ef4444;'>❌ Model not loaded. Please check model file.</p>", {}
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Run prediction
        result = predictor.predict(image, top_k=10)
        
        # Format output
        top_class = result['class_name'].replace('_', ' ').title()
        confidence = result['confidence']
        
        # Confidence indicator
        if confidence > 0.6:
            conf_emoji = "🟢"
            conf_text = "High Confidence"
            conf_class = "confidence-high"
        elif confidence > 0.3:
            conf_emoji = "🟡"
            conf_text = "Medium Confidence"
            conf_class = "confidence-medium"
        else:
            conf_emoji = "🔴"
            conf_text = "Low Confidence"
            conf_class = "confidence-low"
        
        # Create formatted HTML output
        html_output = f"""
        <div class='prediction-box'>
            <h2 style='margin: 0; font-size: 2em;'>🎯 Prediction</h2>
            <h1 style='margin: 10px 0; font-size: 2.5em;'>{top_class}</h1>
            <p class='{conf_class}' style='font-size: 1.3em; margin: 5px 0;'>
                {conf_emoji} {conf_text}
            </p>
            <p style='font-size: 2em; font-weight: bold; margin: 10px 0;'>{confidence:.1%}</p>
        </div>
        """
        
        # Format confidence scores for the label component
        confidences = {
            pred[0].replace('_', ' ').title(): pred[1] 
            for pred in result['predictions']
        }
        
        return html_output, confidences
        
    except Exception as e:
        error_msg = f"<p style='text-align: center; color: #ef4444;'>❌ Error: {str(e)}</p>"
        print(f"Prediction error: {e}")
        return error_msg, {}


# Build Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🖼️ CIFAR-100 Image Classifier
        ### ResNet-56 Deep Learning Model
        
        Upload an image or use your camera to classify it into one of **100 categories**!
        This model was trained on the CIFAR-100 dataset with **73.68% test accuracy**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(
                label="Input Image",
                type="pil",
                height=300,
                sources=["upload", "webcam", "clipboard"]
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔮 Classify", variant="primary", size="lg", scale=2)
                clear_btn = gr.ClearButton([image_input], value="🗑️ Clear", size="lg", scale=1)
            
            gr.Markdown(
                """
                ### 💡 Tips:
                - **Best results**: Clear, centered images
                - **Image size**: Automatically resized to 32×32
                - **Categories**: Animals, vehicles, plants, household items, scenes
                - **Try it**: Upload any image or use your webcam!
                """
            )
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 🎯 Results")
            output_html = gr.HTML(label="Prediction")
            output_confidences = gr.Label(
                label="Top 10 Predictions (with confidence scores)",
                num_top_classes=10
            )
    
    # Model information accordion
    with gr.Accordion("📊 Model Details", open=False):
        gr.Markdown(
            f"""
            ### Architecture: ResNet-56
            - **Depth**: 56 convolutional layers with residual connections
            - **Parameters**: 861,620 trainable parameters
            - **Input**: 32×32 RGB images
            - **Output**: 100 class probabilities
            
            ### Training Details
            - **Dataset**: CIFAR-100 (50,000 train, 10,000 test images)
            - **Optimizer**: SGD with momentum (lr=0.1, momentum=0.9, weight_decay=5e-4)
            - **Scheduler**: MultiStepLR (drops at epochs 40, 70, 85)
            - **Augmentation**: HorizontalFlip, Affine, CoarseDropout, ColorJitter
            - **Training Duration**: 100 epochs
            
            ### Performance
            - **Training Accuracy**: 79.97%
            - **Test Accuracy**: 73.68%
            - **Device**: {predictor.device if predictor else 'N/A'}
            """
        )
    
    # Categories accordion
    with gr.Accordion("📋 All 100 Categories", open=False):
        # Group classes by category
        animals_fish = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
        animals_mammals = ['bear', 'beaver', 'cattle', 'chimpanzee', 'dolphin', 'elephant', 
                          'fox', 'hamster', 'kangaroo', 'leopard', 'lion', 'mouse', 'otter', 
                          'porcupine', 'possum', 'rabbit', 'raccoon', 'seal', 'shrew', 'skunk', 
                          'squirrel', 'tiger', 'whale', 'wolf']
        animals_insects = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'spider']
        animals_reptiles = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
        animals_other = ['crab', 'lobster', 'snail', 'worm']
        
        people = ['baby', 'boy', 'girl', 'man', 'woman']
        
        vehicles = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'rocket', 
                   'streetcar', 'tank', 'tractor', 'train', 'lawn_mower']
        
        trees = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
        flowers = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
        food = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
        
        furniture = ['bed', 'chair', 'couch', 'table', 'wardrobe']
        household = ['bottle', 'bowl', 'can', 'clock', 'cup', 'keyboard', 'lamp', 
                    'plate', 'telephone', 'television']
        
        nature = ['cloud', 'forest', 'mountain', 'plain', 'road', 'sea']
        structures = ['bridge', 'castle', 'house', 'skyscraper']
        
        gr.Markdown("#### 🐠 Aquatic Animals")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in animals_fish]))
        
        gr.Markdown("#### 🦁 Mammals")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in animals_mammals]))
        
        gr.Markdown("#### 🦋 Insects & Arachnids")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in animals_insects]))
        
        gr.Markdown("#### 🦎 Reptiles & Dinosaurs")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in animals_reptiles]))
        
        gr.Markdown("#### 🦀 Other Animals")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in animals_other]))
        
        gr.Markdown("#### 👥 People")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in people]))
        
        gr.Markdown("#### 🚗 Vehicles & Machines")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in vehicles]))
        
        gr.Markdown("#### 🌳 Trees")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in trees]))
        
        gr.Markdown("#### 🌸 Flowers")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in flowers]))
        
        gr.Markdown("#### 🍎 Food")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in food]))
        
        gr.Markdown("#### 🪑 Furniture")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in furniture]))
        
        gr.Markdown("#### 🏠 Household Items")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in household]))
        
        gr.Markdown("#### 🏔️ Natural Scenes")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in nature]))
        
        gr.Markdown("#### 🏰 Structures")
        gr.Markdown(", ".join([c.replace('_', ' ').title() for c in structures]))
        
        # Camel is missing from groups - add it
        gr.Markdown("#### 🐪 Other")
        gr.Markdown("Camel")
    
    # About section
    with gr.Accordion("ℹ️ About This Project", open=False):
        gr.Markdown(
            """
            ### About
            
            This is a deep learning image classification model built with **PyTorch** and deployed 
            using **Gradio** on Hugging Face Spaces. The model uses the **ResNet-56** architecture,
            which consists of 56 layers with residual (skip) connections that allow training very 
            deep networks effectively.
            
            ### CIFAR-100 Dataset
            - **100 fine-grained classes** organized into 20 superclasses
            - **60,000 images** (50,000 training + 10,000 test)
            - **Image size**: 32×32 pixels, RGB color
            - **Classes include**: Animals, vehicles, plants, household items, natural scenes, and more
            
            ### Technologies Used
            - **PyTorch**: Deep learning framework
            - **Albumentations**: Advanced image augmentation
            - **Gradio**: Interactive web interface
            - **Hugging Face Spaces**: Free deployment platform
            
            ### Model Architecture
            ResNet-56 implements deep residual learning with:
            - Initial 3×3 convolution layer
            - 3 stages with 9 residual blocks each (27 blocks total)
            - Each block has 2 convolutional layers (54 layers)
            - Global average pooling + fully connected layer (2 more layers)
            - **Total**: 56 layers deep!
            
            ### Training Techniques
            - **Data Augmentation**: Random flips, rotations, color jitter, cutout
            - **Regularization**: Weight decay (L2 penalty)
            - **Learning Rate Schedule**: Step decay at epochs 40, 70, 85
            - **Batch Normalization**: After every convolution
            - **Residual Connections**: Enable very deep training
            
            ---
            
            **Made with ❤️ for learning and education**
            
            *Note: This model works best on images similar to CIFAR-100 training data. 
            For best results, use clear images with a single centered object.*
            """
        )
    
    # Event handlers
    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[output_html, output_confidences]
    )
    
    # Auto-predict on image upload
    image_input.change(
        fn=predict_image,
        inputs=image_input,
        outputs=[output_html, output_confidences]
    )

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",     # Required for Hugging Face Spaces
        server_port=7860,           # Default Gradio port (HF Spaces uses this)
        share=False,                # Don't create public link (HF manages this)
        show_error=True,            # Show detailed errors for debugging
        favicon_path=None           # Can add custom favicon if desired
    )
