import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class SpiralModel:
    def __init__(self, model_path="models/spiral_model.pth"):
        self.model_path = model_path
        # MPS (Apple Silicon) > CPU. Never CUDA.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Load a ResNet18 model modified for binary classification
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
        self.model = self.model.to(self.device)
        self.is_trained = False
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.is_trained = True
        else:
            print(f"Warning: Spiral model weights not found at {self.model_path}. Evaluation will be random.")
            
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Predicts if a spiral drawing indicates Parkinson's.
        Returns:
            label (str): 'Parkinson Detected' or 'Healthy'
            probability (float): confidence score for the prediction
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Apply Temperature Scaling (T=2.0) to prevent the probabilities from rounding to exactly 100.0%
            probabilities = torch.nn.functional.softmax(outputs / 2.0, dim=1)[0]
            
            prob_healthy = probabilities[0].item()
            prob_parkinson = probabilities[1].item()
            
            # Since model outputs might be random if untrained, we handle it
            if prob_parkinson > 0.5:
                return "Parkinson Detected", prob_parkinson
            else:
                return "Healthy", prob_healthy
