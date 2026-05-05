"""
CNN-based feature extraction from medical images.

Supports ResNet50, VGG16, and Inception models for feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional
import numpy as np


class CNNFeatureExtractor:
    """Extract features using pre-trained CNN models."""
    
    SUPPORTED_MODELS = ["resnet50", "vgg16", "inception"]
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True,
                 freeze_backbone: bool = True, device: str = "cpu"):
        """Initialize CNN feature extractor.
        
        Args:
            model_name: Name of the model ('resnet50', 'vgg16', 'inception')
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            device: Device to run on ('cpu' or 'cuda')
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Use: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.device = device
        self.model = self._build_model(model_name, pretrained, freeze_backbone)
        self.model.to(device)
        self.model.eval()
    
    def _build_model(self, model_name: str, pretrained: bool,
                     freeze_backbone: bool) -> nn.Module:
        """Build feature extraction model.
        
        Args:
            model_name: Model name
            pretrained: Use pretrained weights
            freeze_backbone: Freeze weights
            
        Returns:
            Feature extraction model
        """
        if model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            # Remove classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            # Use features only
            model = model.features
            # Add adaptive pooling
            model = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)))
            feature_dim = 512
        
        elif model_name == "inception":
            model = models.inception_v3(pretrained=pretrained)
            # Remove classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Feature vector
        """
        # Convert to tensor
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features.flatten()
    
    def extract_batch_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Feature matrix (N, D)
        """
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)
        
        return np.array(features_list)


class EnsembleCNNExtractor:
    """Combine features from multiple CNN models."""
    
    def __init__(self, model_names: List[str] = None, device: str = "cpu"):
        """Initialize ensemble feature extractor.
        
        Args:
            model_names: List of model names to use
            device: Device to run on
        """
        if model_names is None:
            model_names = ["resnet50", "vgg16"]
        
        self.extractors = [
            CNNFeatureExtractor(name, pretrained=True, freeze_backbone=True, device=device)
            for name in model_names
        ]
        self.model_names = model_names
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract and concatenate features from all models.
        
        Args:
            image: Input image
            
        Returns:
            Concatenated feature vector
        """
        all_features = []
        for extractor in self.extractors:
            features = extractor.extract_features(image)
            all_features.append(features)
        
        return np.concatenate(all_features)
    
    def extract_batch_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Feature matrix (N, D)
        """
        all_features = []
        for extractor in self.extractors:
            batch_features = extractor.extract_batch_features(images)
            all_features.append(batch_features)
        
        # Concatenate along feature dimension
        return np.concatenate(all_features, axis=1)
