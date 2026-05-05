"""
Hybrid pipeline combining CNN feature extraction with boosting classification.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.preprocessing.preprocessing import ImagePreprocessor
from src.preprocessing.slice_selection import get_slice_selector
from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.boosting import BoostingClassifier
from src.utils.dicom_loader import DICOMLoader


class HybridPipeline:
    """Hybrid pipeline: CNN features + Boosting classifier."""
    
    def __init__(self, cnn_model: str = "resnet50",
                 classifier: str = "gradient_boosting",
                 device: str = "cpu",
                 random_state: int = 42):
        """Initialize hybrid pipeline.
        
        Args:
            cnn_model: CNN model name
            classifier: Classifier type
            device: Device for CNN
            random_state: Random seed
        """
        self.cnn_model = cnn_model
        self.classifier_type = classifier
        self.device = device
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = CNNFeatureExtractor(cnn_model, device=device)
        self.classifier = BoostingClassifier(classifier, random_state=random_state)
        self.scaler = StandardScaler()
        
        self.is_trained = False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return self.preprocessor.preprocess(image)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract CNN features from image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector
        """
        return self.feature_extractor.extract_features(image)
    
    def train(self, X: List[np.ndarray], y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, float]:
        """Train the pipeline.
        
        Args:
            X: List of images
            y: Labels
            validation_split: Validation set ratio
            
        Returns:
            Training metrics
        """
        # Preprocess images
        print("Preprocessing images...")
        X_preprocessed = [self.preprocess_image(img) for img in X]
        
        # Extract features
        print("Extracting CNN features...")
        X_features = [self.extract_features(img) for img in X_preprocessed]
        X_features = np.array(X_features)
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.train(X_scaled, y)
        
        # Evaluate
        metrics = self.classifier.evaluate(X_scaled, y)
        self.is_trained = True
        
        return metrics
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make prediction for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Prediction dictionary with 'aneurysm' (bool) and 'confidence'
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        
        # Preprocess
        img_preprocessed = self.preprocess_image(image)
        
        # Extract features
        features = self.extract_features(img_preprocessed)
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = self.classifier.predict_proba(features_scaled)[0][int(prediction)]
        
        return {
            "aneurysm": bool(prediction),
            "confidence": float(confidence),
        }
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Make predictions for multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def cross_validate(self, X: List[np.ndarray], y: np.ndarray,
                      cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation.
        
        Args:
            X: List of images
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation scores
        """
        # Extract features for all images
        print("Extracting features for cross-validation...")
        X_preprocessed = [self.preprocess_image(img) for img in X]
        X_features = np.array([self.extract_features(img) for img in X_preprocessed])
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Cross-validate
        return self.classifier.cross_validate(X_scaled, y, cv_folds)
