"""
Original pipeline with hand-crafted features and classical ML.
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import cv2

from src.preprocessing.preprocessing import ImagePreprocessor
from src.models.baseline import BaselineClassifier


class OriginalPipeline:
    """Pipeline with hand-crafted features and classical ML classifiers."""
    
    def __init__(self, classifier: str = "random_forest", random_state: int = 42):
        """Initialize original pipeline.
        
        Args:
            classifier: Classifier type ('random_forest', 'svm', etc.)
            random_state: Random seed
        """
        self.classifier_type = classifier
        self.random_state = random_state
        
        self.preprocessor = ImagePreprocessor()
        self.classifier = BaselineClassifier(classifier, random_state=random_state)
        self.scaler = StandardScaler()
        
        self.is_trained = False
    
    def extract_hand_crafted_features(self, image: np.ndarray) -> np.ndarray:
        """Extract hand-crafted features from image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Statistical features
        features.extend([
            np.mean(image),
            np.std(image),
            np.var(image),
            np.min(image),
            np.max(image),
            np.median(image),
        ])
        
        # Histogram features
        hist = np.histogram(image.flatten(), bins=16)[0]
        features.extend(hist)
        
        # Edge detection (Canny)
        edges = cv2.Canny((image / image.max() * 255).astype(np.uint8), 100, 200)
        features.extend([
            np.sum(edges > 0),
            np.mean(edges),
        ])
        
        # Texture features (GLCM-like)
        features.extend([
            np.sum(image > np.mean(image)),
            np.sum(image < np.mean(image)),
        ])
        
        return np.array(features)
    
    def extract_batch_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Feature matrix (N, D)
        """
        features_list = []
        for image in images:
            features = self.extract_hand_crafted_features(image)
            features_list.append(features)
        
        return np.array(features_list)
    
    def train(self, X: List[np.ndarray], y: np.ndarray) -> Dict[str, float]:
        """Train the pipeline.
        
        Args:
            X: List of images
            y: Labels
            
        Returns:
            Training metrics
        """
        # Preprocess images
        print("Preprocessing images...")
        X_preprocessed = [self.preprocessor.preprocess(img) for img in X]
        
        # Extract hand-crafted features
        print("Extracting hand-crafted features...")
        X_features = self.extract_batch_features(X_preprocessed)
        
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
            Prediction dictionary
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        
        # Preprocess
        img_preprocessed = self.preprocessor.preprocess(image)
        
        # Extract features
        features = self.extract_hand_crafted_features(img_preprocessed)
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        proba = self.classifier.predict_proba(features_scaled)[0]
        confidence = proba[int(prediction)]
        
        return {
            "aneurysm": bool(prediction),
            "confidence": float(confidence),
        }
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Make predictions for multiple images.
        
        Args:
            images: List of images
            
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
            cv_folds: CV folds
            
        Returns:
            CV scores
        """
        print("Extracting features for cross-validation...")
        X_preprocessed = [self.preprocessor.preprocess(img) for img in X]
        X_features = self.extract_batch_features(X_preprocessed)
        X_scaled = self.scaler.fit_transform(X_features)
        
        return self.classifier.cross_validate(X_scaled, y, cv_folds)
