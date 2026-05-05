"""
Baseline machine learning models for comparison.

Supports Logistic Regression, KNN, and Random Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional
import pickle


class BaselineClassifier:
    """Baseline classifier wrapper."""
    
    SUPPORTED_MODELS = ["logistic_regression", "knn", "random_forest"]
    
    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        """Initialize baseline classifier.
        
        Args:
            model_type: Type of model
            **kwargs: Model-specific hyperparameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_type} not supported. Use: {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model = self._build_model(model_type, **kwargs)
    
    def _build_model(self, model_type: str, **kwargs):
        """Build baseline model.
        
        Args:
            model_type: Type of model
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        if model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42),
                solver="lbfgs",
                n_jobs=-1
            )
        
        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5),
                n_jobs=-1
            )
        
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier.
        
        Args:
            X: Feature matrix (N, D)
            y: Target labels (N,)
        """
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support predict_proba")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba),
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5) -> dict:
        """Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation scores
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = {
            "accuracy": cross_val_score(self.model, X, y, cv=skf, scoring="accuracy"),
            "precision": cross_val_score(self.model, X, y, cv=skf, scoring="precision"),
            "recall": cross_val_score(self.model, X, y, cv=skf, scoring="recall"),
            "f1": cross_val_score(self.model, X, y, cv=skf, scoring="f1"),
        }
        
        if hasattr(self.model, 'predict_proba'):
            scores["auc_roc"] = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc")
        
        return {
            metric: {"mean": np.mean(cv_scores), "std": np.std(cv_scores)}
            for metric, cv_scores in scores.items()
        }
    
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load(filepath: str) -> 'BaselineClassifier':
        """Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            BaselineClassifier instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = BaselineClassifier.__new__(BaselineClassifier)
        classifier.model = model
        return classifier
