"""
Boosting classifiers for aneurysm detection.

Supports Gradient Boosting and XGBoost.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Tuple, Optional
import pickle

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class BoostingClassifier:
    """Wrapper for boosting classifiers."""
    
    def __init__(self, model_type: str = "gradient_boosting",
                 n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, random_state: int = 42):
        """Initialize boosting classifier.
        
        Args:
            model_type: Type of boosting ('gradient_boosting' or 'xgboost')
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=0
            )
        
        elif model_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
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
            Probability matrix (N, 2)
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
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
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scores = {
            "accuracy": cross_val_score(self.model, X, y, cv=skf, scoring="accuracy"),
            "precision": cross_val_score(self.model, X, y, cv=skf, scoring="precision"),
            "recall": cross_val_score(self.model, X, y, cv=skf, scoring="recall"),
            "f1": cross_val_score(self.model, X, y, cv=skf, scoring="f1"),
            "auc_roc": cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc"),
        }
        
        return {
            metric: {"mean": np.mean(scores), "std": np.std(scores)}
            for metric, scores in scores.items()
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise AttributeError(f"{self.model_type} does not have feature_importances_")
    
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load(filepath: str) -> 'BoostingClassifier':
        """Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            BoostingClassifier instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = BoostingClassifier.__new__(BoostingClassifier)
        classifier.model = model
        return classifier
