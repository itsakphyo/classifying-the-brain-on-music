"""
Model training and evaluation utilities.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Import types for static analysis only
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb
    import lightgbm as lgb

# Models will be imported when available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """Model training and evaluation utilities."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_available_models(self) -> Dict[str, Any]:
        """Get dictionary of available models."""
        models = {}
        
        if SKLEARN_AVAILABLE:
            models.update({
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            })
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        return models
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Dict]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with model comparison results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model comparison")
        
        self.models = self.get_available_models()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for cross-validation")
                
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            self.results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
            
            print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        self.best_model_name = max(self.results.keys(), 
                                 key=lambda x: self.results[x]['mean_score'])
        print(f"\nBest model: {self.best_model_name}")
        
        return self.results
    
    def train_best_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train the best model on the full dataset."""
        if self.best_model_name is None:
            raise ValueError("No best model found. Run compare_models first.")
        
        self.best_model = self.models[self.best_model_name]
        self.best_model.fit(X, y)
        
        return self.best_model
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model evaluation")
            
        predictions = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1': f1_score(y, predictions, average='weighted')
        }
        
        return metrics
    
    def save_model(self, model: Any, filepath: str, 
                   metadata: Optional[Dict] = None) -> None:
        """Save model and metadata."""
        # Save model
        joblib.dump(model, filepath)
        
        # Save metadata
        if metadata:
            metadata_path = Path(filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> Tuple[Any, Optional[Dict]]:
        """Load model and metadata."""
        # Load model
        model = joblib.load(filepath)
        
        # Load metadata if exists
        metadata_path = Path(filepath).with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata


class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     target_names: Optional[List[str]] = None) -> str:
        """Generate detailed classification report."""
        if not SKLEARN_AVAILABLE:
            return "scikit-learn not available for classification report"
        
        return str(classification_report(y_true, y_pred, target_names=target_names))
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        if not SKLEARN_AVAILABLE:
            return np.array([])
        
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None


class PredictionGenerator:
    """Generate and manage predictions."""
    
    def __init__(self, model: Any, preprocessors: Dict[str, Any]):
        self.model = model
        self.preprocessors = preprocessors
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with preprocessing."""
        # Apply preprocessing pipeline
        X_processed = X.copy()
        
        # Scale features if scaler available
        if 'scaler' in self.preprocessors:
            X_processed = self.preprocessors['scaler'].transform(X_processed)
        
        # Select features if selector available
        if 'feature_selector' in self.preprocessors:
            X_processed = self.preprocessors['feature_selector'].transform(X_processed)
        
        # Generate predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        return predictions, probabilities
    
    def create_submission(self, test_data: pd.DataFrame, 
                         output_path: str = 'predictions.csv') -> pd.DataFrame:
        """Create submission file."""
        predictions, probabilities = self.predict(test_data)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'sample_id': range(len(predictions)),
            'predicted_class': predictions
        })
        
        # Add probability columns
        for i in range(probabilities.shape[1]):
            submission[f'prob_class_{i}'] = probabilities[:, i]
        
        # Save to file
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        return submission


def create_model_report(model: Any, model_name: str, 
                       training_metrics: Dict[str, float],
                       validation_metrics: Optional[Dict[str, float]] = None,
                       feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Create comprehensive model report."""
    
    report = {
        'model_info': {
            'name': model_name,
            'type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else {},
            'training_date': datetime.now().isoformat()
        },
        'performance': {
            'training': training_metrics
        }
    }
    
    if validation_metrics:
        report['performance']['validation'] = validation_metrics
    
    if feature_importance is not None:
        report['feature_importance'] = {
            'top_10_features': feature_importance.head(10).to_dict('records'),
            'total_features': len(feature_importance)
        }
    
    return report


def save_training_artifacts(model: Any, preprocessors: Dict[str, Any],
                          metadata: Dict[str, Any],
                          output_dir: str = 'models') -> Dict[str, str]:
    """Save all training artifacts."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = {}
    
    # Save model
    model_filename = f"model_{timestamp}.pkl"
    model_path = output_path / model_filename
    joblib.dump(model, model_path)
    saved_files['model'] = str(model_path)
    
    # Save preprocessors
    for name, preprocessor in preprocessors.items():
        filename = f"{name}_{timestamp}.pkl"
        filepath = output_path / filename
        joblib.dump(preprocessor, filepath)
        saved_files[name] = str(filepath)
    
    # Save metadata
    metadata_filename = f"metadata_{timestamp}.json"
    metadata_path = output_path / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files['metadata'] = str(metadata_path)
    
    return saved_files
