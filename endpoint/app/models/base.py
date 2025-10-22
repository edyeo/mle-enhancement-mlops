"""
Base model wrapper interface for MLOps Endpoint API.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self, model_uri: str) -> None:
        """
        Load model from MLflow model URI.
        
        Args:
            model_uri: MLflow model URI (e.g., "models:/my-model/Production")
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the loaded model.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Prediction results dictionary
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.is_loaded
