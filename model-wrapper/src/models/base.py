"""
Base model wrapper interface for MLOps training pipeline.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers used in training pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        self.metrics = {}
        
    @abstractmethod
    def train(self, data_loader: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            data_loader: Data loader instance
            config: Additional training configuration
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            data_loader: Data loader instance
            config: Additional evaluation configuration
            
        Returns:
            Evaluation results dictionary
        """
        pass
    
    @abstractmethod
    def load(self, model_uri: str) -> None:
        """
        Load model from MLflow model URI.
        
        Args:
            model_uri: MLflow model URI
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
    
    @abstractmethod
    def save_model(self, output_path: str) -> str:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path to the saved model
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "metrics": self.metrics,
            "config": self.config
        }
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (to be implemented by subclasses)."""
        self.metrics.update(metrics)
        logger.info(f"Logged metrics: {metrics}")
