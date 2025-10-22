"""
Model service for managing model loading and inference.
"""
import mlflow
import logging
from typing import Dict, Any, Optional
from ..config import Config
from ..models.base import ModelWrapper
from ..models.pytorch_wrapper import PytorchLightningWrapper
from ..models.llm_wrapper import LangGraphWrapper

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing model loading and inference."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_wrapper: Optional[ModelWrapper] = None
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking URI."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.config.mlflow_tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {str(e)}")
            raise RuntimeError(f"MLflow setup failed: {str(e)}")
    
    def load_model(self) -> ModelWrapper:
        """
        Load model based on configuration.
        
        Returns:
            Loaded model wrapper
        """
        try:
            # Create appropriate model wrapper based on model type
            if self.config.model_type.lower() == "pytorch":
                self.model_wrapper = PytorchLightningWrapper()
            elif self.config.model_type.lower() == "llm":
                self.model_wrapper = LangGraphWrapper()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Get model URI and load model
            model_uri = self.config.get_model_uri()
            logger.info(f"Loading model from URI: {model_uri}")
            
            self.model_wrapper.load(model_uri)
            
            logger.info(f"Model loaded successfully: {self.config.mlflow_model_name}")
            return self.model_wrapper
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def get_model(self) -> ModelWrapper:
        """
        Get the loaded model wrapper.
        
        Returns:
            Model wrapper instance
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model_wrapper is None or not self.model_wrapper.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.model_wrapper
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the loaded model.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            Prediction results
        """
        model = self.get_model()
        return model.predict(input_data)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_wrapper is not None and self.model_wrapper.is_model_loaded()
