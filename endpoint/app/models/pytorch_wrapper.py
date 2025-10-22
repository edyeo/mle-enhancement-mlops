"""
PyTorch Lightning model wrapper for MLOps Endpoint API.
"""
import mlflow
import mlflow.pytorch
from typing import Dict, Any
import logging
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class PytorchLightningWrapper(ModelWrapper):
    """Wrapper for PyTorch Lightning models."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.is_loaded = False
    
    def load(self, model_uri: str) -> None:
        """
        Load PyTorch Lightning model from MLflow.
        
        Args:
            model_uri: MLflow model URI
        """
        try:
            logger.info(f"Loading PyTorch model from {model_uri}")
            
            # Load model using MLflow PyTorch flavor
            self.model = mlflow.pytorch.load_model(model_uri)
            self.is_loaded = True
            
            logger.info("PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the PyTorch model.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            logger.info("Running PyTorch model inference")
            
            # Extract input features from request
            # This is a simplified example - actual implementation depends on model structure
            features = input_data.get("features", [])
            
            # Convert to tensor if needed (simplified)
            import torch
            if isinstance(features, list):
                features = torch.tensor(features, dtype=torch.float32)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    # If model has predict method
                    prediction = self.model.predict(features)
                else:
                    # If model is a PyTorch Lightning model
                    prediction = self.model(features)
            
            # Convert prediction to serializable format
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy().tolist()
            
            result = {
                "prediction": prediction,
                "model_type": "pytorch",
                "status": "success"
            }
            
            logger.info("PyTorch model inference completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"PyTorch model inference failed: {str(e)}")
            return {
                "error": str(e),
                "model_type": "pytorch",
                "status": "error"
            }
