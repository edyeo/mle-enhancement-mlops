"""
LangGraph/LLM model wrapper for MLOps Endpoint API.
"""
import mlflow
import mlflow.pyfunc
from typing import Dict, Any
import logging
import json
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class LangGraphWrapper(ModelWrapper):
    """Wrapper for LangGraph/LLM models."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.is_loaded = False
    
    def load(self, model_uri: str) -> None:
        """
        Load LangGraph/LLM model from MLflow.
        
        Args:
            model_uri: MLflow model URI
        """
        try:
            logger.info(f"Loading LLM model from {model_uri}")
            
            # Load model using MLflow pyfunc flavor
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.is_loaded = True
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the LLM model.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            logger.info("Running LLM model inference")
            
            # Extract input text/prompt from request
            prompt = input_data.get("prompt", "")
            if not prompt:
                prompt = input_data.get("text", "")
            
            if not prompt:
                raise ValueError("No prompt or text provided in input data")
            
            # Prepare input for MLflow pyfunc model
            model_input = {
                "prompt": prompt,
                "input_data": input_data
            }
            
            # Run inference using MLflow pyfunc
            prediction = self.model.predict(model_input)
            
            # Handle different prediction formats
            if isinstance(prediction, str):
                result = {
                    "response": prediction,
                    "model_type": "llm",
                    "status": "success"
                }
            elif isinstance(prediction, dict):
                result = {
                    "response": prediction.get("response", prediction),
                    "model_type": "llm",
                    "status": "success",
                    "metadata": prediction.get("metadata", {})
                }
            else:
                result = {
                    "response": str(prediction),
                    "model_type": "llm",
                    "status": "success"
                }
            
            logger.info("LLM model inference completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"LLM model inference failed: {str(e)}")
            return {
                "error": str(e),
                "model_type": "llm",
                "status": "error"
            }
