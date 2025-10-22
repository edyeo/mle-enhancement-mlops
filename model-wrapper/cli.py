#!/usr/bin/env python3
"""
Command Line Interface for MLOps Model Wrapper.
"""
import argparse
import logging
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import ConfigManager
from src.models.pytorch_wrapper import PytorchLightningWrapper
from src.data_loaders.local_loader import LocalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(config_path: str, output_path: str) -> Dict[str, Any]:
    """Train a model using the provided configuration."""
    try:
        logger.info("Starting model training")
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config_manager.merge_env_vars()
        
        if not config_manager.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        config = config_manager.get_full_config()
        
        # Initialize data loader
        data_config = config_manager.get_data_config()
        data_loader = LocalDataLoader(data_config)
        
        # Load data
        data_loader.load_data()
        
        # Initialize model wrapper
        model_type = config_manager.get("model", {}).get("type", "pytorch")
        
        if model_type.lower() == "pytorch":
            model_wrapper = PytorchLightningWrapper(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        train_loader = data_loader.get_train_data()
        val_loader = data_loader.get_validation_data()
        
        # Combine train and val loaders for training
        # In a real implementation, you might want to handle this differently
        training_result = model_wrapper.train(train_loader)
        
        # Evaluate model
        test_loader = data_loader.get_test_data()
        evaluation_result = model_wrapper.evaluate(test_loader)
        
        # Save model
        model_path = model_wrapper.save_model(output_path)
        
        logger.info("Model training completed successfully")
        
        return {
            "status": "success",
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "model_path": model_path,
            "run_id": training_result.get("run_id")
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def evaluate_model(config_path: str, model_uri: str) -> Dict[str, Any]:
    """Evaluate a model using the provided configuration."""
    try:
        logger.info("Starting model evaluation")
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config_manager.merge_env_vars()
        
        config = config_manager.get_full_config()
        
        # Initialize data loader
        data_config = config_manager.get_data_config()
        data_loader = LocalDataLoader(data_config)
        
        # Load data
        data_loader.load_data()
        
        # Initialize model wrapper
        model_type = config_manager.get("model", {}).get("type", "pytorch")
        
        if model_type.lower() == "pytorch":
            model_wrapper = PytorchLightningWrapper(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model
        model_wrapper.load(model_uri)
        
        # Evaluate model
        test_loader = data_loader.get_test_data()
        evaluation_result = model_wrapper.evaluate(test_loader)
        
        logger.info("Model evaluation completed successfully")
        
        return {
            "status": "success",
            "evaluation_result": evaluation_result
        }
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def predict_model(config_path: str, model_uri: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction using the provided model."""
    try:
        logger.info("Starting model prediction")
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config_manager.merge_env_vars()
        
        config = config_manager.get_full_config()
        
        # Initialize model wrapper
        model_type = config_manager.get("model", {}).get("type", "pytorch")
        
        if model_type.lower() == "pytorch":
            model_wrapper = PytorchLightningWrapper(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model
        model_wrapper.load(model_uri)
        
        # Run prediction
        prediction_result = model_wrapper.predict(input_data)
        
        logger.info("Model prediction completed successfully")
        
        return {
            "status": "success",
            "prediction_result": prediction_result
        }
        
    except Exception as e:
        logger.error(f"Model prediction failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MLOps Model Wrapper CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Configuration file path")
    train_parser.add_argument("--output", required=True, help="Output directory path")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--config", required=True, help="Configuration file path")
    eval_parser.add_argument("--model-uri", required=True, help="Model URI")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--config", required=True, help="Configuration file path")
    predict_parser.add_argument("--model-uri", required=True, help="Model URI")
    predict_parser.add_argument("--input", required=True, help="Input data JSON string")
    
    args = parser.parse_args()
    
    if args.command == "train":
        result = train_model(args.config, args.output)
        print(f"Training result: {result}")
        
    elif args.command == "evaluate":
        result = evaluate_model(args.config, args.model_uri)
        print(f"Evaluation result: {result}")
        
    elif args.command == "predict":
        import json
        input_data = json.loads(args.input)
        result = predict_model(args.config, args.model_uri, input_data)
        print(f"Prediction result: {result}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
