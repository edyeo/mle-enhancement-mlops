"""
Evaluation component for Kubeflow pipeline.
Evaluates trained models and logs metrics to MLflow.
"""
import os
import logging
import yaml
import json
import pickle
from typing import Dict, Any, Optional
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluator for trained PyTorch Lightning models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_config = config.get("mlflow", {})
        self.evaluation_config = config.get("evaluation", {})
        
        # MLflow setup
        self.tracking_uri = self.mlflow_config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = self.mlflow_config.get("experiment_name", "mlops-training")
        self.model_name = self.mlflow_config.get("model_name", "my-model")
        
        # Evaluation parameters
        self.metrics = self.evaluation_config.get("metrics", ["mse", "mae", "r2"])
        self.threshold = self.evaluation_config.get("threshold", 0.5)
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {str(e)}")
            raise RuntimeError(f"MLflow setup failed: {str(e)}")
    
    def load_model(self, model_path: str):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model using MLflow PyTorch flavor
            model = mlflow.pytorch.load_model(model_path)
            
            # Load training metadata
            metadata_path = os.path.join(os.path.dirname(model_path), "training_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.run_id = metadata["run_id"]
            self.feature_columns = metadata["feature_columns"]
            self.target_column = metadata["target_column"]
            self.model_config = metadata["model_config"]
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def load_test_data(self, data_path: str) -> DataLoader:
        """Load test data and create data loader."""
        try:
            logger.info(f"Loading test data from {data_path}")
            
            # Load preprocessing artifacts
            artifacts_path = os.path.join(data_path, "preprocessing_artifacts.pkl")
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Load test dataset
            test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))
            
            # Prepare features and targets
            X_test = test_df[self.feature_columns].values
            y_test = test_df[self.target_column].values
            
            # Convert to tensors
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            # Create dataset and data loader
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=2
            )
            
            logger.info(f"Created test data loader with {len(test_loader)} batches")
            return test_loader
            
        except Exception as e:
            logger.error(f"Failed to load test data: {str(e)}")
            raise RuntimeError(f"Test data loading failed: {str(e)}")
    
    def evaluate_model(self, model, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model and compute metrics."""
        try:
            logger.info("Starting model evaluation")
            
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # Get predictions
                    predictions = model(batch_x)
                    
                    # Convert to numpy arrays
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.cpu().numpy()
                    if isinstance(batch_y, torch.Tensor):
                        batch_y = batch_y.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_targets.append(batch_y)
            
            # Concatenate all predictions and targets
            y_pred = np.concatenate(all_predictions, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            
            # Flatten arrays if needed
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            if y_true.ndim > 1:
                y_true = y_true.flatten()
            
            # Compute metrics based on problem type
            metrics = {}
            
            # Determine problem type based on output size
            output_size = self.model_config.get("output_size", 1)
            
            if output_size == 1:
                # Regression problem
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["mae"] = mean_absolute_error(y_true, y_pred)
                metrics["r2"] = r2_score(y_true, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                
            else:
                # Classification problem
                # Convert predictions to class labels
                if y_pred.ndim > 1:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    y_pred_classes = (y_pred > self.threshold).astype(int)
                
                metrics["accuracy"] = accuracy_score(y_true, y_pred_classes)
                metrics["precision"] = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
                metrics["f1"] = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
            
            logger.info(f"Evaluation completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, float], run_id: str):
        """Log evaluation metrics to MLflow."""
        try:
            logger.info(f"Logging metrics to MLflow run: {run_id}")
            
            # Set the active run
            with mlflow.start_run(run_id=run_id):
                # Log evaluation metrics
                mlflow.log_metrics(metrics)
                
                # Log evaluation metadata
                evaluation_metadata = {
                    "evaluation_config": self.evaluation_config,
                    "metrics_computed": list(metrics.keys()),
                    "test_samples": len(metrics)  # This would be better calculated from actual data
                }
                
                mlflow.log_params(evaluation_metadata)
                
            logger.info("Metrics logged to MLflow successfully")
            
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {str(e)}")
            raise RuntimeError(f"MLflow logging failed: {str(e)}")
    
    def register_model(self, model_path: str, metrics: Dict[str, float]) -> str:
        """Register model in MLflow Model Registry."""
        try:
            logger.info("Registering model in MLflow Model Registry")
            
            # Create model URI
            model_uri = f"runs:/{self.run_id}/model"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name,
                description=f"Model trained with metrics: {metrics}"
            )
            
            logger.info(f"Model registered: {self.model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise RuntimeError(f"Model registration failed: {str(e)}")
    
    def save_evaluation_results(self, metrics: Dict[str, float], model_version: str, output_path: str) -> str:
        """Save evaluation results to output path."""
        try:
            logger.info(f"Saving evaluation results to {output_path}")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Prepare evaluation results
            evaluation_results = {
                "run_id": self.run_id,
                "model_name": self.model_name,
                "model_version": model_version,
                "metrics": metrics,
                "evaluation_config": self.evaluation_config,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "model_config": self.model_config
            }
            
            # Save results
            results_path = os.path.join(output_path, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}")
            raise RuntimeError(f"Results saving failed: {str(e)}")


def main():
    """Main function for evaluation component."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model evaluation component")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--model-path", required=True, help="Trained model directory path")
    parser.add_argument("--data-path", required=True, help="Processed data directory path")
    parser.add_argument("--output", required=True, help="Output directory path")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Setup MLflow
        evaluator.setup_mlflow()
        
        # Load model
        model = evaluator.load_model(args.model_path)
        
        # Load test data
        test_loader = evaluator.load_test_data(args.data_path)
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, test_loader)
        
        # Log metrics to MLflow
        evaluator.log_metrics_to_mlflow(metrics, evaluator.run_id)
        
        # Register model
        model_version = evaluator.register_model(args.model_path, metrics)
        
        # Save evaluation results
        results_path = evaluator.save_evaluation_results(metrics, model_version, args.output)
        
        print(f"Evaluation completed successfully.")
        print(f"Model version: {model_version}")
        print(f"Metrics: {metrics}")
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
