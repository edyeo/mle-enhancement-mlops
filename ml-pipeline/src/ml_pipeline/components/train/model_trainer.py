"""
Training component for Kubeflow pipeline.
Trains PyTorch Lightning models using the processed data.
"""
import os
import sys
import logging
import yaml
import json
import pickle
from typing import Dict, Any, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLflowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class LightningModel(pl.LightningModule):
    """PyTorch Lightning model wrapper."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.input_size = model_config.get("input_size", 10)
        self.hidden_size = model_config.get("hidden_size", 64)
        self.output_size = model_config.get("output_size", 1)
        self.dropout = model_config.get("dropout", 0.2)
        self.learning_rate = model_config.get("learning_rate", 0.001)
        
        # Create model
        self.model = SimpleNeuralNetwork(
            self.input_size, 
            self.hidden_size, 
            self.output_size, 
            self.dropout
        )
        
        # Loss function
        self.criterion = nn.MSELoss() if self.output_size == 1 else nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ModelTrainer:
    """Model trainer for PyTorch Lightning models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.mlflow_config = config.get("mlflow", {})
        
        # Training parameters
        self.max_epochs = self.training_config.get("max_epochs", 10)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 5)
        self.save_checkpoints = self.training_config.get("save_checkpoints", True)
        
        # MLflow setup
        self.tracking_uri = self.mlflow_config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = self.mlflow_config.get("experiment_name", "mlops-training")
        self.model_name = self.mlflow_config.get("model_name", "my-model")
        
        self.run_id = None
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created new experiment: {self.experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to setup experiment: {str(e)}")
                experiment_id = "0"  # Use default experiment
            
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {str(e)}")
            raise RuntimeError(f"MLflow setup failed: {str(e)}")
    
    def load_data(self, data_path: str) -> Dict[str, DataLoader]:
        """Load processed data and create data loaders."""
        try:
            logger.info(f"Loading data from {data_path}")
            
            # Load preprocessing artifacts
            artifacts_path = os.path.join(data_path, "preprocessing_artifacts.pkl")
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            self.feature_columns = artifacts["feature_columns"]
            self.target_column = artifacts["target_column"]
            
            # Load datasets
            train_df = pd.read_csv(os.path.join(data_path, "train_data.csv"))
            val_df = pd.read_csv(os.path.join(data_path, "validation_data.csv"))
            
            # Prepare features and targets
            X_train = train_df[self.feature_columns].values
            y_train = train_df[self.target_column].values
            X_val = val_df[self.feature_columns].values
            y_val = val_df[self.target_column].values
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            
            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
            
            logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
            
            return {
                "train": train_loader,
                "validation": val_loader
            }
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise RuntimeError(f"Data loading failed: {str(e)}")
    
    def train_model(self, data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Train the PyTorch Lightning model."""
        try:
            logger.info("Starting model training")
            
            # Setup MLflow
            self.setup_mlflow()
            
            # Start MLflow run
            run = mlflow.start_run(experiment_id=self.experiment_id)
            self.run_id = run.info.run_id
            
            # Create model
            lightning_model = LightningModel(self.model_config)
            
            # Setup MLflow logger
            mlflow_logger = MLflowLogger(
                experiment_name=self.experiment_name,
                tracking_uri=self.tracking_uri
            )
            
            # Setup callbacks
            callbacks = []
            
            if self.save_checkpoints:
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    filename='best-model-{epoch:02d}-{val_loss:.2f}'
                )
                callbacks.append(checkpoint_callback)
            
            if self.early_stopping_patience > 0:
                early_stopping_callback = EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    mode='min'
                )
                callbacks.append(early_stopping_callback)
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                logger=mlflow_logger,
                callbacks=callbacks,
                log_every_n_steps=10,
                accelerator='auto',
                devices='auto'
            )
            
            # Train model
            trainer.fit(lightning_model, data_loaders["train"], data_loaders["validation"])
            
            # Log parameters and metrics
            mlflow.log_params(self.model_config)
            mlflow.log_params(self.training_config)
            
            # Get best model path
            best_model_path = None
            if self.save_checkpoints and checkpoint_callback.best_model_path:
                best_model_path = checkpoint_callback.best_model_path
            
            logger.info("Model training completed successfully")
            
            return {
                "status": "success",
                "run_id": self.run_id,
                "model_type": "pytorch_lightning",
                "epochs_trained": trainer.current_epoch,
                "best_model_path": best_model_path,
                "model": lightning_model
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            if self.run_id:
                mlflow.end_run(status="FAILED")
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def save_model(self, model, output_path: str) -> str:
        """Save the trained model."""
        try:
            logger.info(f"Saving model to {output_path}")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Save model using MLflow PyTorch flavor
            model_path = os.path.join(output_path, "model")
            mlflow.pytorch.save_model(
                pytorch_model=model,
                path=model_path,
                conda_env={
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.9",
                        "pytorch",
                        "pytorch-lightning",
                        "mlflow"
                    ]
                }
            )
            
            # Save training metadata
            metadata = {
                "run_id": self.run_id,
                "model_config": self.model_config,
                "training_config": self.training_config,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "model_type": "pytorch_lightning"
            }
            
            metadata_path = os.path.join(output_path, "training_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")


def main():
    """Main function for training component."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model training component")
    parser.add_argument("--config", required=True, help="Configuration file path")
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
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Load data
        data_loaders = trainer.load_data(args.data_path)
        
        # Train model
        training_result = trainer.train_model(data_loaders)
        
        # Save model
        model_path = trainer.save_model(training_result["model"], args.output)
        
        print(f"Training completed successfully. Model saved to: {model_path}")
        print(f"MLflow run ID: {training_result['run_id']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
