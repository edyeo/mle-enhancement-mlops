"""
PyTorch Lightning model wrapper for MLOps training pipeline.
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLflowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional, Union
import logging
from torch.utils.data import DataLoader

from ..models.base import BaseModelWrapper
from ..utils.mlflow_utils import MLflowUtils

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


class PytorchLightningWrapper(BaseModelWrapper):
    """PyTorch Lightning model wrapper for training and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.mlflow_config = config.get("mlflow", {})
        
        # Training parameters
        self.max_epochs = self.training_config.get("max_epochs", 10)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 5)
        
        # MLflow setup
        self.mlflow_utils = MLflowUtils(self.mlflow_config)
        self.run_id = None
        
    def train(self, data_loader: DataLoader, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the PyTorch Lightning model."""
        try:
            logger.info("Starting PyTorch Lightning training")
            
            # Start MLflow run
            self.run_id = self.mlflow_utils.start_run()
            
            # Create model
            lightning_model = LightningModel(self.model_config)
            
            # Setup MLflow logger
            mlflow_logger = MLflowLogger(
                experiment_name=self.mlflow_config.get("experiment_name", "pytorch-training"),
                tracking_uri=self.mlflow_utils.tracking_uri
            )
            
            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    filename='best-model-{epoch:02d}-{val_loss:.2f}'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    mode='min'
                )
            ]
            
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
            trainer.fit(lightning_model, data_loader)
            
            # Save model
            self.model = lightning_model
            self.is_trained = True
            
            # Log training completion
            self.mlflow_utils.log_params(self.model_config)
            self.mlflow_utils.log_params(self.training_config)
            
            logger.info("PyTorch Lightning training completed successfully")
            
            return {
                "status": "success",
                "run_id": self.run_id,
                "model_type": "pytorch_lightning",
                "epochs_trained": trainer.current_epoch,
                "best_model_path": trainer.checkpoint_callback.best_model_path
            }
            
        except Exception as e:
            logger.error(f"PyTorch Lightning training failed: {str(e)}")
            if self.run_id:
                self.mlflow_utils.end_run(status="FAILED")
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def evaluate(self, data_loader: DataLoader, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        try:
            logger.info("Starting model evaluation")
            
            # Create trainer for evaluation
            trainer = pl.Trainer(
                accelerator='auto',
                devices='auto',
                logger=False
            )
            
            # Run evaluation
            results = trainer.test(self.model, data_loader)
            
            # Extract metrics
            test_loss = results[0]['test_loss'] if results else 0.0
            
            # Log metrics to MLflow
            self.mlflow_utils.log_metrics({"test_loss": test_loss})
            
            logger.info(f"Model evaluation completed. Test loss: {test_loss}")
            
            return {
                "status": "success",
                "test_loss": test_loss,
                "model_type": "pytorch_lightning",
                "metrics": results[0] if results else {}
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def load(self, model_uri: str) -> None:
        """Load model from MLflow model URI."""
        try:
            logger.info(f"Loading PyTorch Lightning model from {model_uri}")
            
            # Load model using MLflow PyTorch flavor
            self.model = mlflow.pytorch.load_model(model_uri)
            self.is_trained = True
            
            logger.info("PyTorch Lightning model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch Lightning model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the loaded model."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        try:
            logger.info("Running PyTorch Lightning model inference")
            
            # Extract features
            features = input_data.get("features", [])
            if isinstance(features, list):
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(features)
            
            # Convert to serializable format
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy().tolist()
            
            return {
                "prediction": prediction,
                "model_type": "pytorch_lightning",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"PyTorch Lightning model inference failed: {str(e)}")
            return {
                "error": str(e),
                "model_type": "pytorch_lightning",
                "status": "error"
            }
    
    def save_model(self, output_path: str) -> str:
        """Save the trained model."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        try:
            logger.info(f"Saving PyTorch Lightning model to {output_path}")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Save model using MLflow PyTorch flavor
            model_path = os.path.join(output_path, "model")
            mlflow.pytorch.save_model(
                pytorch_model=self.model,
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
            
            logger.info(f"PyTorch Lightning model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save PyTorch Lightning model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")
