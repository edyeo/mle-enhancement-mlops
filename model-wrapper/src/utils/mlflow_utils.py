"""
MLflow utilities for model training and logging.
"""
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowUtils:
    """Utility class for MLflow operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracking_uri = config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = config.get("experiment_name", "mlops-training")
        self.run_id = None
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
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
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        try:
            if run_name is None:
                run_name = f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name
            )
            
            self.run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")
            
            return self.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            raise RuntimeError(f"MLflow run start failed: {str(e)}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        try:
            if self.run_id:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run: {self.run_id} with status: {status}")
                self.run_id = None
            else:
                logger.warning("No active MLflow run to end")
                
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        try:
            if self.run_id:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {list(params.keys())}")
            else:
                logger.warning("No active MLflow run to log parameters")
                
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        try:
            if self.run_id:
                mlflow.log_metrics(metrics)
                logger.info(f"Logged metrics: {list(metrics.keys())}")
            else:
                logger.warning("No active MLflow run to log metrics")
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_artifacts(self, artifacts_path: str) -> None:
        """Log artifacts to MLflow."""
        try:
            if self.run_id and os.path.exists(artifacts_path):
                mlflow.log_artifacts(artifacts_path)
                logger.info(f"Logged artifacts from: {artifacts_path}")
            else:
                logger.warning(f"No active MLflow run or artifacts path not found: {artifacts_path}")
                
        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")
    
    def log_model(self, model, model_name: str, model_type: str = "pytorch") -> str:
        """Log model to MLflow."""
        try:
            if not self.run_id:
                raise RuntimeError("No active MLflow run")
            
            if model_type.lower() == "pytorch":
                model_uri = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_name,
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
            elif model_type.lower() == "pyfunc":
                model_uri = mlflow.pyfunc.log_model(
                    python_model=model,
                    artifact_path=model_name,
                    conda_env={
                        "channels": ["defaults"],
                        "dependencies": [
                            "python=3.9",
                            "mlflow"
                        ]
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Logged model: {model_uri}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise RuntimeError(f"Model logging failed: {str(e)}")
    
    def register_model(self, model_uri: str, model_name: str, 
                      description: Optional[str] = None) -> str:
        """Register model in MLflow Model Registry."""
        try:
            if not self.run_id:
                raise RuntimeError("No active MLflow run")
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description or f"Model trained at {datetime.now()}"
            )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise RuntimeError(f"Model registration failed: {str(e)}")
    
    def transition_model_stage(self, model_name: str, version: str, 
                              stage: str) -> None:
        """Transition model to a specific stage."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Transitioned model {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {str(e)}")
            raise RuntimeError(f"Model stage transition failed: {str(e)}")
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run."""
        if self.run_id:
            try:
                run = mlflow.get_run(self.run_id)
                return {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time
                }
            except Exception as e:
                logger.error(f"Failed to get run info: {str(e)}")
                return {}
        return {}
