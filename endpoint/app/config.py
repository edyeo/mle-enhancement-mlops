"""
Configuration management for the MLOps Endpoint API.
"""
import os
import yaml
from typing import Optional
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """Configuration class for the MLOps Endpoint API."""
    
    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_model_name: str = Field(default="my-model", env="MLFLOW_MODEL_NAME")
    mlflow_model_stage: str = Field(default="Production", env="MLFLOW_MODEL_STAGE")
    
    # Model Settings
    model_type: str = Field(default="pytorch", env="MODEL_TYPE")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Feature Store Settings
    feature_store_enabled: bool = Field(default=False, env="FEATURE_STORE_ENABLED")
    feature_store_uri: str = Field(default="", env="FEATURE_STORE_URI")
    
    # LLM Settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model_name: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL_NAME")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                # Convert YAML keys to environment variable format
                env_vars = {}
                for key, value in config_data.items():
                    env_key = key.upper()
                    env_vars[env_key] = value
                return cls(**env_vars)
        return cls()
    
    def get_model_uri(self) -> str:
        """Get the MLflow model URI for the configured model."""
        return f"models:/{self.mlflow_model_name}/{self.mlflow_model_stage}"
