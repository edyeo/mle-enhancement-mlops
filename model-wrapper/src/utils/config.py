"""
Configuration management utilities.
"""
import os
import yaml
import json
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for model training."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            logger.info(f"Loading configuration from {config_path}")
            
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise RuntimeError(f"Configuration loading failed: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)
    
    def get_nested(self, keys: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        keys_list = keys.split('.')
        value = self.config
        
        for key in keys_list:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def set_nested(self, keys: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys_list = keys.split('.')
        config = self.config
        
        for key in keys_list[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys_list[-1]] = value
    
    def save_config(self, output_path: str) -> None:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif output_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported output file format: {output_path}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise RuntimeError(f"Configuration saving failed: {str(e)}")
    
    def merge_env_vars(self) -> None:
        """Merge environment variables into configuration."""
        env_mappings = {
            "MLFLOW_TRACKING_URI": "mlflow.tracking_uri",
            "MLFLOW_EXPERIMENT_NAME": "mlflow.experiment_name",
            "MODEL_TYPE": "model.type",
            "DATA_SOURCE_PATH": "data.file_path",
            "DATA_LOADER_TYPE": "data.loader_type",
            "BATCH_SIZE": "training.batch_size",
            "MAX_EPOCHS": "training.max_epochs",
            "LEARNING_RATE": "model.learning_rate"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert string values to appropriate types
                if env_value.lower() in ['true', 'false']:
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif env_value.replace('.', '').isdigit():
                    env_value = float(env_value)
                
                self.set_nested(config_key, env_value)
                logger.info(f"Merged environment variable {env_var} -> {config_key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("model", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get("training", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get("data", {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get("mlflow", {})
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required_sections = ["model", "training", "data", "mlflow"]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate MLflow config
        mlflow_config = self.get_mlflow_config()
        if not mlflow_config.get("tracking_uri"):
            logger.error("MLflow tracking_uri is required")
            return False
        
        # Validate data config
        data_config = self.get_data_config()
        if not data_config.get("file_path") and not data_config.get("loader_type"):
            logger.error("Data file_path or loader_type is required")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        return self.config.copy()
