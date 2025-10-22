"""
Base data loader interface for MLOps training pipeline.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source = config.get("data_source", "")
        
    @abstractmethod
    def load_data(self) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Load data from the configured source.
        
        Returns:
            Loaded data as DataFrame or list of dictionaries
        """
        pass
    
    @abstractmethod
    def get_train_data(self) -> Any:
        """Get training data."""
        pass
    
    @abstractmethod
    def get_validation_data(self) -> Any:
        """Get validation data."""
        pass
    
    @abstractmethod
    def get_test_data(self) -> Any:
        """Get test data."""
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        return {
            "data_source": self.data_source,
            "config": self.config
        }


class BaseDataset(Dataset):
    """Base PyTorch Dataset class."""
    
    def __init__(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], 
                 feature_columns: List[str], 
                 target_column: Optional[str] = None):
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        if isinstance(data, pd.DataFrame):
            self.features = data[feature_columns].values
            self.targets = data[target_column].values if target_column else None
        else:
            # Handle list of dictionaries
            self.features = [[row[col] for col in feature_columns] for row in data]
            self.targets = [row[target_column] for row in data] if target_column else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features, target
        else:
            return features
