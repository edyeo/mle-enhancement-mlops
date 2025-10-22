"""
Local data loader for loading data from local files.
"""
import os
import pandas as pd
import json
from typing import Dict, Any, Optional, Union, List
from torch.utils.data import DataLoader
import logging

from .base import BaseDataLoader, BaseDataset

logger = logging.getLogger(__name__)


class LocalDataLoader(BaseDataLoader):
    """Data loader for local files (CSV, Parquet, JSON)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = config.get("file_path", "")
        self.file_format = config.get("file_format", "csv")
        self.feature_columns = config.get("feature_columns", [])
        self.target_column = config.get("target_column", None)
        self.train_split = config.get("train_split", 0.8)
        self.val_split = config.get("val_split", 0.1)
        self.test_split = config.get("test_split", 0.1)
        self.batch_size = config.get("batch_size", 32)
        self.shuffle = config.get("shuffle", True)
        
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Load data from local file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        logger.info(f"Loading data from {self.file_path}")
        
        try:
            if self.file_format.lower() == "csv":
                self.data = pd.read_csv(self.file_path)
            elif self.file_format.lower() == "parquet":
                self.data = pd.read_parquet(self.file_path)
            elif self.file_format.lower() == "json":
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            logger.info(f"Loaded {len(self.data)} records")
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def _split_data(self) -> None:
        """Split data into train/validation/test sets."""
        if self.data is None:
            self.load_data()
        
        if isinstance(self.data, pd.DataFrame):
            # Shuffle data
            data_shuffled = self.data.sample(frac=1).reset_index(drop=True)
            
            # Calculate split indices
            n_samples = len(data_shuffled)
            train_end = int(n_samples * self.train_split)
            val_end = train_end + int(n_samples * self.val_split)
            
            # Split data
            self.train_data = data_shuffled[:train_end]
            self.val_data = data_shuffled[train_end:val_end]
            self.test_data = data_shuffled[val_end:]
            
        else:
            # Handle list of dictionaries
            import random
            random.shuffle(self.data)
            
            n_samples = len(self.data)
            train_end = int(n_samples * self.train_split)
            val_end = train_end + int(n_samples * self.val_split)
            
            self.train_data = self.data[:train_end]
            self.val_data = self.data[train_end:val_end]
            self.test_data = self.data[val_end:]
        
        logger.info(f"Data split - Train: {len(self.train_data)}, "
                   f"Val: {len(self.val_data)}, Test: {len(self.test_data)}")
    
    def get_train_data(self) -> DataLoader:
        """Get training data loader."""
        if self.train_data is None:
            self._split_data()
        
        dataset = BaseDataset(
            self.train_data, 
            self.feature_columns, 
            self.target_column
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2
        )
    
    def get_validation_data(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_data is None:
            self._split_data()
        
        dataset = BaseDataset(
            self.val_data, 
            self.feature_columns, 
            self.target_column
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def get_test_data(self) -> DataLoader:
        """Get test data loader."""
        if self.test_data is None:
            self._split_data()
        
        dataset = BaseDataset(
            self.test_data, 
            self.feature_columns, 
            self.target_column
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded data."""
        info = super().get_data_info()
        
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                info.update({
                    "total_samples": len(self.data),
                    "feature_columns": self.feature_columns,
                    "target_column": self.target_column,
                    "data_shape": self.data.shape,
                    "data_types": self.data.dtypes.to_dict()
                })
            else:
                info.update({
                    "total_samples": len(self.data),
                    "feature_columns": self.feature_columns,
                    "target_column": self.target_column
                })
        
        return info
