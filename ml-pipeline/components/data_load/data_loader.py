"""
Data loading component for Kubeflow pipeline.
Loads data from PostgreSQL database and performs preprocessing.
"""
import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import logging
from typing import Dict, Any, Optional
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for PostgreSQL database."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get("database", {})
        self.preprocessing_config = config.get("preprocessing", {})
        
        # Database connection parameters
        self.host = self.db_config.get("host", "localhost")
        self.port = self.db_config.get("port", 5432)
        self.database = self.db_config.get("database", "mlops")
        self.username = self.db_config.get("username", "postgres")
        self.password = self.db_config.get("password", "")
        
        # Query configuration
        self.query = self.db_config.get("query", "")
        self.table_name = self.db_config.get("table_name", "")
        
        # Preprocessing configuration
        self.feature_columns = self.preprocessing_config.get("feature_columns", [])
        self.target_column = self.preprocessing_config.get("target_column", "")
        self.scale_features = self.preprocessing_config.get("scale_features", True)
        self.encode_categorical = self.preprocessing_config.get("encode_categorical", True)
        
        self.scaler = None
        self.label_encoder = None
        
    def connect_to_database(self):
        """Create database connection."""
        try:
            connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            logger.info(f"Connected to PostgreSQL database: {self.database}")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from PostgreSQL database."""
        try:
            self.connect_to_database()
            
            if self.query:
                # Use custom query
                logger.info("Loading data using custom query")
                df = pd.read_sql(self.query, self.engine)
            elif self.table_name:
                # Load entire table
                logger.info(f"Loading data from table: {self.table_name}")
                df = pd.read_sql(f"SELECT * FROM {self.table_name}", self.engine)
            else:
                raise ValueError("Either 'query' or 'table_name' must be specified")
            
            logger.info(f"Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise RuntimeError(f"Data loading failed: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data."""
        try:
            logger.info("Starting data preprocessing")
            
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Handle missing values
            missing_strategy = self.preprocessing_config.get("missing_strategy", "drop")
            if missing_strategy == "drop":
                processed_df = processed_df.dropna()
            elif missing_strategy == "fill_mean":
                numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_columns] = processed_df[numeric_columns].fillna(processed_df[numeric_columns].mean())
            elif missing_strategy == "fill_median":
                numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_columns] = processed_df[numeric_columns].fillna(processed_df[numeric_columns].median())
            
            # Encode categorical variables
            if self.encode_categorical:
                categorical_columns = processed_df.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if col != self.target_column:  # Don't encode target if it's categorical
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                        logger.info(f"Encoded categorical column: {col}")
            
            # Scale features
            if self.scale_features and self.feature_columns:
                numeric_features = [col for col in self.feature_columns if col in processed_df.columns]
                if numeric_features:
                    self.scaler = StandardScaler()
                    processed_df[numeric_features] = self.scaler.fit_transform(processed_df[numeric_features])
                    logger.info(f"Scaled features: {numeric_features}")
            
            logger.info(f"Preprocessing completed. Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise RuntimeError(f"Preprocessing failed: {str(e)}")
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        try:
            logger.info("Splitting data into train/validation/test sets")
            
            # Get split ratios
            train_ratio = self.preprocessing_config.get("train_ratio", 0.8)
            val_ratio = self.preprocessing_config.get("val_ratio", 0.1)
            test_ratio = self.preprocessing_config.get("test_ratio", 0.1)
            
            # Validate ratios
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
            # Shuffle data
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate split indices
            n_samples = len(df_shuffled)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)
            
            # Split data
            train_data = df_shuffled[:train_end]
            val_data = df_shuffled[train_end:val_end]
            test_data = df_shuffled[val_end:]
            
            logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            return {
                "train": train_data,
                "validation": val_data,
                "test": test_data
            }
            
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise RuntimeError(f"Data splitting failed: {str(e)}")
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], output_path: str) -> str:
        """Save processed data to output path."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Save each dataset
            for split_name, df in data_dict.items():
                file_path = os.path.join(output_path, f"{split_name}_data.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {split_name} data to {file_path}")
            
            # Save preprocessing artifacts
            artifacts_path = os.path.join(output_path, "preprocessing_artifacts.pkl")
            artifacts = {
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "preprocessing_config": self.preprocessing_config
            }
            
            with open(artifacts_path, 'wb') as f:
                pickle.dump(artifacts, f)
            logger.info(f"Saved preprocessing artifacts to {artifacts_path}")
            
            # Save data info
            info_path = os.path.join(output_path, "data_info.json")
            data_info = {
                "total_samples": sum(len(df) for df in data_dict.values()),
                "train_samples": len(data_dict["train"]),
                "validation_samples": len(data_dict["validation"]),
                "test_samples": len(data_dict["test"]),
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "data_shape": data_dict["train"].shape
            }
            
            with open(info_path, 'w') as f:
                json.dump(data_info, f, indent=2)
            logger.info(f"Saved data info to {info_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise RuntimeError(f"Data saving failed: {str(e)}")


def main():
    """Main function for data loading component."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data loading component")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--output", required=True, help="Output directory path")
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize data loader
        data_loader = DataLoader(config)
        
        # Load data
        raw_data = data_loader.load_data()
        
        # Preprocess data
        processed_data = data_loader.preprocess_data(raw_data)
        
        # Split data
        split_data = data_loader.split_data(processed_data)
        
        # Save data
        output_path = data_loader.save_data(split_data, args.output)
        
        print(f"Data loading completed successfully. Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
