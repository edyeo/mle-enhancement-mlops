"""
Kubeflow Pipeline definition for MLOps training pipeline.
"""
import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model, Dataset
from typing import NamedTuple

from components.data_load.data_loader import DataLoader
from components.train.model_trainer import ModelTrainer
from components.evaluation.model_evaluator import ModelEvaluator

@component(
    base_image="localhost:65000/mlops-pipeline:latest"
)
def data_load_component(
    config_path: str,
    output_data: Output[Dataset]
) -> NamedTuple('DataLoadOutput', [('data_path', str), ('data_info', str)]):
    """Data loading component."""
    import os
    import sys
    import yaml
    import json
    import logging
    from typing import NamedTuple
    
    # Add the component path to sys.path
    sys.path.append('/components/data_load')
    

    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize data loader
        data_loader = DataLoader(config)
        
        # Load and preprocess data
        raw_data = data_loader.load_data()
        processed_data = data_loader.preprocess_data(raw_data)
        split_data = data_loader.split_data(processed_data)
        
        # Save data
        output_path = data_loader.save_data(split_data, output_data.path)
        
        # Get data info
        data_info = {
            "total_samples": sum(len(df) for df in split_data.values()),
            "train_samples": len(split_data["train"]),
            "validation_samples": len(split_data["validation"]),
            "test_samples": len(split_data["test"]),
            "feature_columns": data_loader.feature_columns,
            "target_column": data_loader.target_column
        }
        
        logger.info(f"Data loading completed. Output: {output_path}")
        
        return NamedTuple('DataLoadOutput', [('data_path', str), ('data_info', str)])(
            data_path=output_path,
            data_info=json.dumps(data_info)
        )
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


@component(
    base_image="localhost:65000/mlops-pipeline:latest"
)
def train_component(
    config_path: str,
    data_path: Input[Dataset],
    output_model: Output[Model]
) -> NamedTuple('TrainOutput', [('model_path', str), ('run_id', str)]):
    """Model training component."""
    import os
    import sys
    import yaml
    import logging
    from typing import NamedTuple
    
    # Add the component path to sys.path
    sys.path.append('/components/train')

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Load data
        data_loaders = trainer.load_data(data_path.path)
        
        # Train model
        training_result = trainer.train_model(data_loaders)
        
        # Save model
        model_path = trainer.save_model(training_result["model"], output_model.path)
        
        logger.info(f"Training completed successfully. Model saved to: {model_path}")
        
        return NamedTuple('TrainOutput', [('model_path', str), ('run_id', str)])(
            model_path=model_path,
            run_id=training_result["run_id"]
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@component(
    base_image="localhost:65000/mlops-pipeline:latest"
)
def evaluation_component(
    config_path: str,
    model_path: Input[Model],
    data_path: Input[Dataset],
    output_results: Output[Artifact]
) -> NamedTuple('EvaluationOutput', [('model_version', str), ('metrics', str)]):
    """Model evaluation component."""
    import os
    import sys
    import yaml
    import json
    import logging
    from typing import NamedTuple
    
    # Add the component path to sys.path
    sys.path.append('/components/evaluation')

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Setup MLflow
        evaluator.setup_mlflow()
        
        # Load model
        model = evaluator.load_model(model_path.path)
        
        # Load test data
        test_loader = evaluator.load_test_data(data_path.path)
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, test_loader)
        
        # Log metrics to MLflow
        evaluator.log_metrics_to_mlflow(metrics, evaluator.run_id)
        
        # Register model
        model_version = evaluator.register_model(model_path.path, metrics)
        
        # Save evaluation results
        results_path = evaluator.save_evaluation_results(metrics, model_version, output_results.path)
        
        logger.info(f"Evaluation completed successfully. Model version: {model_version}")
        
        return NamedTuple('EvaluationOutput', [('model_version', str), ('metrics', str)])(
            model_version=model_version,
            metrics=json.dumps(metrics)
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


@pipeline(
    name="mlops-training-pipeline",
    description="MLOps training pipeline with data loading, training, and evaluation"
)
def mlops_training_pipeline(
    config_path: str = "/config/pipeline_config.yaml",
    experiment_name: str = "mlops-training"
):
    """
    MLOps training pipeline.
    
    Args:
        config_path: Path to the pipeline configuration file
        experiment_name: MLflow experiment name
    """
    
    # Data loading step
    data_load_task = data_load_component(
        config_path=config_path
    )
    
    # Model training step
    train_task = train_component(
        config_path=config_path,
        data_path=data_load_task.outputs['output_data']
    )
    
    # Model evaluation step
    evaluation_task = evaluation_component(
        config_path=config_path,
        model_path=train_task.outputs['output_model'],
        data_path=data_load_task.outputs['output_data']
    )
    
    # Set dependencies
    train_task.after(data_load_task)
    evaluation_task.after(train_task)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=mlops_training_pipeline,
        package_path="mlops_training_pipeline.yaml"
    )
    print("Pipeline compiled successfully!")
