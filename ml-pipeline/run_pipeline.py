#!/usr/bin/env python3
"""
Kubeflow 파이프라인 실행 스크립트
MLOps Training Pipeline을 Kubeflow에 제출하고 실행합니다.
"""
import os
import sys
import argparse
import logging
from typing import Optional

# Add pipeline directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

import kfp
from mlops_pipeline import mlops_training_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KubeflowPipelineRunner:
    """Kubeflow 파이프라인 실행기."""
    
    def __init__(self, kubeflow_host: str):
        self.kubeflow_host = kubeflow_host
        self.client = None
        
    def connect(self) -> bool:
        """Kubeflow 서버에 연결."""
        try:
            self.client = kfp.Client(host=self.kubeflow_host)
            logger.info(f"Connected to Kubeflow at {self.kubeflow_host}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kubeflow: {str(e)}")
            return False
    
    def compile_pipeline(self, output_file: str = "mlops_training_pipeline.yaml") -> bool:
        """파이프라인을 컴파일합니다."""
        try:
            logger.info("Compiling pipeline...")
            kfp.compiler.Compiler().compile(
                pipeline_func=mlops_training_pipeline,
                package_path=output_file
            )
            logger.info(f"Pipeline compiled successfully: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Pipeline compilation failed: {str(e)}")
            return False
    
    def submit_pipeline(
        self,
        experiment_name: str = "MLOps Training Pipeline",
        config_path: str = "/config/pipeline_config.yaml",
        pipeline_name: str = "mlops-training-pipeline"
    ) -> Optional[str]:
        """파이프라인을 제출하고 실행합니다."""
        try:
            if not self.client:
                logger.error("Not connected to Kubeflow. Call connect() first.")
                return None
            
            logger.info(f"Submitting pipeline to experiment: {experiment_name}")
            
            # 파이프라인 실행
            run = self.client.create_run_from_pipeline_func(
                pipeline_func=mlops_training_pipeline,
                arguments={
                    "config_path": config_path,
                    "experiment_name": "mlops-training"
                },
                experiment_name=experiment_name
            )
            
            run_id = run.run_id
            logger.info(f"Pipeline submitted successfully!")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Pipeline UI: {self.kubeflow_host}/#/runs/details/{run_id}")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Pipeline submission failed: {str(e)}")
            return None
    
    def get_run_status(self, run_id: str) -> Optional[dict]:
        """실행 상태를 확인합니다."""
        try:
            if not self.client:
                logger.error("Not connected to Kubeflow. Call connect() first.")
                return None
            
            run_details = self.client.get_run(run_id)
            status = {
                "run_id": run_id,
                "state": run_details.state,
                "created_at": run_details.created_at,
                "finished_at": run_details.finished_at,
                "pipeline_spec": run_details.pipeline_spec
            }
            
            logger.info(f"Run {run_id} status: {run_details.state}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to get run status: {str(e)}")
            return None
    
    def list_experiments(self) -> list:
        """실험 목록을 조회합니다."""
        try:
            if not self.client:
                logger.error("Not connected to Kubeflow. Call connect() first.")
                return []
            
            experiments = self.client.list_experiments()
            experiment_list = []
            
            for exp in experiments.experiments:
                experiment_list.append({
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "created_at": exp.created_at
                })
            
            logger.info(f"Found {len(experiment_list)} experiments")
            return experiment_list
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {str(e)}")
            return []
    
    def list_runs(self, experiment_id: Optional[str] = None) -> list:
        """실행 목록을 조회합니다."""
        try:
            if not self.client:
                logger.error("Not connected to Kubeflow. Call connect() first.")
                return []
            
            if experiment_id:
                runs = self.client.list_runs(experiment_id=experiment_id)
            else:
                runs = self.client.list_runs()
            
            run_list = []
            for run in runs.runs:
                run_list.append({
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "finished_at": run.finished_at
                })
            
            logger.info(f"Found {len(run_list)} runs")
            return run_list
            
        except Exception as e:
            logger.error(f"Failed to list runs: {str(e)}")
            return []


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="Kubeflow MLOps Pipeline Runner")
    parser.add_argument("--host", default="http://localhost:43000", 
                       help="Kubeflow server host (default: http://localhost:43000)")
    parser.add_argument("--action", choices=["compile", "submit", "status", "list-experiments", "list-runs"],
                       default="submit", help="Action to perform")
    parser.add_argument("--run-id", help="Run ID for status check")
    parser.add_argument("--experiment-name", default="MLOps Training Pipeline",
                       help="Experiment name")
    parser.add_argument("--config-path", default="/config/pipeline_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output-file", default="mlops_training_pipeline.yaml",
                       help="Compiled pipeline output file")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = KubeflowPipelineRunner(args.host)
    
    if args.action == "compile":
        # Compile pipeline only
        success = runner.compile_pipeline(args.output_file)
        if success:
            print(f"Pipeline compiled successfully: {args.output_file}")
        else:
            print("Pipeline compilation failed")
            sys.exit(1)
    
    elif args.action == "submit":
        # Connect and submit pipeline
        if not runner.connect():
            print("Failed to connect to Kubeflow")
            sys.exit(1)
        
        run_id = runner.submit_pipeline(
            experiment_name=args.experiment_name,
            config_path=args.config_path
        )
        
        if run_id:
            print(f"Pipeline submitted successfully!")
            print(f"Run ID: {run_id}")
            print(f"Pipeline UI: {args.host}/#/runs/details/{run_id}")
        else:
            print("Pipeline submission failed")
            sys.exit(1)
    
    elif args.action == "status":
        # Check run status
        if not args.run_id:
            print("Run ID is required for status check")
            sys.exit(1)
        
        if not runner.connect():
            print("Failed to connect to Kubeflow")
            sys.exit(1)
        
        status = runner.get_run_status(args.run_id)
        if status:
            print(f"Run Status: {status['state']}")
            print(f"Created: {status['created_at']}")
            print(f"Finished: {status['finished_at']}")
        else:
            print("Failed to get run status")
            sys.exit(1)
    
    elif args.action == "list-experiments":
        # List experiments
        if not runner.connect():
            print("Failed to connect to Kubeflow")
            sys.exit(1)
        
        experiments = runner.list_experiments()
        if experiments:
            print("Experiments:")
            for exp in experiments:
                print(f"  - {exp['name']} (ID: {exp['id']})")
        else:
            print("No experiments found")
    
    elif args.action == "list-runs":
        # List runs
        if not runner.connect():
            print("Failed to connect to Kubeflow")
            sys.exit(1)
        
        runs = runner.list_runs()
        if runs:
            print("Runs:")
            for run in runs:
                print(f"  - {run['name']} (ID: {run['id']}, State: {run['state']})")
        else:
            print("No runs found")


if __name__ == "__main__":
    main()
