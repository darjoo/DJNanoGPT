import mlflow
import os
from typing import Dict, Any, Optional

from src.config import LoggingConfig, TrainingConfig

class Logger:
    def __init__(self, logging_config: LoggingConfig):
        self.log_interval = logging_config.log_interval
        self.log_dir = logging_config.log_dir

        self.mlflow = logging_config.mlflow
        
        if self.mlflow:
            # Set tracking URI
            mlflow.set_tracking_uri(logging_config.mlflow_tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(logging_config.mlflow_experiment_name)

            print(f"MLflow tracking started. Experiment: {logging_config.mlflow_experiment_name}")
            print(f"MLflow UI available at: mlflow ui --backend-store-uri {logging_config.mlflow_tracking_uri}")

    def start_run(self, run_id: str = None, run_name: str = None, tags: Dict[str, str] = None):
        """Start an MLflow run with optional tags and name."""
        if self.mlflow:
            mlflow.start_run(run_id=run_id, run_name=run_name)
            if tags:
                mlflow.set_tags(tags)

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self.mlflow and mlflow.active_run():
            mlflow.end_run(status=status)

    def log_params(self, params: dict):
        """Log parameters to MLflow."""
        if self.mlflow:
            # MLflow has a limit of 500 characters per param value
            # and 100 params per batch, so we'll log them carefully
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"Warning: Could not log all parameters: {e}")
                # Try logging one by one if batch fails
                for key, value in params.items():
                    try:
                        mlflow.log_param(key, value)
                    except Exception as e:
                        print(f"Warning: Could not log parameter {key}: {e}")

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to MLflow."""
        if self.mlflow:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log a local file or directory as an artifact."""
        if self.mlflow:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            else:
                print(f"Warning: Artifact path does not exist: {local_path}")

    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a PyTorch model to MLflow."""
        if self.mlflow:
            try:
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            except Exception as e:
                print(f"Warning: Could not log model: {e}")

    def log_dict(self, dictionary: dict, file_name: str):
        """Log a dictionary as a JSON artifact."""
        if self.mlflow:
            mlflow.log_dict(dictionary, file_name)

    def set_tags(self, tags: Dict[str, Any]):
        """Set tags for the current run."""
        if self.mlflow:
            mlflow.set_tags(tags)

    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact."""
        if self.mlflow:
            mlflow.log_text(text, artifact_file)

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.mlflow and mlflow.active_run():
            return mlflow.active_run().info.run_id
        return None