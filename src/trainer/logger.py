import mlflow

from src.config import LoggingConfig, TrainingConfig

class Logger:
    def __init__(self, logging_config: LoggingConfig, training_config: TrainingConfig):
        self.log_interval = logging_config.log_interval
        self.checkpoint_interval = training_config.checkpoint_interval
        self.log_dir = logging_config.log_dir

        self.lr = training_config.learning_rate

        self.mlflow = logging_config.mlflow
        
        if self.mlflow:
            # Set tracking URI
            mlflow.set_tracking_uri(logging_config.mlflow_tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(logging_config.mlflow_experiment_name)

            print(f"MLflow tracking started. Experiment: {logging_config.mlflow_experiment_name}")
            print(f"MLflow UI available at: mlflow ui --backend-store-uri {logging_config.mlflow_tracking_uri}")

    def start_run(self, run_id: str = None):
        if self.mlflow:
            mlflow.start_run(run_id=run_id)

    def log_params(self, params: dict):
        if self.mlflow:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        if self.mlflow:
            mlflow.log_metrics(metrics, step=step)