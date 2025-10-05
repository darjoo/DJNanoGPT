from dataclasses import dataclass

@dataclass
class LoggingConfig:
    """
    Configuration for logging and checkpointing during training.
    
    Attributes:
        log_interval (int): Number of iterations between logging training metrics.
        checkpoint_interval (int): Number of iterations between saving model checkpoints.
        log_dir (str): Directory to save logs and checkpoints.
        mlflow (bool): Whether to use MLflow for experiment tracking.
    """

    log_interval: int = 1
    log_dir: str = 'logs'

    mlflow: bool = True
    mlflow_experiment_name: str = 'gpt-training'
    mlflow_tracking_uri: str = './mlruns'