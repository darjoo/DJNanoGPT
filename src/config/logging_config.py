from dataclasses import dataclass
from typing import Optional

@dataclass
class LoggingConfig:
    """
    Configuration for logging and checkpointing during training.
    
    Attributes:
        log_interval (int): Number of iterations between logging training metrics.
        checkpoint_interval (int): Number of iterations between saving model checkpoints.
        log_dir (str): Directory to save logs and checkpoints.
        wandb (bool): Whether to use Weights & Biases for experiment tracking.
        wandb_project (str): Name of the wandb project.
        wandb_dir (str): Directory to save wandb logs locally.
        wandb_entity (str): Optional wandb entity (username or team).
        wandb_mode (str): Mode for wandb ('online', 'offline', or 'disabled').
        wandb_base_url (str): Base URL for wandb server (e.g., 'http://localhost:8080' for local Docker).
    """

    log_interval: int = 1
    log_dir: str = 'logs'

    wandb: bool = True
    wandb_project: str = 'gpt-training'
    wandb_dir: str = './wandb'
    wandb_entity: str = None
    wandb_mode: str = 'online'  # Use 'online' to sync with local Docker server
    wandb_base_url: Optional[str] = None  # Provide a value when using a self-hosted Weights & Biases server