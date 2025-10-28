import os
import tempfile
from typing import Dict, Optional
from functools import wraps

try:  # Optional dependency: wandb may not be installed
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - exercised only when wandb missing
    wandb = None  # type: ignore

from src.config import LoggingConfig


def _requires_wandb(func):
    """Decorator to skip execution if wandb is not enabled or no active run."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.wandb_enabled or self._run is None:
            return None
        return func(self, *args, **kwargs)
    return wrapper


class Logger:
    def __init__(self, logging_config: LoggingConfig):
        self.log_interval = logging_config.log_interval
        self.log_dir = logging_config.log_dir

        self._wandb = wandb
        self.wandb_enabled = bool(logging_config.wandb and self._wandb is not None)
        self.wandb_config = logging_config
        self._run = None  # Cache for active wandb run
        
        if logging_config.wandb and not self.wandb_enabled:
            print("Warning: wandb package is not available; disabling Weights & Biases logging.")

        if self.wandb_enabled:
            # Load environment variables only when wandb is enabled
            from dotenv import load_dotenv
            load_dotenv()
            
            # Set base URL for local wandb server if specified
            if logging_config.wandb_base_url:
                os.environ['WANDB_BASE_URL'] = logging_config.wandb_base_url
                if not os.environ.get('WANDB_API_KEY'):
                    print("Warning: WANDB_API_KEY not found in environment variables. Please set it in .env file.")
                print(f"Weights & Biases tracking configured with local server: {logging_config.wandb_base_url}")
            
            print(f"Weights & Biases tracking configured. Project: {logging_config.wandb_project}")
            print(f"Mode: {logging_config.wandb_mode}")
            if logging_config.wandb_mode == 'offline':
                print(f"Running in offline mode. Logs will be saved to: {logging_config.wandb_dir}")
                print(f"To sync later, run: wandb sync {logging_config.wandb_dir}/<run-folder>")

    def start_run(self, run_id: str = None, run_name: str = None, tags: Dict[str, str] = None):
        """Start a wandb run with optional tags and name."""
        if not self.wandb_enabled:
            return

        self._run = self._wandb.init(
            project=self.wandb_config.wandb_project,
            entity=self.wandb_config.wandb_entity,
            name=run_name,
            id=run_id,
            dir=self.wandb_config.wandb_dir,
            mode=self.wandb_config.wandb_mode,
            tags=list(tags.values()) if tags else None,
            resume="allow" if run_id else None
        )
        # Log tags as config if provided
        if tags:
            self._wandb.config.update({f"tag_{k}": v for k, v in tags.items()})

    def end_run(self, status: str = "FINISHED"):
        """End the current wandb run."""
        if not self.wandb_enabled or self._run is None:
            return

        # Map status to wandb exit codes
        exit_code = 0 if status == "FINISHED" else 1
        self._wandb.finish(exit_code=exit_code)
        self._run = None  # Clear cached run

    @_requires_wandb
    def log_params(self, params: dict):
        """Log parameters to wandb."""
        try:
            self._wandb.config.update(params, allow_val_change=True)
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")

    @_requires_wandb
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to wandb."""
        self._wandb.log(metrics, step=step)

    @_requires_wandb
    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact."""
        temp_file = os.path.join(tempfile.gettempdir(), artifact_file)
        try:
            with open(temp_file, 'w') as f:
                f.write(text)
            self._wandb.save(temp_file)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass  # Ignore errors during cleanup

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self._run is not None:
            return self._run.id
        return None