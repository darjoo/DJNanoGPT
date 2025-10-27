import os
import tempfile
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:  # Optional dependency: wandb may not be installed
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - exercised only when wandb missing
    wandb = None  # type: ignore

from src.config import LoggingConfig

# Load environment variables from .env file
load_dotenv()

class Logger:
    def __init__(self, logging_config: LoggingConfig):
        self.log_interval = logging_config.log_interval
        self.log_dir = logging_config.log_dir

        self._wandb = wandb
        self.wandb_enabled = bool(logging_config.wandb and self._wandb is not None)
        self.wandb_config = logging_config
        
        if logging_config.wandb and not self.wandb_enabled:
            print("Warning: wandb package is not available; disabling Weights & Biases logging.")

        if self.wandb_enabled:
            # Set base URL for local wandb server if specified
            if hasattr(logging_config, 'wandb_base_url') and logging_config.wandb_base_url:
                os.environ['WANDB_BASE_URL'] = logging_config.wandb_base_url
                # For local servers, API key is loaded from .env file
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

        self._wandb.init(
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
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            # Map status to wandb exit codes
            exit_code = 0 if status == "FINISHED" else 1
            self._wandb.finish(exit_code=exit_code)

    def log_params(self, params: dict):
        """Log parameters to wandb."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            try:
                self._wandb.config.update(params, allow_val_change=True)
            except Exception as e:
                print(f"Warning: Could not log all parameters: {e}")

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to wandb."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log a local file or directory as an artifact."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            if os.path.exists(local_path):
                self._wandb.save(local_path, base_path=os.path.dirname(local_path) if artifact_path else None)
            else:
                print(f"Warning: Artifact path does not exist: {local_path}")

    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a PyTorch model to wandb."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            try:
                # Save model to temporary file and log it
                import torch
                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, f"{artifact_path}.pt")
                torch.save(model.state_dict(), model_path)
                self._wandb.save(model_path)
                # Log as wandb artifact for versioning
                artifact = self._wandb.Artifact(artifact_path, type='model')
                artifact.add_file(model_path)
                self._wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Could not log model: {e}")

    def log_dict(self, dictionary: dict, file_name: str):
        """Log a dictionary as a JSON artifact."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            import json
            temp_file = os.path.join(tempfile.gettempdir(), file_name)
            with open(temp_file, 'w') as f:
                json.dump(dictionary, f, indent=2)
            self._wandb.save(temp_file)

    def set_tags(self, tags: Dict[str, Any]):
        """Set tags for the current run."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            self._wandb.run.tags = self._wandb.run.tags + tuple(str(v) for v in tags.values())

    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact."""
        if not self.wandb_enabled:
            return

        if self._wandb.run is not None:
            temp_file = os.path.join(tempfile.gettempdir(), artifact_file)
            with open(temp_file, 'w') as f:
                f.write(text)
            self._wandb.save(temp_file)

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if not self.wandb_enabled:
            return None

        if self._wandb.run is not None:
            return self._wandb.run.id
        return None