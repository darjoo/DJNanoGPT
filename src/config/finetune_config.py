from dataclasses import dataclass

@dataclass
class FinetuneConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 16
    grad_clip: float = 1.0
    checkpoint_interval: int = 1  # Save checkpoint every n epochs
    eval_iters: int = 5  # Evaluate every n iterations
    checkpoint_dir: str = 'finetune_checkpoints'