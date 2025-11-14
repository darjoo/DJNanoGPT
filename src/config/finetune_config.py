from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning a GPT model.

    Attributes:
        batch_size (int): The number of sequences processed in one forward/backward pass.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating model.
        grad_clip (float): Maximum norm for gradient clipping.
        checkpoint_interval (int): Save checkpoint every n epochs.
        eval_interval (int): Evaluate every n iterations.
        eval_steps (int): Number of batches to use for evaluation.
        checkpoint_dir (str): Directory to save fine-tuning checkpoints.
    """

    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 16
    grad_clip: float = 1.0
    checkpoint_interval: int = 1  # Save checkpoint every n epochs
    eval_interval: int = 5  # Evaluate every n iterations
    eval_steps: int = 10  # Number of batches to use for evaluation
    checkpoint_dir: str = "finetune_checkpoints"
