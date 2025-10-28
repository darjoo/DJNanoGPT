"""Utilities for loading and saving model checkpoints."""
import os
import torch
from typing import Dict, Any, Tuple, Optional

from src.config import GPTConfig


def load_checkpoint(
    checkpoint_path: str, 
    device: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a model checkpoint and fix state dict keys if needed.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to ('cuda' or 'cpu')
        
    Returns:
        Tuple of (checkpoint dict, cleaned state_dict)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If model args validation fails
        RuntimeError: If checkpoint loading fails
    """
    # Validate checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Register safe globals for torch.load
    torch.serialization.add_safe_globals([GPTConfig])
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    # Extract state dict
    state_dict = checkpoint['model']
    
    # Fix keys from compiled models (remove '_orig_mod.' prefix)
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    return checkpoint, state_dict


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    model_args: Dict[str, Any],
    iter_num: int,
    best_val_loss: float,
    config: Any,
    run_id: Optional[str] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_path: Path where to save the checkpoint
        model: The model to save
        optimizer: The optimizer state to save
        scaler: The gradient scaler state to save
        model_args: Dictionary of model arguments
        iter_num: Current iteration number
        best_val_loss: Best validation loss so far
        config: Model configuration
        run_id: Optional wandb/mlflow run ID
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'wandb_run_id': run_id
    }
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {checkpoint_path}: {e}")
