"""Data loading utilities for training language models.

This module provides:
- DataLoader: A class for loading and batching tokenized text data from memory-mapped files.
"""

import os

import numpy as np
import torch


class DataLoader:
    """A data loader for loading batches of tokenized text data.

    This class provides functionality to load and batch data from memory-mapped files
    for training, validation, or finetuning language models.

    Attributes:
    ----------
    data_dir : str
        Directory containing the data files.
    block_size : int
        Size of each sequence block.
    batch_size : int
        Number of sequences per batch.
    device : str
        Device to use for tensor operations ('cpu' or 'cuda').
    random : bool
        Whether to randomly sample batches.
    current_i : int
        Current index for sequential batch sampling.

    Methods:
    -------
    get_data_size(split: str) -> int
        Get the size of the dataset for a given split.
    get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]
        Generate a batch of input and target sequences.

    """

    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: str, random: bool = True) -> None:
        """Initialize the DataLoader.

        Args:
            data_dir (str): Directory containing the data files
            block_size (int): Size of each sequence block
            batch_size (int): Number of sequences per batch
            device (str): Device to use for tensor operations ('cpu' or 'cuda')
            random (bool): Whether to randomly sample batches (default: True)

        """
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.random = random
        self.current_i = 0

    def get_data_size(self, split: str) -> int:
        """Get the size of the dataset for a given split.

        Args:
            split (str): Dataset split - 'train', 'validation', or 'finetune'

        Returns:
            int: Number of tokens in the dataset

        Raises:
            ValueError: If split is not 'train', 'validation', or 'finetune'

        """
        if split == "train":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_train.bin"), dtype=np.uint16, mode="r")
        elif split == "validation":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_validation.bin"), dtype=np.uint16, mode="r")
        elif split == "finetune":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_finetune.bin"), dtype=np.uint16, mode="r")
        else:
            raise ValueError("split must be 'train', 'validation', or 'finetune'")

        return len(data)

    def get_batch(self, split):
        """Generate a batch of data.

        1. Recreate the memmap every time to prevent memory leak
        2. Randomly select batch_size starting indices for the sequences
        3. For each starting index, get the sequence of length block_size as input (x)
        and the sequence shifted by one as target (y)

        Args:
            split (str): 'train' or 'validation' or 'finetune'

        Returns:
            x (torch.Tensor): Input tensor of shape (batch_size, block_size)
            y (torch.Tensor): Target tensor of shape (batch_size, block_size)

        """
        # Recreate memmap every batch to avoid memory leak
        if split == "train":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_train.bin"), dtype=np.uint16, mode="r")
        elif split == "validation":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_validation.bin"), dtype=np.uint16, mode="r")
        elif split == "finetune":
            data = np.memmap(os.path.join(self.data_dir, "tinystories_finetune.bin"), dtype=np.uint16, mode="r")
        else:
            raise ValueError("split must be 'train', 'validation', or 'finetune'")

        max_idx = len(data) - self.block_size
        if max_idx <= 0:
            raise ValueError(
                f"Dataset split '{split}' with length {len(data)} is too short for block_size {self.block_size}.",
            )
        if self.random:
            ix = torch.randint(0, max_idx, (self.batch_size,))
        else:
            ix = torch.arange(self.current_i, self.current_i + self.batch_size) % max_idx
            self.current_i = (self.current_i + self.batch_size) % max_idx
        x = torch.stack([torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + self.block_size + 1]).astype(np.int64)) for i in ix])
        if self.device == "cuda":
            # Pin arrays x & y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
