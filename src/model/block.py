import torch
import torch.nn as nn

from ..config import GPTConfig
from . import MLP, CausalSelfAttention


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        """Forward pass through the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        x = x + self.attention(self.ln_1(x))  # Attention with residual connection
        x = x + self.mlp(self.ln_2(x))  # MLP with residual connection
        return x
