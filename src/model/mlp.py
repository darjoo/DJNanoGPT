import torch
import torch.nn as nn
from ..config import GPTConfig

class MLP(nn.Module):
    """Feed-forward block used inside the transformer layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        expansion = 4 * config.hidden_size
        self.linear = nn.Linear(config.hidden_size, expansion, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.projection = nn.Linear(expansion, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(input)
        hidden = self.gelu(hidden)
        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        return hidden