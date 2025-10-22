import torch
import torch.nn as nn
from ..config import GPTConfig

class MLP(nn.Module):
    """
    A feedforward neural network (MLP).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion: hidden_size -> 4 * hidden_size
        # Original "Attention is All You Need" paper used this 4x expansion factor
        self.linear = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        # Projection: 4 * hidden_size -> hidden_size
        self.projection = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor):
        x = self.linear(input)  # Expand 
        x = self.gelu(x)        # Non-linearity
        x = self.projection(x)  # Project back
        x = self.dropout(x)
        return x