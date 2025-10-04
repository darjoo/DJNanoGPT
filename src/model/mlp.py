import torch
import torch.nn as nn
from ..config import GPTConfig

class MLP(nn.Module):
    """
    A feedforward neural network (MLP).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion: n_embedding -> 4 * n_embedding
        # Original "Attention is All You Need" paper used this 4x expansion factor
        self.linear = nn.Linear(config.n_embedding, 4 * config.n_embedding, bias=config.bias)
        self.gelu = nn.GELU()
        # Projection: 4 * n_embedding -> n_embedding
        self.projection = nn.Linear(4 * config.n_embedding, config.n_embedding, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor):
        x = self.linear(input)  # Expand 
        x = self.gelu(x)        # Non-linearity
        x = self.projection(x)  # Project back
        x = self.dropout(x)
        return x