import torch
import torch.nn as nn
from ..config import GPTConfig

class MLP(nn.Module):
    """
    A feedforward neural network (MLP).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        expansion = 4 * config.hidden_size
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, expansion, bias=config.bias),
            nn.GELU(approximate='tanh'),
            nn.Linear(expansion, config.hidden_size, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, input: torch.Tensor):
        return self.ff(input)