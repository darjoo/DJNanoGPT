import torch
import torch.nn as nn
from ..config import GPTConfig
from . import CasualSelfAttention, MLP

class Block(nn.Module):
    """
    Transformer block
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.attention = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, input: torch.Tensor):
        x = input + self.attention(self.ln_1(input))  # Attention with residual connection
        x = x + self.mlp(self.ln_2(x))                # MLP with residual connection
        return x