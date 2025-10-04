import torch
import torch.nn as nn
from ..config import GPTConfig

class CasualSelfAttention(nn.Module):
    """
    Casual self-attention mechanism.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embedding % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Key, Query, Value projections for all heads
        self.c_attention = nn.Linear(config.n_embedding, 3 * config.n_embedding, bias=config.bias)

        # Output projection
        self.c_projection = nn.Linear(config.n_embedding, config.n_embedding, bias=config.bias)

        # Regularization
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embedding = config.n_embedding
        self.dropout = config.dropout
        
        # Flash attention
        self.flash_attention = hasattr(nn.functional, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor, layer_past: tuple = None):
        batch_size, sequence_length, embedding_dim = x.size()

        # Calculate query, key and values for all heads and move head forward to be the batch dimension
        query, key, value = self.c_attention(x).split(self.n_embedding, dim=2)
        key = key.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)
        query = query.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)
        value = value.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)

        # Casual self attention: (B, nH, T, Hsz) x (B, nH, Hsz, T) -> (B, nH, T, T)
        # is_causal=True applies the causal mask automatically (The triangular mask)
        y = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim) # re-assemble all head outputs side by side

        # Output projection
        y = self.residual_dropout(self.c_projection(y))
        return y