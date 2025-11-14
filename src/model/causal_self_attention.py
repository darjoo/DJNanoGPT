import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GPTConfig
from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Causal self-attention mechanism."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.hidden_size = config.hidden_size
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # Key, Query, Value projections for all heads
        self.c_attention = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config.bias)

        # Output projection
        self.c_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)

        # Regularization
        self.residual_dropout = nn.Dropout(config.dropout)

        self.use_rotary = config.use_rotary_embeddings
        if self.use_rotary:
            self.rotary_dim = config.rotary_dim if config.rotary_dim is not None else self.head_dim
            assert self.rotary_dim % 2 == 0, "Rotary embedding dimension must be even"
            assert self.rotary_dim <= self.head_dim, "rotary_dim cannot exceed head dimension"
            self.rotary_emb = RotaryEmbedding(
                dim=self.rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_dim = 0
            self.rotary_emb = None

    def forward(self, x: torch.Tensor):
        """Forward pass of the causal self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = x.size()

        # chunk(3) splits the projection into equal Q/K/V slices along the feature axis
        query, key, value = self.c_attention(x).chunk(3, dim=-1)

        def shape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        query, key, value = map(shape_heads, (query, key, value))

        if self.use_rotary:
            cos, sin = self.rotary_emb(
                sequence_length,
                device=query.device,
                dtype=query.dtype,
            )
            query, key = apply_rotary_pos_emb(query, key, cos, sin, self.rotary_dim)

        # Casual self attention: (B, nH, T, Hsz) x (B, nH, Hsz, T) -> (B, nH, T, T)
        # is_causal=True applies the causal mask automatically (The triangular mask)
        y = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)

        # Output projection
        y = self.residual_dropout(self.c_projection(y))
        return y
