import torch
import torch.nn as nn

from ..config import GPTConfig
from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.n_embedding = config.hidden_size
        self.n_head = config.num_attention_heads
        self.dropout = config.dropout

        # Key, Query, Value projections for all heads
        self.c_attention = nn.Linear(self.n_embedding, 3 * self.n_embedding, bias=config.bias)

        # Output projection
        self.c_projection = nn.Linear(self.n_embedding, self.n_embedding, bias=config.bias)

        # Regularization
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        
        # Flash attention
        self.flash_attention = hasattr(nn.functional, 'scaled_dot_product_attention')

        self.use_rotary = config.use_rotary_embeddings
        if self.use_rotary:
            self.rotary_dim = config.rotary_dim if config.rotary_dim is not None else config.head_dim
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
        batch_size, sequence_length, embedding_dim = x.size()

        # Calculate query, key and values for all heads and move head forward to be the batch dimension
        query, key, value = self.c_attention(x).split(self.n_embedding, dim=2)
        key = key.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)
        query = query.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)
        value = value.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (batch_size, n_head, sequence_length, head_dim)

        if self.use_rotary:
            cos, sin = self.rotary_emb(
                sequence_length,
                device=query.device,
                dtype=query.dtype,
            )
            query, key = apply_rotary_pos_emb(query, key, cos, sin, self.rotary_dim)

        # Casual self attention: (B, nH, T, Hsz) x (B, nH, Hsz, T) -> (B, nH, T, T)
        # is_causal=True applies the causal mask automatically (The triangular mask)
        y = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim) # re-assemble all head outputs side by side

        # Output projection
        y = self.residual_dropout(self.c_projection(y))
        return y