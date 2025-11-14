import math
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..config import GPTConfig
from . import Block


class GPT(PreTrainedModel):
    """GPT language model."""

    config_class = GPTConfig
    _tied_weights_keys: ClassVar[list[str]] = ["lm_head.weight"]

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.config = config

        transformer_modules = {
            "word_token_embed": nn.Embedding(config.vocab_size, config.hidden_size),
            "drop": nn.Dropout(config.dropout),
            "hidden": nn.Sequential(*[Block(config) for _ in range(config.num_hidden_layers)]),
            "ln_final": nn.LayerNorm(config.hidden_size, bias=config.bias),
        }
        if not config.use_rotary_embeddings:
            transformer_modules["word_position_embed"] = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )

        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding lookup table."""
        return self.transformer.word_token_embed

    def set_input_embeddings(self, new_embeddings):
        """Replace the token embedding layer with ``new_embeddings``."""
        self.transformer.word_token_embed = new_embeddings

    def get_output_embeddings(self):
        """Return the linear head used to project hidden states."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Replace the output projection layer with ``new_embeddings``."""
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie the input embedding weights to the output projection.

        The embedding layer shares the lm_head's weight tensor.
        """
        self.transformer.word_token_embed.weight = self.lm_head.weight

    def _init_weights(self, module):
        """Initialize module weights using the configured initializer."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("c_projection.weight") or name.endswith("projection.weight"):
                param.data.normal_(
                    mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
                )

    def get_num_params(self, non_embedding: bool = False):
        """Return the total parameter count.

        Args:
            non_embedding (bool): If ``True``, exclude positional embedding parameters
                while keeping token embeddings since they double as lm_head weights.

        Returns:
            int: Number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "word_position_embed" in self.transformer:
            n_params -= self.transformer.word_position_embed.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """Forward pass through the model.

        Args:
            idx (torch.LongTensor): Token ids of shape ``(batch, seq_len)``.
            targets (torch.LongTensor | None): Optional labels for language-model
                loss computation, matching ``idx`` shape.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Logits over the vocabulary and
            the cross-entropy loss when ``targets`` are provided.
        """
        device = idx.device
        _, seq_len = idx.size()  # (batch_size, sequence_length)
        assert seq_len <= self.config.max_position_embeddings, (
            f"Cannot forward sequence of length {seq_len}, the block size is {self.config.max_position_embeddings}."
        )
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Forward the GPT model
        token_embeddings = self.transformer.word_token_embed(idx)  # Shape (batch_size, seq_len, n_embedding)
        if "word_position_embed" in self.transformer:
            position_embeddings = self.transformer.word_position_embed(pos)  # Shape (seq_len, n_embedding)
            token_embeddings = token_embeddings + position_embeddings
        x = self.transformer.drop(token_embeddings)  # Apply dropout after embedding stage
        x = self.transformer.hidden(x)
        x = self.transformer.ln_final(x)  # Final layer norm

        if targets is not None:
            # If targets are provided, compute the loss
            logits = self.lm_head(x)  # Shape (batch_size, seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # If no targets, return the logits for the last time step
            # Inference-time mini optimization: only forward the last position
            logits = self.lm_head(x[:, [-1], :])  # Shape (batch_size, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens from the model given a context."""
        for _ in range(max_new_tokens):
            # If the context is longer than the block size, crop it to the last block_size tokens
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_position_embeddings
                else idx[:, -self.config.max_position_embeddings :]
            )

            # Forward the model to get the logits for the next token
            logits, _ = self(idx_cond)

            # Focus only on the last time step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # Safety check
                logits[logits < v[:, [-1]]] = -float("Inf")  # Set all logits not in top k to -infinity

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)  # Shape (batch_size, vocab_size)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # Shape (batch_size, 1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1)  # Shape (batch_size, current_seq_len + 1)

        return idx
