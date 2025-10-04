import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from ..config import GPTConfig
from . import Block

class GPT(nn.Module):
    """
    GPT Language Model
    """
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.vocab_size is not None, "vocab_size must be specified in the config"
        assert config.block_size is not None, "block_size must be specified in the config"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            word_token_embed = nn.Embedding(config.vocab_size, config.n_embedding),
            word_position_embed = nn.Embedding(config.block_size, config.n_embedding),
            drop = nn.Dropout(config.dropout),
            hidden = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]),
            ln_final = nn.LayerNorm(config.n_embedding, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embedding, config.vocab_size, bias=False)
        # Weight tying, shares the weight matrix between word_token_embed and lm_head
        self.transformer.word_token_embed.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / ((2 * config.n_layer) ** 0.5))

        # Number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False):
        """
        Get the number of parameters in the model.
        Args:
            non_embedding (bool): If True, excludes the parameters of the positional embedding layers. The word token embeddings are still included as they're used as weights in the final layer.

        Returns:
            int: Number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= (self.transformer.word_position_embed.weight.numel())
        return n_params
    
    def forward(self, idx, targets=None):
        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, the block size is {self.config.block_size}."
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Forward the GPT model
        token_embeddings = self.transformer.word_token_embed(idx)  # Shape (batch_size, seq_len, n_embedding)
        position_embeddings = self.transformer.word_position_embed(pos)  # Shape (seq_len, n_embedding)
        combined_embeddings = token_embeddings + position_embeddings  # Shape (batch_size, seq_len, n_embedding)
        x = self.transformer.drop(combined_embeddings) # Apply dropout after adding token and position embeddings
        x = self.transformer.hidden(x)
        x = self.transformer.ln_final(x) # Final layer norm

        if targets is not None:
            # If targets are provided, compute the loss
            logits = self.lm_head(x) # Shape (batch_size, seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # If no targets, return the logits for the last time step
            # Inference-time mini optimization: only forward the last position
            logits = self.lm_head(x[:, [-1], :]) # Shape (batch_size, 1, vocab_size)
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        """
        Crop the model's block size to the specified value.
        A pretrained model can be used with a smaller block size than it was trained with.
        Args:
            block_size (int): The new block size to crop to.
        """
        assert block_size <= self.config.block_size, f"Cannot crop to block size {block_size}, the model's block size is {self.config.block_size}."
        self.config.block_size = block_size
        self.transformer.word_position_embed = nn.Parameter(self.transformer.word_position_embed.weight[:block_size])
        for block in self.transformer.hidden:
            if hasattr(block.attention, 'bias'):
                block.attention.bias = block.attention.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out the parameters that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optimizer groups. Any parameters that is 2D (e.g. a weight matrix) and not a bias or LayerNorm/Embedding parameter will be weight decayed
        # Fx all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optimizer_groups = [
            {
                'params': decay_params, 
                'weight_decay': weight_decay
            },
            {
                'params': nodecay_params,
                'weight_decay': 0.0
            }
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters.")
        print(f"Number of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters.")

        # Create AdamW optimizer, use fused AdamW on CUDA if possible
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using {'fused' if use_fused else 'standard'} AdamW optimizer")

        return optimizer
    
    def estimate_mfu(self,  fwdbwd_per_iter, dt):
        """
        Estimate the model flops utilization (MFU) given the number of forward+backward passes per iteration
        """
        # First estimation of the number of flops per iteration
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        # Layers, heads, head size, context length
        L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embedding // self.config.n_head, self.config.block_size

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express flops throughput as ratio of 5090 bfloat16 peak flops
        flops_archieved = flops_per_iter * (1.0 / dt) # in flops/sec
        flops_peak = 210e12 # 5090 GPU bfloat16 peak flops is 210 TFLOPS
        mfu = flops_archieved / flops_peak
        return mfu
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens from the model given a context.
        """
        for _ in range(max_new_tokens):
            # If the context is longer than the block size, crop it to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the next token
            logits, _ = self(idx_cond)

            # Focus only on the last time step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # Safety check
                logits[logits < v[:, [-1]]] = -float('Inf') # Set all logits not in top k to -infinity
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1) # Shape (batch_size, vocab_size)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # Shape (batch_size, 1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1) # Shape (batch_size, current_seq_len + 1)

        return idx