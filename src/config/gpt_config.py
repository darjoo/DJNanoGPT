from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    # Padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50257 
    n_layer: int = 6
    n_head: int = 6
    n_embedding: int = 384
    dropout: float = 0.1
    bias: bool = True