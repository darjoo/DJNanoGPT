from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab size, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50257 
    n_layer: int = 12
    n_head: int = 12
    n_embedding: int = 768
    dropout: float = 0.0
    bias: bool = True
    device: str = 'cuda'  # 'cpu', 'cuda', 'mps'