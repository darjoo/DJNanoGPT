from dataclasses import dataclass, asdict

@dataclass
class GPTConfig:
    block_size: int = 256 # Context/Sequence length
    # Padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50257 
    n_layer: int = 6
    n_head: int = 6
    n_embedding: int = 384
    dropout: float = 0.1
    bias: bool = True

    def get(self, key, default=None):
        """Mimic dict.get() so libraries expecting a dict config work."""
        return getattr(self, key, default)

    def to_dict(self):
        """For full compatibility with HF-style configs."""
        return asdict(self)