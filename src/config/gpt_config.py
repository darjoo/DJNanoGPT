from transformers import PretrainedConfig


class GPTConfig(PretrainedConfig):
    """Configuration class for GPT model.

    Attributes:
        model_type (str): The model type identifier.
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Dimensionality of the embeddings and hidden states (n_embedding).
        num_hidden_layers (int): Number of transformer blocks (n_layer).
        num_attention_heads (int): Number of attention heads (n_head).
        max_position_embeddings (int): Maximum sequence length (block_size).
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
        bos_token_id (int): Beginning of sequence token ID.
        eos_token_id (int): End of sequence token ID.
        tie_word_embeddings (bool): Whether to tie input and output embeddings.
        initializer_range (float): Standard deviation for weight initialization.
        use_rotary_embeddings (bool): Whether to use rotary position embeddings.
        rotary_dim (int | None): Dimensionality of rotary embeddings.
        rope_theta (float): Base parameter for rotary embeddings.
    """

    model_type = "djgpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 384,  # n_embedding
        num_hidden_layers: int = 6,  # n_layer
        num_attention_heads: int = 6,  # n_head
        max_position_embeddings: int = 256,  # block_size (Context/Sequence length)
        dropout: float = 0.1,
        bias: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        use_rotary_embeddings: bool = False,
        rotary_dim: int | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.dropout = dropout
        self.bias = bias
        self.initializer_range = initializer_range
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
