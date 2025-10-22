from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    model_type = "djgpt"

    def __init__(self, 
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
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.bias = bias
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        super().__init__(
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        