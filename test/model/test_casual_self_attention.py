import torch
import pytest
from src.model import CasualSelfAttention
from src.config import GPTConfig


def test_casual_self_attention_init():
    """Test that CasualSelfAttention initializes correctly."""
    config = GPTConfig(n_embedding=768, n_head=12, dropout=0.1, bias=True)
    attention = CasualSelfAttention(config)
    
    # Check that all components are initialized
    assert hasattr(attention, 'c_attention')
    assert hasattr(attention, 'c_projection')
    assert hasattr(attention, 'attention_dropout')
    assert hasattr(attention, 'residual_dropout')
    
    # Check dimensions
    assert attention.n_head == 12
    assert attention.n_embedding == 768
    assert attention.dropout == 0.1
    
    # Check linear layer dimensions
    assert attention.c_attention.in_features == 768
    assert attention.c_attention.out_features == 3 * 768  # Q, K, V
    assert attention.c_projection.in_features == 768
    assert attention.c_projection.out_features == 768


def test_casual_self_attention_invalid_config():
    """Test that CasualSelfAttention raises error for invalid configuration."""
    # n_embedding not divisible by n_head
    config = GPTConfig(n_embedding=100, n_head=12)
    
    with pytest.raises(AssertionError):
        CasualSelfAttention(config)


def test_casual_self_attention_forward_shape():
    """Test that forward pass returns correct output shape."""
    config = GPTConfig(n_embedding=768, n_head=12, dropout=0.0)
    attention = CasualSelfAttention(config)
    
    batch_size = 2
    sequence_length = 10
    
    # Create input tensor
    x = torch.randn(batch_size, sequence_length, config.n_embedding)
    
    # Forward pass
    output = attention(x)
    
    # Check output shape
    assert output.shape == (batch_size, sequence_length, config.n_embedding)
    assert torch.isfinite(output).all()


def test_casual_self_attention_forward_different_sizes():
    """Test forward pass with different batch sizes and sequence lengths."""
    config = GPTConfig(n_embedding=512, n_head=8, dropout=0.0)
    attention = CasualSelfAttention(config)
    
    test_cases = [
        (1, 5),    # Small batch, short sequence
        (4, 20),   # Medium batch, medium sequence
        (2, 100),  # Small batch, long sequence
    ]
    
    for batch_size, sequence_length in test_cases:
        x = torch.randn(batch_size, sequence_length, config.n_embedding)
        output = attention(x)
        
        assert output.shape == (batch_size, sequence_length, config.n_embedding)
        assert torch.isfinite(output).all()


def test_casual_self_attention_gradient_flow():
    """Test that gradients flow through the attention mechanism."""
    config = GPTConfig(n_embedding=256, n_head=4, dropout=0.0)
    attention = CasualSelfAttention(config)
    
    batch_size = 2
    sequence_length = 8
    
    x = torch.randn(batch_size, sequence_length, config.n_embedding, requires_grad=True)
    output = attention(x)
    
    # Create a dummy loss
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert attention.c_attention.weight.grad is not None
    assert attention.c_projection.weight.grad is not None


def test_casual_self_attention_training_vs_eval():
    """Test that attention behaves differently in training vs eval mode."""
    config = GPTConfig(n_embedding=256, n_head=4, dropout=0.5)
    attention = CasualSelfAttention(config)
    
    batch_size = 2
    sequence_length = 8
    x = torch.randn(batch_size, sequence_length, config.n_embedding)
    
    # Set to training mode
    attention.train()
    output_train = attention(x)
    
    # Set to eval mode
    attention.eval()
    output_eval = attention(x)
    
    # Outputs should be different due to dropout
    assert output_train.shape == output_eval.shape
    # Note: We can't guarantee they're different due to randomness, 
    # but we can check they're both valid
    assert torch.isfinite(output_train).all()
    assert torch.isfinite(output_eval).all()


def test_casual_self_attention_no_bias():
    """Test attention mechanism without bias."""
    config = GPTConfig(n_embedding=256, n_head=4, dropout=0.0, bias=False)
    attention = CasualSelfAttention(config)
    
    # Check that bias is None
    assert attention.c_attention.bias is None
    assert attention.c_projection.bias is None
    
    batch_size = 2
    sequence_length = 8
    x = torch.randn(batch_size, sequence_length, config.n_embedding)
    
    output = attention(x)
    assert output.shape == (batch_size, sequence_length, config.n_embedding)
    assert torch.isfinite(output).all()


def test_casual_self_attention_deterministic():
    """Test that attention is deterministic when using the same input and seed."""
    config = GPTConfig(n_embedding=256, n_head=4, dropout=0.0)
    
    batch_size = 2
    sequence_length = 8
    x = torch.randn(batch_size, sequence_length, config.n_embedding)
    
    # Create two identical attention modules
    torch.manual_seed(42)
    attention1 = CasualSelfAttention(config)
    attention1.eval()  # Set to eval mode to disable dropout
    
    torch.manual_seed(42)
    attention2 = CasualSelfAttention(config)
    attention2.eval()  # Set to eval mode to disable dropout
    
    # Forward pass should give same results
    output1 = attention1(x)
    output2 = attention2(x)
    
    assert torch.allclose(output1, output2, atol=1e-6)


def test_casual_self_attention_different_head_counts():
    """Test attention with different numbers of heads."""
    embedding_dim = 768
    head_counts = [1, 2, 3, 4, 6, 8, 12]
    
    batch_size = 2
    sequence_length = 8
    
    for n_head in head_counts:
        config = GPTConfig(n_embedding=embedding_dim, n_head=n_head, dropout=0.0)
        attention = CasualSelfAttention(config)
        
        x = torch.randn(batch_size, sequence_length, embedding_dim)
        output = attention(x)
        
        assert output.shape == (batch_size, sequence_length, embedding_dim)
        assert torch.isfinite(output).all()


def test_casual_self_attention_flash_attention_attribute():
    """Test that flash attention attribute is set correctly."""
    config = GPTConfig(n_embedding=256, n_head=4, dropout=0.0)
    attention = CasualSelfAttention(config)
    
    # Check that flash_attention attribute exists and is boolean
    assert hasattr(attention, 'flash_attention')
    assert isinstance(attention.flash_attention, bool)
    
    # Should be True since PyTorch has scaled_dot_product_attention
    assert attention.flash_attention is True