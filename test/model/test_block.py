import torch
import pytest
from src.config import GPTConfig
from src.model import Block

def test_block_init():
    """Test Block initialization with default config"""
    config = GPTConfig()
    block = Block(config)
    
    # Check that all components are initialized
    assert hasattr(block, 'ln_1')
    assert hasattr(block, 'attention')
    assert hasattr(block, 'ln_2')
    assert hasattr(block, 'mlp')
    
    # Check that layer norms have correct dimensions
    assert block.ln_1.weight.shape == (config.hidden_size,)
    assert block.ln_2.weight.shape == (config.hidden_size,)

def test_block_init_custom_config():
    """Test Block initialization with custom config"""
    config = GPTConfig(
        hidden_size=512,
        num_attention_heads=8,
        bias=False
    )
    block = Block(config)
    
    # Check that layer norms have correct dimensions
    assert block.ln_1.weight.shape == (config.hidden_size,)
    assert block.ln_2.weight.shape == (config.hidden_size,)

def test_block_forward_shape():
    """Test that Block forward pass maintains correct tensor shapes"""
    config = GPTConfig(
        max_position_embeddings=128,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4
    )
    block = Block(config)
    
    batch_size = 2
    seq_len = 64
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = block.forward(x)
    
    # Check output shape matches input shape
    assert output.shape == x.shape
    assert output.shape == (batch_size, seq_len, config.hidden_size)

def test_block_forward_values():
    """Test that Block forward pass produces reasonable values"""
    config = GPTConfig(
        max_position_embeddings=64,
        hidden_size=128,
        num_attention_heads=4,
        dropout=0.0  # Disable dropout for deterministic testing
    )
    block = Block(config)
    
    batch_size = 1
    seq_len = 32
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = block.forward(x)
    
    # Check that output is finite
    assert torch.isfinite(output).all()
    
    # Check that output is different from input (block should transform the input)
    assert not torch.allclose(output, x, atol=1e-6)

def test_block_residual_connections():
    """Test that residual connections work properly"""
    config = GPTConfig(
        max_position_embeddings=32,
        hidden_size=64,
        num_attention_heads=2,
        dropout=0.0
    )
    block = Block(config)
    
    batch_size = 1
    seq_len = 16
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Get intermediate results to verify residual connections
    ln1_out = block.ln_1(x)
    attn_out = block.attention(ln1_out)
    residual1 = x + attn_out
    
    ln2_out = block.ln_2(residual1)
    mlp_out = block.mlp(ln2_out)
    final_out = residual1 + mlp_out
    
    # Compare with actual forward pass
    block_out = block.forward(x)
    
    assert torch.allclose(block_out, final_out, atol=1e-6)

def test_block_gradient_flow():
    """Test that gradients flow through the block properly"""
    config = GPTConfig(
        max_position_embeddings=32,
        hidden_size=64,
        num_attention_heads=2,
        dropout=0.0
    )
    block = Block(config)
    
    batch_size = 1
    seq_len = 16
    
    # Create input tensor with gradient tracking
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    
    # Forward pass
    output = block.forward(x)
    
    # Compute a simple loss
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are computed for input
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    
    # Check that model parameters have gradients
    for param in block.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

def test_block_eval_mode():
    """Test Block in evaluation mode"""
    config = GPTConfig(
        max_position_embeddings=32,
        hidden_size=64,
        num_attention_heads=2,
        dropout=0.1  # Enable dropout to test eval mode
    )
    block = Block(config)
    
    batch_size = 1
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test in training mode
    block.train()
    train_output = block.forward(x)
    
    # Test in eval mode
    block.eval()
    eval_output = block.forward(x)
    
    # Outputs should be finite in both modes
    assert torch.isfinite(train_output).all()
    assert torch.isfinite(eval_output).all()

def test_block_device_placement():
    """Test that Block can be moved to different devices"""
    config = GPTConfig(
        hidden_size=64,
        num_attention_heads=2
    )
    block = Block(config)
    
    # Test CPU placement
    block_cpu = block.to('cpu')
    x_cpu = torch.randn(1, 8, config.hidden_size)
    output_cpu = block_cpu.forward(x_cpu)
    assert output_cpu.device.type == 'cpu'
    
    # Test CUDA placement if available
    if torch.cuda.is_available():
        block_cuda = block.to('cuda')
        x_cuda = torch.randn(1, 8, config.hidden_size).to('cuda')
        output_cuda = block_cuda.forward(x_cuda)
        assert output_cuda.device.type == 'cuda'

def test_block_different_sequence_lengths():
    """Test Block with different sequence lengths"""
    config = GPTConfig(
        max_position_embeddings=512,  # Large max_position_embeddings
        hidden_size=128,
        num_attention_heads=4
    )
    block = Block(config)
    
    batch_size = 2
    
    # Test various sequence lengths
    for seq_len in [1, 16, 64, 256]:
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = block.forward(x)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert torch.isfinite(output).all()

def test_block_parameter_count():
    """Test that Block has expected number of parameters"""
    config = GPTConfig(
        hidden_size=768,
        num_attention_heads=12
    )
    block = Block(config)
    
    total_params = sum(p.numel() for p in block.parameters())
    
    # Should have parameters from:
    # - 2 LayerNorms
    # - 1 CausalSelfAttention
    # - 1 MLP
    assert total_params > 0
    
    # Check that all parameters are trainable by default
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    assert trainable_params == total_params