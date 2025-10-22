import torch
import pytest
from src.model import MLP
from src.config import GPTConfig


def test_mlp_init():
    """Test MLP initialization with default config."""
    config = GPTConfig()
    mlp = MLP(config)
    
    # Check layer dimensions
    assert mlp.linear.in_features == config.hidden_size
    assert mlp.linear.out_features == 4 * config.hidden_size
    assert mlp.projection.in_features == 4 * config.hidden_size
    assert mlp.projection.out_features == config.hidden_size
    
    # Check bias configuration
    assert mlp.linear.bias is not None if config.bias else mlp.linear.bias is None
    assert mlp.projection.bias is not None if config.bias else mlp.projection.bias is None


def test_mlp_init_no_bias():
    """Test MLP initialization with bias disabled."""
    config = GPTConfig(bias=False)
    mlp = MLP(config)
    
    assert mlp.linear.bias is None
    assert mlp.projection.bias is None


def test_mlp_forward_shape():
    """Test that MLP forward pass preserves input shape."""
    config = GPTConfig(hidden_size=768, dropout=0.0)
    mlp = MLP(config)
    
    # Test different batch sizes and sequence lengths
    test_shapes = [
        (1, 768),           # Single token
        (32, 768),          # Batch of tokens
        (8, 1024, 768),     # Batch with sequence length
        (4, 512, 768),      # Different sequence length
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape)
        output = mlp(x)
        assert output.shape == x.shape, f"Shape mismatch for input {shape}"


def test_mlp_forward_values():
    """Test that MLP forward pass produces reasonable outputs."""
    config = GPTConfig(hidden_size=64, dropout=0.0)  # Smaller for testing
    mlp = MLP(config)
    mlp.eval()  # Disable dropout for deterministic testing
    
    x = torch.randn(2, 64)
    output = mlp(x)
    
    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    # Check output is not all zeros (network should transform input)
    assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
    
    # Check output changes with different inputs
    x2 = torch.randn(2, 64)
    output2 = mlp(x2)
    assert not torch.allclose(output, output2), "Different inputs produce same output"


def test_mlp_deterministic():
    """Test that MLP produces deterministic outputs when dropout is disabled."""
    config = GPTConfig(hidden_size=64, dropout=0.0)
    mlp = MLP(config)
    mlp.eval()
    
    x = torch.randn(2, 64)
    
    # Multiple forward passes should give same result
    output1 = mlp(x)
    output2 = mlp(x)
    
    assert torch.allclose(output1, output2), "MLP not deterministic with dropout disabled"


def test_mlp_dropout_effect():
    """Test that dropout affects training mode but not eval mode."""
    config = GPTConfig(hidden_size=64, dropout=0.5)
    mlp = MLP(config)
    
    x = torch.randn(2, 64)
    
    # In training mode, outputs should vary due to dropout
    mlp.train()
    outputs_train = [mlp(x) for _ in range(5)]
    
    # Check that not all outputs are identical (dropout should cause variation)
    all_same = all(torch.allclose(outputs_train[0], out) for out in outputs_train[1:])
    assert not all_same, "Dropout not working in training mode"
    
    # In eval mode, outputs should be deterministic
    mlp.eval()
    output1 = mlp(x)
    output2 = mlp(x)
    assert torch.allclose(output1, output2), "MLP not deterministic in eval mode"


def test_mlp_gradient_flow():
    """Test that gradients flow properly through MLP."""
    config = GPTConfig(hidden_size=64, dropout=0.0)
    mlp = MLP(config)
    
    x = torch.randn(2, 64, requires_grad=True)
    output = mlp(x)
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for parameter {name}"
    
    # Check that input gradients exist
    assert x.grad is not None, "No gradient for input"


@pytest.mark.parametrize("n_embedding", [64, 128, 256, 768])
def test_mlp_different_embedding_sizes(n_embedding):
    """Test MLP with different embedding dimensions."""
    config = GPTConfig(hidden_size=n_embedding, dropout=0.0)
    mlp = MLP(config)
    
    x = torch.randn(4, n_embedding)
    output = mlp(x)
    
    assert output.shape == (4, n_embedding)
    assert torch.isfinite(output).all()


def test_mlp_gelu_activation():
    """Test that GELU activation is applied correctly."""
    config = GPTConfig(n_embedding=64, dropout=0.0)
    mlp = MLP(config)
    
    # Test that activation function is GELU
    assert isinstance(mlp.gelu, torch.nn.GELU)
    
    # Test GELU properties on a simple input
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    gelu_output = mlp.gelu(x)
    
    # GELU should be approximately 0 for negative inputs, positive for positive inputs
    assert gelu_output[0] < 0.1  # GELU(-2) ≈ 0
    assert gelu_output[2] == 0.0  # GELU(0) = 0
    assert gelu_output[4] > 1.9   # GELU(2) ≈ 2


def test_mlp_expansion_factor():
    """Test that MLP uses 4x expansion factor as mentioned in comments."""
    config = GPTConfig(hidden_size=128)
    mlp = MLP(config)
    
    # Check that hidden dimension is 4x the embedding dimension
    hidden_dim = mlp.linear.out_features
    assert hidden_dim == 4 * config.hidden_size, "MLP should use 4x expansion factor"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mlp_cuda():
    """Test MLP on CUDA device."""
    config = GPTConfig(hidden_size=64, dropout=0.0)
    mlp = MLP(config).cuda()
    
    x = torch.randn(2, 64).cuda()
    output = mlp(x)
    
    assert output.device.type == 'cuda'
    assert output.shape == x.shape
    assert torch.isfinite(output).all()