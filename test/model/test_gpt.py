import torch
import torch.nn as nn
import pytest
from unittest.mock import patch

from src.model import GPT
from src.config import GPTConfig


class TestGPT:
    """Test suite for the GPT model."""
    
    @pytest.fixture
    def config(self):
        """Create a minimal GPT config for testing."""
        return GPTConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embedding=64,
            dropout=0.1,
            bias=True,
            device='cpu'
        )
    
    @pytest.fixture
    def small_config(self):
        """Create an even smaller config for faster tests."""
        return GPTConfig(
            block_size=32,
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embedding=16,
            dropout=0.0,
            bias=False,
            device='cpu'
        )
    
    def test_gpt_init_with_valid_config(self, config):
        """Test GPT initialization with valid configuration."""
        with patch('builtins.print'):  # Suppress print statements
            model = GPT(config)
        
        assert isinstance(model, nn.Module)
        assert model.config == config
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'lm_head')
        
        # Check transformer components
        assert hasattr(model.transformer, 'word_token_embed')
        assert hasattr(model.transformer, 'word_position_embed')
        assert hasattr(model.transformer, 'drop')
        assert hasattr(model.transformer, 'hidden')
        assert hasattr(model.transformer, 'ln_final')
        
        # Check embedding dimensions
        assert model.transformer.word_token_embed.num_embeddings == config.vocab_size
        assert model.transformer.word_token_embed.embedding_dim == config.n_embedding
        assert model.transformer.word_position_embed.num_embeddings == config.block_size
        assert model.transformer.word_position_embed.embedding_dim == config.n_embedding
        
        # Check number of transformer blocks
        assert len(model.transformer.hidden) == config.n_layer
        
        # Check weight tying
        assert model.transformer.word_token_embed.weight is model.lm_head.weight
    
    def test_gpt_init_without_vocab_size(self):
        """Test that GPT initialization fails without vocab_size."""
        config = GPTConfig(vocab_size=None, block_size=128)
        with pytest.raises(AssertionError, match="vocab_size must be specified"):
            GPT(config)
    
    def test_gpt_init_without_block_size(self):
        """Test that GPT initialization fails without block_size."""
        config = GPTConfig(vocab_size=1000, block_size=None)
        with pytest.raises(AssertionError, match="block_size must be specified"):
            GPT(config)
    
    def test_get_num_params(self, small_config):
        """Test parameter counting functionality."""
        with patch('builtins.print'):
            model = GPT(small_config)
        
        total_params = model.get_num_params()
        non_embedding_params = model.get_num_params(non_embedding=True)
        
        assert isinstance(total_params, int)
        assert isinstance(non_embedding_params, int)
        assert total_params > non_embedding_params
        assert total_params > 0
        assert non_embedding_params > 0
        
        # The difference should be the position embedding parameters
        pos_embed_params = model.transformer.word_position_embed.weight.numel()
        assert total_params - non_embedding_params == pos_embed_params
    
    def test_forward_training_mode(self, small_config):
        """Test forward pass with targets (training mode)."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()  # Set to eval mode to avoid randomness from dropout
        
        batch_size, seq_len = 2, 16
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(idx, targets)
        
        # Check output shapes
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_forward_inference_mode(self, small_config):
        """Test forward pass without targets (inference mode)."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()
        
        batch_size, seq_len = 2, 16
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(idx)
        
        # Check output shapes - should only return logits for last position
        assert logits.shape == (batch_size, 1, small_config.vocab_size)
        assert loss is None
        assert torch.isfinite(logits).all()
    
    def test_forward_sequence_too_long(self, small_config):
        """Test that forward pass fails with sequence longer than block_size."""
        with patch('builtins.print'):
            model = GPT(small_config)
        
        batch_size = 2
        seq_len = small_config.block_size + 1  # Longer than block_size
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        with pytest.raises(AssertionError, match="Cannot forward sequence of length"):
            model(idx)
    
    def test_crop_block_size(self, config):
        """Test cropping the model's block size."""
        with patch('builtins.print'):
            model = GPT(config)
        
        original_block_size = config.block_size
        new_block_size = original_block_size // 2
        
        model.crop_block_size(new_block_size)
        
        assert model.config.block_size == new_block_size
        assert model.transformer.word_position_embed.shape[0] == new_block_size
    
    def test_crop_block_size_invalid(self, config):
        """Test that cropping to larger block size fails."""
        with patch('builtins.print'):
            model = GPT(config)
        
        invalid_block_size = config.block_size + 1
        
        with pytest.raises(AssertionError, match="Cannot crop to block size"):
            model.crop_block_size(invalid_block_size)
    
    def test_configure_optimizers(self, small_config):
        """Test optimizer configuration."""
        with patch('builtins.print'):
            model = GPT(small_config)
        
        weight_decay = 0.1
        learning_rate = 1e-4
        betas = (0.9, 0.95)
        device_type = 'cpu'
        
        optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
        
        assert hasattr(optimizer, 'param_groups')
        assert len(optimizer.param_groups) == 2  # decay and no-decay groups
        
        # Check that parameters are properly grouped
        decay_group = optimizer.param_groups[0]
        nodecay_group = optimizer.param_groups[1]
        
        assert decay_group['weight_decay'] == weight_decay
        assert nodecay_group['weight_decay'] == 0.0
        assert decay_group['lr'] == learning_rate
        assert nodecay_group['lr'] == learning_rate
    
    def test_estimate_mfu(self, small_config):
        """Test model FLOPS utilization estimation."""
        with patch('builtins.print'):
            model = GPT(small_config)
        
        fwdbwd_per_iter = 4
        dt = 0.1  # 100ms per iteration
        
        mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
        
        assert isinstance(mfu, float)
        assert mfu >= 0
        assert mfu <= 1  # MFU should be between 0 and 1
    
    def test_generate_basic(self, small_config):
        """Test basic text generation."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()
        
        # Start with a simple context
        batch_size = 1
        context_len = 5
        max_new_tokens = 10
        
        idx = torch.randint(0, small_config.vocab_size, (batch_size, context_len))
        
        generated = model.generate(idx, max_new_tokens)
        
        assert generated.shape == (batch_size, context_len + max_new_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < small_config.vocab_size)
        
        # Check that original context is preserved
        assert torch.equal(generated[:, :context_len], idx)
    
    def test_generate_with_temperature(self, small_config):
        """Test generation with different temperature values."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()
        
        idx = torch.randint(0, small_config.vocab_size, (1, 5))
        max_new_tokens = 5
        
        # Test different temperatures
        for temp in [0.1, 1.0, 2.0]:
            generated = model.generate(idx, max_new_tokens, temperature=temp)
            assert generated.shape == (1, 10)
            assert torch.all(generated >= 0)
            assert torch.all(generated < small_config.vocab_size)
    
    def test_generate_with_top_k(self, small_config):
        """Test generation with top-k sampling."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()
        
        idx = torch.randint(0, small_config.vocab_size, (1, 5))
        max_new_tokens = 5
        
        # Test with different top_k values
        for top_k in [1, 5, 10]:
            generated = model.generate(idx, max_new_tokens, top_k=top_k)
            assert generated.shape == (1, 10)
            assert torch.all(generated >= 0)
            assert torch.all(generated < small_config.vocab_size)
    
    def test_generate_long_context(self, small_config):
        """Test generation when context exceeds block size."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.eval()
        
        # Create context longer than block_size
        long_context_len = small_config.block_size + 10
        idx = torch.randint(0, small_config.vocab_size, (1, long_context_len))
        
        generated = model.generate(idx, max_new_tokens=5)
        
        # Should crop to block_size and then generate
        assert generated.shape == (1, long_context_len + 5)
        assert torch.all(generated >= 0)
        assert torch.all(generated < small_config.vocab_size)
    
    def test_model_deterministic_with_seed(self, small_config):
        """Test that model produces deterministic outputs with fixed seed."""
        torch.manual_seed(42)
        with patch('builtins.print'):
            model1 = GPT(small_config)
        model1.eval()
        
        torch.manual_seed(42)
        with patch('builtins.print'):
            model2 = GPT(small_config)
        model2.eval()
        
        idx = torch.randint(0, small_config.vocab_size, (1, 10))
        
        # Both models should produce the same output
        logits1, _ = model1(idx)
        logits2, _ = model2(idx)
        
        assert torch.allclose(logits1, logits2, atol=1e-6)
    
    def test_gradient_flow(self, small_config):
        """Test that gradients flow properly through the model."""
        with patch('builtins.print'):
            model = GPT(small_config)
        model.train()
        
        batch_size, seq_len = 2, 10
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(idx, targets)
        loss.backward()
        
        # Check that gradients exist for key parameters
        assert model.lm_head.weight.grad is not None
        assert model.transformer.word_token_embed.weight.grad is not None
        assert model.transformer.ln_final.weight.grad is not None
        
        # Check that gradients are not zero or NaN
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient in {name}"


# Additional integration tests
class TestGPTIntegration:
    """Integration tests for GPT model."""
    
    def test_training_step_integration(self):
        """Test a complete training step."""
        config = GPTConfig(
            block_size=64,
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embedding=32,
            dropout=0.0,
            bias=False,
            device='cpu'
        )
        
        with patch('builtins.print'):
            model = GPT(config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Simulate training data
        batch_size, seq_len = 4, 32
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        model.train()
        optimizer.zero_grad()
        
        logits, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
    
    def test_inference_generation_integration(self):
        """Test a complete inference generation."""
        config = GPTConfig(
            block_size=64,
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embedding=32,
            dropout=0.0,
            bias=False,
            device='cpu'
        )
        
        with patch('builtins.print'):
            model = GPT(config)
        model.eval()
        
        # Start with a prompt
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
        
        assert generated.shape == (1, 30)
        assert torch.equal(generated[:, :10], prompt)
        assert torch.all(generated >= 0)
        assert torch.all(generated < config.vocab_size)