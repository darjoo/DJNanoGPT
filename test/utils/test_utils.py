"""Tests for utility functions."""
import os
import tempfile
import torch
import pytest

from src.config import GPTConfig
from src.model import GPT
from src.utils import load_checkpoint, save_checkpoint, get_system_info, get_memory_usage


class TestCheckpointUtils:
    """Test checkpoint loading and saving utilities."""
    
    @pytest.fixture
    def device(self):
        """Get available device."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @pytest.fixture
    def small_model(self):
        """Create a small GPT model for testing."""
        config = GPTConfig(
            max_position_embeddings=32,
            vocab_size=100,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=16,
            dropout=0.0,
            bias=False
        )
        return GPT(config)
    
    def test_save_and_load_checkpoint(self, small_model, device, tmp_path):
        """Test saving and loading a checkpoint."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
        
        model_args = {
            'n_layer': 1,
            'n_head': 2,
            'n_embedding': 16,
            'block_size': 32,
            'vocab_size': 100,
            'bias': False
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(tmp_path, 'test_checkpoint.pt')
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=small_model,
            optimizer=optimizer,
            scaler=scaler,
            model_args=model_args,
            iter_num=100,
            best_val_loss=1.5,
            config=small_model.config,
            run_id='test_run'
        )
        
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        checkpoint, state_dict = load_checkpoint(checkpoint_path, device)
        
        assert checkpoint['iter_num'] == 100
        assert checkpoint['best_val_loss'] == 1.5
        assert checkpoint['wandb_run_id'] == 'test_run'
        assert 'model' in checkpoint
        assert 'optimizer' in checkpoint
        assert 'scaler' in checkpoint
        
        # Verify state dict can be loaded into model
        small_model.load_state_dict(state_dict)
    
    def test_load_nonexistent_checkpoint(self, device):
        """Test loading a checkpoint that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint('nonexistent_checkpoint.pt', device)
    
    def test_checkpoint_handles_compiled_model_keys(self, small_model, device, tmp_path):
        """Test that checkpoint handles _orig_mod. prefix from compiled models."""
        small_model = small_model.to(device)
        
        # Simulate compiled model state dict with _orig_mod. prefix
        state_dict = small_model.state_dict()
        compiled_state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
        
        # Create a checkpoint with compiled keys
        checkpoint_dict = {
            'model': compiled_state_dict,
            'optimizer': torch.optim.AdamW(small_model.parameters()).state_dict(),
            'scaler': torch.amp.GradScaler('cuda').state_dict(),
            'model_args': {},
            'iter_num': 0,
            'best_val_loss': 0.0,
            'config': small_model.config,
            'wandb_run_id': None
        }
        
        checkpoint_path = os.path.join(tmp_path, 'compiled_checkpoint.pt')
        torch.save(checkpoint_dict, checkpoint_path)
        
        # Load checkpoint - should automatically strip prefix
        checkpoint, cleaned_state_dict = load_checkpoint(checkpoint_path, device)
        
        # Verify prefix was removed
        for key in cleaned_state_dict.keys():
            assert not key.startswith('_orig_mod.')
        
        # Verify model can load the cleaned state dict
        small_model.load_state_dict(cleaned_state_dict)


class TestSystemInfoUtils:
    """Test system information utilities."""
    
    @pytest.fixture
    def device(self):
        """Get available device."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_get_system_info(self, device):
        """Test getting system information."""
        info = get_system_info(device)
        
        # Check required fields
        assert 'platform' in info
        assert 'python_version' in info
        assert 'pytorch_version' in info
        assert 'device' in info
        assert 'cpu_count' in info
        assert 'total_ram_gb' in info
        
        # Check types
        assert isinstance(info['platform'], str)
        assert isinstance(info['python_version'], str)
        assert isinstance(info['pytorch_version'], str)
        assert info['device'] == device
        assert isinstance(info['cpu_count'], int)
        assert isinstance(info['total_ram_gb'], float)
        
        # Check CUDA-specific fields
        if device == 'cuda':
            assert 'cuda_version' in info
            assert 'gpu_name' in info
            assert 'gpu_memory_gb' in info
    
    def test_get_memory_usage(self, device):
        """Test getting memory usage statistics."""
        memory = get_memory_usage(device)
        
        # Check CPU memory fields
        assert 'cpu_memory_used_gb' in memory
        assert 'cpu_memory_percent' in memory
        
        # Check types and values
        assert isinstance(memory['cpu_memory_used_gb'], float)
        assert isinstance(memory['cpu_memory_percent'], float)
        assert 0 <= memory['cpu_memory_percent'] <= 100
        
        # Check GPU memory fields if available
        if device == 'cuda':
            assert 'gpu_memory_allocated_gb' in memory
            assert 'gpu_memory_reserved_gb' in memory
            assert 'gpu_memory_percent' in memory
            
            assert isinstance(memory['gpu_memory_allocated_gb'], float)
            assert isinstance(memory['gpu_memory_reserved_gb'], float)
            assert isinstance(memory['gpu_memory_percent'], float)
            assert 0 <= memory['gpu_memory_percent'] <= 100
    
    def test_memory_usage_changes_with_allocation(self, device):
        """Test that memory usage reflects actual allocations."""
        if device != 'cuda':
            pytest.skip("Test requires CUDA")
        
        # Get initial memory in bytes for more precision
        torch.cuda.reset_peak_memory_stats()
        initial_allocated = torch.cuda.memory_allocated()
        
        # Allocate a large tensor (about 4MB)
        large_tensor = torch.randn(1000, 1000, device=device)
        
        # Get memory after allocation
        after_allocated = torch.cuda.memory_allocated()
        
        # Memory usage should increase by at least the tensor size
        # 1000 * 1000 * 4 bytes (float32) = 4MB
        assert after_allocated > initial_allocated, f"Memory should increase: {initial_allocated} -> {after_allocated}"
        
        # Now test the utility function
        memory_stats = get_memory_usage(device)
        assert memory_stats['gpu_memory_allocated_gb'] >= 0
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
