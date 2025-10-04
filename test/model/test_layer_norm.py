import torch
from src.model import LayerNorm

def test_layernorm_init_and_forward():
    n_dimensions = 4
    ln = LayerNorm(n_dimensions, bias=True).to('cuda')
    x = torch.randn(2, n_dimensions).to('cuda')
    out = ln(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    