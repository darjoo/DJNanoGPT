from __future__ import annotations

import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
	"""Helper that rotates the last dimension to apply the rotary transform."""
	x1 = x[..., ::2]
	x2 = x[..., 1::2]
	return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(
	query: torch.Tensor,
	key: torch.Tensor,
	cos: torch.Tensor,
	sin: torch.Tensor,
	rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Apply rotary position embeddings to the first rotary_dim of the Q/K tensors."""

	query_rot = query[..., :rotary_dim]
	key_rot = key[..., :rotary_dim]
	cos = cos[..., :rotary_dim]
	sin = sin[..., :rotary_dim]

	query_pass = query[..., rotary_dim:]
	key_pass = key[..., rotary_dim:]

	query_rot = (query_rot * cos) + (rotate_half(query_rot) * sin)
	key_rot = (key_rot * cos) + (rotate_half(key_rot) * sin)

	if query_pass.numel() == 0:
		query_out = query_rot
		key_out = key_rot
	else:
		query_out = torch.cat((query_rot, query_pass), dim=-1)
		key_out = torch.cat((key_rot, key_pass), dim=-1)

	return query_out, key_out


class RotaryEmbedding(nn.Module):
	"""Pre-compute rotary embedding coefficients for a given head dimension."""

	def __init__(
		self,
		dim: int,
		max_position_embeddings: int,
		base: float = 10000.0,
	) -> None:
		super().__init__()
		if dim % 2 != 0:
			raise ValueError("Rotary embedding dimension must be even")

		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)

		self.max_seq_len_cached = 0
		self.register_buffer("cos_cached", torch.empty(0), persistent=False)
		self.register_buffer("sin_cached", torch.empty(0), persistent=False)

		self._set_cos_sin_cache(max_position_embeddings)

	def _set_cos_sin_cache(self, seq_len: int) -> None:
		"""(Re-)build the cached cos/sin tables up to seq_len."""
		positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
		freqs = torch.outer(positions, self.inv_freq)
		emb = torch.cat((freqs, freqs), dim=-1)
		self.cos_cached = emb.cos()[None, None, :, :]
		self.sin_cached = emb.sin()[None, None, :, :]
		self.max_seq_len_cached = seq_len

	def forward(
		self,
		seq_len: int,
		*,
		device: torch.device,
		dtype: torch.dtype,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return cos/sin tables cropped to seq_len on the requested device/dtype."""
		if seq_len > self.max_seq_len_cached:
			self._set_cos_sin_cache(seq_len)

		cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
		sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
		return cos, sin

