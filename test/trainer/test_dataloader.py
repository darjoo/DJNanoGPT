import numpy as np
import pytest
import torch

from src.trainer.dataloader import DataLoader


def _write_split(tmp_path, split_name: str, data: np.ndarray) -> None:
    np.asarray(data, dtype=np.uint16).tofile(tmp_path / f"tinystories_{split_name}.bin")


def test_get_batch_random_avoids_last_window(tmp_path):
    data = np.arange(32, dtype=np.uint16)
    _write_split(tmp_path, "train", data)
    loader = DataLoader(str(tmp_path), block_size=4, batch_size=5, device="cpu", random=True)

    torch.manual_seed(0)
    x, y = loader.get_batch("train")

    assert x.shape == (5, 4)
    assert y.shape == (5, 4)
    assert torch.equal(y, x + 1)
    assert torch.max(x[:, -1]) <= len(data) - 2


def test_get_batch_sequential_wraps_and_preserves_targets(tmp_path):
    data = np.arange(12, dtype=np.uint16)
    _write_split(tmp_path, "train", data)
    loader = DataLoader(str(tmp_path), block_size=4, batch_size=3, device="cpu", random=False)

    x1, y1 = loader.get_batch("train")
    x2, y2 = loader.get_batch("train")
    x3, y3 = loader.get_batch("train")

    assert torch.equal(x1[:, 0], torch.tensor([0, 1, 2]))
    assert torch.equal(x2[:, 0], torch.tensor([3, 4, 5]))
    assert torch.equal(x3[:, 0], torch.tensor([6, 7, 0]))

    assert torch.equal(y1, x1 + 1)
    assert torch.equal(y2, x2 + 1)
    assert torch.equal(y3, x3 + 1)

def test_get_batch_raises_when_dataset_too_short(tmp_path):
    data = np.arange(4, dtype=np.uint16)
    _write_split(tmp_path, "train", data)
    loader = DataLoader(str(tmp_path), block_size=4, batch_size=1, device="cpu")

    with pytest.raises(ValueError):
        loader.get_batch("train")
