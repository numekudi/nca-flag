from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def deterministic() -> None:
    torch.manual_seed(0)


@pytest.fixture
def target_png(tmp_path: Path) -> Path:
    """中央に不透明な正方形を持つ 32x32 の RGBA PNG。"""
    array = np.zeros((32, 32, 4), dtype=np.uint8)
    array[8:24, 8:24] = (255, 0, 0, 255)
    path = tmp_path / "target.png"
    Image.fromarray(array, mode="RGBA").save(path)
    return path
