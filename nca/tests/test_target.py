from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nca.config import TargetConfig
from nca.target import load_target, target_loss


def test_load_target_pads_to_grid(target_png: Path) -> None:
    config = TargetConfig(path=target_png, size=40, padding=16)
    target = load_target(target_png, config.size, config.padding)
    assert target.shape == (1, 4, config.grid, config.grid)
    assert config.grid == 72


def test_padding_region_is_fully_transparent(target_png: Path) -> None:
    target = load_target(target_png, size=8, padding=4)
    alpha = target[0, 3]
    assert float(alpha[:4].max()) == 0.0
    assert float(alpha[-4:].max()) == 0.0


def test_rgb_is_premultiplied_by_alpha(target_png: Path) -> None:
    """透明な画素は RGB も 0 になる。CA の状態表現と一致させるため。"""
    target = load_target(target_png, size=32, padding=0)
    transparent = target[0, 3] == 0.0
    assert float(target[0, :3, transparent].abs().max()) == 0.0


def test_opaque_target_without_alpha_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(path)
    with pytest.raises(ValueError, match="must be RGBA"):
        load_target(path, size=8, padding=0)


def test_target_loss_is_per_sample(target_png: Path) -> None:
    target = load_target(target_png, size=8, padding=0)
    batch = torch.zeros(3, 16, 8, 8)
    batch[0, :4] = target[0]
    loss = target_loss(batch, target)
    assert loss.shape == (3,)
    assert float(loss[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(loss[1]) > 0.0


def test_invalid_target_config_is_rejected(target_png: Path) -> None:
    with pytest.raises(ValueError):
        TargetConfig(path=target_png, size=0)
    with pytest.raises(ValueError):
        TargetConfig(path=target_png, padding=-1)
