from __future__ import annotations

import pytest
import torch

from nca.config import ModelConfig
from nca.model import NeuralCA, make_seed, to_rgb, to_rgba


def test_zero_initialized_output_makes_the_first_step_a_no_op() -> None:
    """出力層のゼロ初期化により、学習前の CA は状態を一切変えない。"""
    model = NeuralCA(ModelConfig())
    x = make_seed(16, model.config.channels)
    stepped = model(x, model.sample_noise(x))
    assert torch.equal(stepped, x)


def test_dead_cells_stay_dead() -> None:
    """完全に空のグリッドからは何も生まれない（自然発生はしない）。"""
    model = NeuralCA(ModelConfig())
    torch.nn.init.normal_(model.output.weight, std=0.5)
    x = torch.zeros(1, model.config.channels, 16, 16)
    assert torch.count_nonzero(model.rollout(x, 8)) == 0


def test_growth_stays_adjacent_to_living_cells() -> None:
    """1 ステップで広がるのはシードの 1 近傍まで。飛び火しない。"""
    model = NeuralCA(ModelConfig())
    torch.nn.init.normal_(model.output.weight, std=0.5)
    grid = 16
    x = make_seed(grid, model.config.channels)

    alive = NeuralCA.alive_mask(model(x, torch.zeros(1, 1, grid, grid)))[0, 0]
    rows, columns = torch.nonzero(alive, as_tuple=True)
    center = grid // 2
    # alive_mask 自体が 3x3 の max pool なので、シードの 2 近傍が上限になる。
    assert int((rows - center).abs().max()) <= 2
    assert int((columns - center).abs().max()) <= 2


def test_fire_rate_zero_noise_updates_every_cell() -> None:
    """noise=0 なら全細胞が発火し、noise=1 なら誰も発火しない。"""
    model = NeuralCA(ModelConfig(fire_rate=0.5))
    torch.nn.init.normal_(model.output.weight, std=0.5)
    x = make_seed(16, model.config.channels)

    all_fired = model(x, torch.zeros(1, 1, 16, 16))
    none_fired = model(x, torch.ones(1, 1, 16, 16))
    assert not torch.equal(all_fired, x)
    assert torch.equal(none_fired, x)


def test_noise_shape_mismatch_fails_loudly() -> None:
    model = NeuralCA(ModelConfig())
    x = make_seed(16, model.config.channels)
    with pytest.raises(ValueError, match="noise shape"):
        model(x, torch.zeros(1, 1, 8, 8))


def test_perception_is_not_learnable() -> None:
    """知覚カーネルは固定。学習パラメータは更新則 MLP のみ。"""
    model = NeuralCA(ModelConfig())
    names = {name for name, _ in model.named_parameters()}
    assert names == {"hidden.weight", "hidden.bias", "output.weight"}


def test_to_rgb_renders_transparent_cells_as_white() -> None:
    x = torch.zeros(1, 16, 4, 4)
    assert torch.equal(to_rgb(x), torch.ones(1, 3, 4, 4))


def test_to_rgba_keeps_transparent_cells_fully_transparent() -> None:
    """アルファ 0 の細胞は透明のまま。白背景合成と違い、色を足さない。"""
    x = torch.zeros(1, 16, 4, 4)
    rgba = to_rgba(x)
    assert rgba.shape == (1, 4, 4, 4)
    assert torch.equal(rgba, torch.zeros(1, 4, 4, 4))


def test_to_rgba_unpremultiplies_to_straight_alpha() -> None:
    """premultiplied な赤（alpha=0.5, R=0.5）を割り戻すと straight では R=1.0 になる。"""
    x = torch.zeros(1, 16, 1, 1)
    x[:, 0] = 0.5  # premultiplied R = alpha * 1.0
    x[:, 3] = 0.5  # alpha
    rgba = to_rgba(x)
    assert torch.allclose(rgba[:, 0], torch.tensor(1.0))  # straight R
    assert torch.allclose(rgba[:, 3], torch.tensor(0.5))  # alpha は保持


def test_model_is_resolution_independent() -> None:
    """全て 1x1 畳み込みと 3x3 固定カーネルなので、任意のグリッドで動く。"""
    model = NeuralCA(ModelConfig())
    for grid in (8, 33, 64):
        x = make_seed(grid, model.config.channels)
        assert model(x, model.sample_noise(x)).shape == (1, 16, grid, grid)


def test_invalid_model_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(channels=4)
    with pytest.raises(ValueError):
        ModelConfig(fire_rate=0.0)
