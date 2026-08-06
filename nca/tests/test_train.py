from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nca.checkpoint import load_checkpoint
from nca.config import ModelConfig, Regime, TargetConfig, TrainConfig
from nca.model import make_seed
from nca.pool import SamplePool, SeedSource
from nca.render import RenderConfig, evolve
from nca.target import load_target
from nca.train import build_batch_source, train

CPU = torch.device("cpu")


def test_growing_regime_uses_seed_source(target_png: Path) -> None:
    seed = make_seed(16, 16)
    target = load_target(target_png, size=16, padding=0)
    generator = torch.Generator().manual_seed(0)
    config = TrainConfig(regime=Regime.GROWING)
    assert isinstance(build_batch_source(config, seed, target, generator), SeedSource)


def test_persistent_regime_disables_damage(target_png: Path) -> None:
    """損傷は REGENERATING でのみ有効。CLI で damage_count を残したままでも混ざらない。"""
    seed = make_seed(16, 16)
    target = load_target(target_png, size=16, padding=0)
    generator = torch.Generator().manual_seed(0)
    config = TrainConfig(regime=Regime.PERSISTENT, damage_count=3)
    source = build_batch_source(config, seed, target, generator)
    assert isinstance(source, SamplePool)
    assert source.damage_count == 0


def test_invalid_train_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        TrainConfig(batch_size=1)
    with pytest.raises(ValueError):
        TrainConfig(min_rollout=100, max_rollout=50)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=4, damage_count=4)
    with pytest.raises(ValueError):
        TrainConfig(pool_size=4, batch_size=8)


@pytest.mark.slow
def test_short_training_reduces_loss_and_round_trips(target_png: Path, tmp_path: Path) -> None:
    """数十ステップだけ回し、学習が進むこととチェックポイントが復元できることを見る。"""
    checkpoint = tmp_path / "nca.pt"
    target_config = TargetConfig(path=target_png, size=16, padding=4)
    model_config = ModelConfig(channels=8, hidden=32)
    config = TrainConfig(
        regime=Regime.REGENERATING,
        steps=30,
        batch_size=4,
        pool_size=16,
        min_rollout=8,
        max_rollout=12,
        damage_count=1,
        checkpoint=checkpoint,
    )

    result = train(target_config, config, model_config, CPU)
    assert result.final_loss < 0.25

    loaded = load_checkpoint(checkpoint, CPU)
    assert loaded.grid == target_config.grid
    assert loaded.model.config.channels == 8

    x = make_seed(loaded.grid, 8)
    reference = result.model.cpu()(x, torch.zeros(1, 1, loaded.grid, loaded.grid))
    restored = loaded.model(x, torch.zeros(1, 1, loaded.grid, loaded.grid))
    assert torch.allclose(reference, restored, atol=1e-6)


@pytest.mark.slow
def test_render_produces_frames_including_damage(target_png: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "nca.pt"
    train(
        TargetConfig(path=target_png, size=16, padding=4),
        TrainConfig(
            steps=5, batch_size=4, pool_size=8, min_rollout=4, max_rollout=6, checkpoint=checkpoint
        ),
        ModelConfig(channels=8, hidden=32),
        CPU,
    )
    loaded = load_checkpoint(checkpoint, CPU)
    frames = evolve(loaded.model, RenderConfig(grid=loaded.grid, steps=10, stride=2, scale=2), CPU)
    # 初期状態 1 枚 + stride ごとの 5 枚。
    assert len(frames) == 6
    assert frames[0].size == (loaded.grid * 2, loaded.grid * 2)


def test_render_config_rejects_damage_outside_the_run() -> None:
    with pytest.raises(ValueError, match="outside"):
        RenderConfig(grid=16, steps=10, damage_at=(20,))
