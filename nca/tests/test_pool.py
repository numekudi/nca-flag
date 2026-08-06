from __future__ import annotations

import pytest
import torch

from nca.config import DamagePattern
from nca.model import make_seed
from nca.pool import SamplePool, SeedSource, damage_masks

CHANNELS = 16
GRID = 16


def _pool(damage_count: int, damage_pattern: DamagePattern = DamagePattern.DISC) -> SamplePool:
    seed = make_seed(GRID, CHANNELS)
    return SamplePool(
        states=seed.repeat(32, 1, 1, 1),
        seed=seed,
        loss_fn=lambda state: state[:, :4].mean(dim=(1, 2, 3)),
        damage_count=damage_count,
        damage_pattern=damage_pattern,
        generator=torch.Generator().manual_seed(0),
    )


def test_seed_source_always_returns_fresh_seeds() -> None:
    seed = make_seed(GRID, CHANNELS)
    source = SeedSource(seed=seed)
    index, batch = source.next_batch(4)
    assert batch.shape == (4, CHANNELS, GRID, GRID)
    assert torch.equal(batch, seed.repeat(4, 1, 1, 1))
    # 書き戻し先が無いので commit は状態を持たない。
    source.commit(index, batch)


def test_pool_reseeds_the_worst_sample() -> None:
    """損失が最大のサンプルはシードに置き換えられ、プールに滞留しない。"""
    pool = _pool(damage_count=0)
    pool.states[:] = torch.randn_like(pool.states)
    _, batch = pool.next_batch(8)
    assert torch.equal(batch[0], pool.seed[0])


def test_pool_damages_only_the_tail() -> None:
    pool = _pool(damage_count=3)
    pool.states[:] = 1.0
    _, batch = pool.next_batch(8)
    # 先頭はシード、末尾 3 件は円形に削られている。中間はそのまま。
    assert torch.equal(batch[1:5], torch.ones_like(batch[1:5]))
    assert float(batch[-3:].min()) == 0.0


def test_commit_writes_back_to_the_sampled_slots() -> None:
    pool = _pool(damage_count=0)
    index, batch = pool.next_batch(4)
    updated = batch + 1.0
    pool.commit(index, updated)
    assert torch.equal(pool.states[index], updated)


@pytest.mark.parametrize("pattern", list(DamagePattern))
def test_damage_mask_is_binary_and_removes_something(pattern: DamagePattern) -> None:
    masks = damage_masks(5, GRID, torch.Generator().manual_seed(1), torch.device("cpu"), pattern)
    assert masks.shape == (5, 1, GRID, GRID)
    assert set(masks.unique().tolist()) == {0.0, 1.0}
    assert all(float(mask.mean()) < 1.0 for mask in masks)


def _is_constant_along(mask: torch.Tensor, dim: int) -> bool:
    """mask [1, GRID, GRID] が指定軸に沿って一定か（= その軸方向に貫通した帯か）。"""
    return bool((mask.amin(dim=dim) == mask.amax(dim=dim)).all())


def test_vertical_damage_cuts_through_every_row() -> None:
    """縦の帯は列単位で 0/1 が決まる（= y 方向に一定）。"""
    masks = damage_masks(
        8, GRID, torch.Generator().manual_seed(2), torch.device("cpu"), DamagePattern.VERTICAL
    )
    # 行方向（dim=-2）に潰しても値が変わらない = 縦に貫通している。
    assert all(_is_constant_along(mask, dim=-2) for mask in masks)


def test_horizontal_damage_cuts_through_every_column() -> None:
    masks = damage_masks(
        8, GRID, torch.Generator().manual_seed(2), torch.device("cpu"), DamagePattern.HORIZONTAL
    )
    assert all(_is_constant_along(mask, dim=-1) for mask in masks)


def test_mixed_damage_draws_all_three_patterns() -> None:
    """mixed は 3 種を混ぜる。十分な標本数なら円・縦・横がすべて現れる。"""
    masks = damage_masks(
        64, GRID, torch.Generator().manual_seed(3), torch.device("cpu"), DamagePattern.MIXED
    )
    vertical = [m for m in masks if _is_constant_along(m, dim=-2)]
    horizontal = [m for m in masks if _is_constant_along(m, dim=-1)]
    disc = [
        m for m in masks if not _is_constant_along(m, dim=-2) and not _is_constant_along(m, dim=-1)
    ]
    assert vertical and horizontal and disc


def test_pool_applies_the_configured_pattern() -> None:
    """プールに指定した形がそのまま損傷に使われる。"""
    pool = _pool(damage_count=3, damage_pattern=DamagePattern.HORIZONTAL)
    pool.states[:] = 1.0
    _, batch = pool.next_batch(8)
    # 損傷した末尾は、可視 1 チャンネルを見れば横に貫通した帯になっている。
    assert all(_is_constant_along(sample[:1], dim=-1) for sample in batch[-3:])
