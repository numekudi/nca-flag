"""学習バッチの供給元。

レジーム間の違いは「どの状態から CA を回し始めるか」だけなので、そこだけを
差し替え可能な戦略として切り出す。学習ループ本体はどちらが来ても同じコードで動く。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from nca.config import DamagePattern


class BatchSource(Protocol):
    """学習 1 ステップ分の初期状態を供給する。"""

    def next_batch(self, size: int) -> tuple[Tensor, Tensor]:
        """(プール上の位置, 初期状態 [size, C, H, W]) を返す。"""
        ...

    def commit(self, index: Tensor, state: Tensor) -> None:
        """ロールアウト後の状態を書き戻す。"""
        ...


@dataclass(frozen=True, slots=True)
class SeedSource:
    """毎回まっさらなシードから始める（GROWING レジーム）。

    書き戻す先が無いので commit は何もしない。
    """

    seed: Tensor

    def next_batch(self, size: int) -> tuple[Tensor, Tensor]:
        return torch.empty(0, dtype=torch.long), self.seed.repeat(size, 1, 1, 1)

    def commit(self, index: Tensor, state: Tensor) -> None:
        return None


@dataclass(slots=True)
class SamplePool:
    """途中状態を貯めて再開する（PERSISTENT / REGENERATING レジーム）。

    毎回シードから始めると CA は「短時間で目標形状を作る」ことしか学ばない。
    到達済みの状態から再開させることで、初めて「その形を保ち続ける」圧力がかかる。
    """

    states: Tensor
    """[pool_size, C, H, W]。初期値は全てシード。"""

    seed: Tensor

    loss_fn: Callable[[Tensor], Tensor]
    """バッチをサンプルごとの損失 [B] に落とす。並べ替えにのみ使う。"""

    damage_count: int
    damage_pattern: DamagePattern
    generator: torch.Generator

    def next_batch(self, size: int) -> tuple[Tensor, Tensor]:
        device = self.states.device
        index = torch.randperm(self.states.shape[0], generator=self.generator, device=device)[:size]
        state = self.states[index].clone()

        # 損失の大きい順に並べ替える。壊れて戻らなくなったサンプルが先頭に来る。
        rank = self.loss_fn(state).argsort(descending=True)
        index, state = index[rank], state[rank]

        # 最悪の 1 枠をシードに戻す。これがプールの汚染を防ぐ唯一の排出口で、
        # 同時に「シードから成長する」能力を学習し続けるための供給源にもなる。
        state[0] = self.seed[0]

        # 損失の小さい（= よく出来ている）末尾に損傷を与える。既に形になったものを
        # 壊すからこそ、再生が学習される。
        if self.damage_count > 0:
            state[-self.damage_count :] *= damage_masks(
                self.damage_count, state.shape[-1], self.generator, device, self.damage_pattern
            )
        return index, state

    def commit(self, index: Tensor, state: Tensor) -> None:
        self.states[index] = state


def damage_masks(
    count: int,
    grid: int,
    generator: torch.Generator,
    device: torch.device,
    pattern: DamagePattern = DamagePattern.DISC,
) -> Tensor:
    """損傷マスク [count, 1, grid, grid] を作る（切り取られる領域が 0、残る領域が 1）。

    座標は grid の大きさに依らず [-1, 1] に正規化するため、損傷の大きさはグリッド解像度
    ではなく画面上の割合で決まる。
    """
    axis = torch.linspace(-1.0, 1.0, grid, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")

    if pattern is DamagePattern.MIXED:
        return _mixed_masks(count, y, x, generator, device)
    return _MASK_BUILDERS[pattern](count, y, x, generator, device)


type _MaskBuilder = Callable[[int, Tensor, Tensor, torch.Generator, torch.device], Tensor]


def _disc_masks(
    count: int, y: Tensor, x: Tensor, generator: torch.Generator, device: torch.device
) -> Tensor:
    """円形に切り取る。中心は中央寄りに散らし、半径は画面幅の 10〜40%。"""
    shape = (count, 1, 1, 1)
    center_x = torch.rand(shape, generator=generator, device=device) - 0.5
    center_y = torch.rand(shape, generator=generator, device=device) - 0.5
    radius = 0.1 + 0.3 * torch.rand(shape, generator=generator, device=device)

    squared_distance = (x - center_x) ** 2 + (y - center_y) ** 2
    return (squared_distance > radius**2).float()


def _band_masks(
    coordinate: Tensor, count: int, generator: torch.Generator, device: torch.device
) -> Tensor:
    """指定軸に垂直な帯を、グリッドの端から端まで切り取る。

    円と違って残った領域が完全に分断されるため、再生には帯を跨いだ復元が要る。
    """
    shape = (count, 1, 1, 1)
    center = torch.rand(shape, generator=generator, device=device) - 0.5
    half_width = 0.05 + 0.15 * torch.rand(shape, generator=generator, device=device)
    return ((coordinate - center).abs() > half_width).float()


def _vertical_masks(
    count: int, y: Tensor, x: Tensor, generator: torch.Generator, device: torch.device
) -> Tensor:
    """縦一直線。x 方向に幅を持ち、y 方向は全域。"""
    return _band_masks(x, count, generator, device)


def _horizontal_masks(
    count: int, y: Tensor, x: Tensor, generator: torch.Generator, device: torch.device
) -> Tensor:
    """横一直線。y 方向に幅を持ち、x 方向は全域。"""
    return _band_masks(y, count, generator, device)


_MASK_BUILDERS: dict[DamagePattern, _MaskBuilder] = {
    DamagePattern.DISC: _disc_masks,
    DamagePattern.VERTICAL: _vertical_masks,
    DamagePattern.HORIZONTAL: _horizontal_masks,
}


def _mixed_masks(
    count: int, y: Tensor, x: Tensor, generator: torch.Generator, device: torch.device
) -> Tensor:
    """サンプルごとに基本パターンを一様に選ぶ。

    全パターン分をまとめて作ってから 1 つ選ぶ。count は高々バッチサイズなので、
    捨てる分の計算量よりも乱数消費が選択結果に依らず一定になる利点を取る。
    """
    builders = list(_MASK_BUILDERS.values())
    candidates = torch.stack([build(count, y, x, generator, device) for build in builders])
    choice = torch.randint(
        len(builders), (1, count, 1, 1, 1), generator=generator, device=device
    ).expand_as(candidates[:1])
    return candidates.gather(0, choice).squeeze(0)
