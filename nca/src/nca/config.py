"""学習・描画の設定値。

すべて frozen dataclass で表現し、CLI からの入力を一箇所で型付けする。
グローバル定数を各モジュールに散らさないことで、テストから任意の設定を注入できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch


class Regime(StrEnum):
    """学習レジーム。Distill 論文の Experiment 1-3 に対応する。

    バッチの供給元（毎回シード / サンプルプール）と損傷の有無だけが異なり、
    モデル構造も損失も共通である。
    """

    GROWING = "growing"
    """毎回シードから固定ステップ。目標形状には到達するが、その後放置すると崩壊する。"""

    PERSISTENT = "persistent"
    """サンプルプールから再開することで、到達後も形状を保つことを学習する。"""

    REGENERATING = "regenerating"
    """プール + 損傷。切り取られても再生する。外からノイズを注入する用途にはこれが必須。"""


class DamagePattern(StrEnum):
    """損傷マスクの形。何を「壊し方」として経験させるかで、再生能力の性質が変わる。

    円は局所的な欠損、帯は格子を跨ぐ大域的な切断であり、後者は分断された両側が
    互いの情報無しに復元できるかを問う点で難度が異なる。
    """

    DISC = "disc"
    """円形に切り取る（Distill 論文の既定）。"""

    VERTICAL = "vertical"
    """縦一直線。列方向の帯を上下いっぱいに切り取る。"""

    HORIZONTAL = "horizontal"
    """横一直線。行方向の帯を左右いっぱいに切り取る。"""

    MIXED = "mixed"
    """上記 3 種をサンプルごとに一様に選ぶ。1 回の学習で全パターンを経験させる。"""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """CA の容量。推論コストは channels と hidden にほぼ比例する。"""

    channels: int = 16
    """状態のチャンネル数。先頭 4 が可視の RGBA、残りが隠れ状態。"""

    hidden: int = 128
    """更新則 MLP の中間幅。"""

    fire_rate: float = 0.5
    """1 ステップで更新される細胞の割合。細胞間の同期を壊し、大域クロックへの依存を防ぐ。"""

    def __post_init__(self) -> None:
        if self.channels <= 4:
            raise ValueError(f"channels must exceed the 4 visible RGBA ones, got {self.channels}")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden}")
        if not 0.0 < self.fire_rate <= 1.0:
            raise ValueError(f"fire_rate must be in (0, 1], got {self.fire_rate}")


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """目標画像の読み込み設定。"""

    path: Path
    """RGBA の PNG。アルファが「細胞が生きているか」の教師になるため、透過は必須。"""

    size: int = 40
    """目標画像を収める正方形の一辺。"""

    padding: int = 16
    """成長する余地として目標の周囲に確保する余白。グリッド全体は size + 2 * padding になる。"""

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.padding < 0:
            raise ValueError(f"padding must not be negative, got {self.padding}")

    @property
    def grid(self) -> int:
        """シミュレーショングリッドの一辺。"""
        return self.size + 2 * self.padding


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """学習ループのハイパーパラメータ。既定値は Distill 論文に準拠する。"""

    regime: Regime = Regime.REGENERATING
    steps: int = 8000
    batch_size: int = 8
    pool_size: int = 1024
    """サンプルプールの大きさ。GROWING では使われない。"""

    min_rollout: int = 64
    max_rollout: int = 96
    """1 回の学習ステップで回す CA ステップ数の範囲（毎回この範囲から一様に選ぶ）。"""

    learning_rate: float = 2e-3
    lr_decay_at: int = 2000
    lr_decay: float = 0.1
    """lr_decay_at ステップ目で学習率を lr_decay 倍する。"""

    damage_count: int = 3
    """バッチ内で損傷を与えるサンプル数。REGENERATING 以外では 0 として扱う。"""

    damage_pattern: DamagePattern = DamagePattern.DISC
    """損傷マスクの形。実験ごとに切り替える対象。"""

    seed: int = 0
    checkpoint: Path = Path("artifacts/nca.pt")

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 (one slot is always reseeded)")
        if not 0 < self.min_rollout <= self.max_rollout:
            raise ValueError(f"invalid rollout range [{self.min_rollout}, {self.max_rollout}]")
        if self.pool_size < self.batch_size:
            raise ValueError("pool_size must be at least batch_size")
        # 損傷を与える枠と、必ずシードで置き換える先頭 1 枠は重ねられない。
        if not 0 <= self.damage_count < self.batch_size:
            raise ValueError(f"damage_count must be in [0, batch_size), got {self.damage_count}")

    @property
    def effective_damage_count(self) -> int:
        """レジームを踏まえた実際の損傷数。損傷は REGENERATING でのみ意味を持つ。"""
        return self.damage_count if self.regime is Regime.REGENERATING else 0


def resolve_device(preferred: str | None = None) -> torch.device:
    """利用可能なアクセラレータを解決する。

    明示指定があればそれを尊重し、存在しなければ即座に失敗させる
    （CPU への暗黙フォールバックは学習時間の想定を静かに壊すため行わない）。
    """
    if preferred is not None:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
