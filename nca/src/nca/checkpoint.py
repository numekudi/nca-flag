"""チェックポイントの入出力。学習と描画/エクスポート間の唯一の受け渡し形式。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from nca.config import ModelConfig
from nca.model import NeuralCA


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: NeuralCA
    grid: int
    """学習時のグリッド一辺。推論時は変えられる（CA は解像度非依存）が、既定値として持つ。"""

    loss: float


def save_checkpoint(path: Path, model: NeuralCA, grid: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": model.config.channels,
            "hidden": model.config.hidden,
            "fire_rate": model.config.fire_rate,
            "grid": grid,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | None = None) -> LoadedCheckpoint:
    """学習済み重みを読み、eval モードのモデルを返す。

    weights_only=True で読むため、想定外のオブジェクトが混ざった .pt は読み込み時に失敗する。
    """
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    model = NeuralCA(
        ModelConfig(
            channels=int(payload["channels"]),
            hidden=int(payload["hidden"]),
            fire_rate=float(payload["fire_rate"]),
        )
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return LoadedCheckpoint(model=model, grid=int(payload["grid"]), loss=float(payload["loss"]))
