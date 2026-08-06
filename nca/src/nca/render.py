"""学習結果の目視確認用レンダリング。

数値の損失だけでは「成長したが放置すると崩れる」「損傷から戻らない」といった
NCA 固有の失敗が見えないため、時間発展そのものを GIF に出して確認する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from nca.config import DamagePattern
from nca.model import NeuralCA, make_seed, to_rgb, to_rgba
from nca.pool import damage_masks


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """時間発展の切り出し方。"""

    grid: int
    steps: int = 200
    stride: int = 2
    """何ステップごとに 1 フレーム記録するか。"""

    scale: int = 4
    """最近傍拡大の倍率。CA の格子をそのまま見せるので補間はしない。"""

    damage_at: tuple[int, ...] = ()
    """このステップに達した時点で損傷を与える。再生能力の確認用。"""

    damage_pattern: DamagePattern = DamagePattern.DISC
    """damage_at で与える損傷の形。学習時と別の形を指定して汎化を見ることもできる。"""

    transparent: bool = False
    """True なら白背景に合成せず、細胞のアルファをそのまま持つ RGBA で書き出す。
    白地の目標（日章旗など）が白背景に溶けて消えるのを避けたいときに使う。"""

    background: float = 1.0
    """非透過時に合成する背景のグレー値 [0, 1]（既定 1.0 = 白）。白地の目標を
    灰色地（例 0.5）に置くと、GIF などで白地が背景と区別できる。transparent 時は無視。"""

    seed: int = 0

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not 0.0 <= self.background <= 1.0:
            raise ValueError(f"background must be in [0, 1], got {self.background}")
        for step in self.damage_at:
            if not 0 <= step < self.steps:
                raise ValueError(f"damage_at {step} is outside [0, {self.steps})")


def evolve(model: NeuralCA, config: RenderConfig, device: torch.device) -> list[Image.Image]:
    """シードから config.steps 進め、途中経過のフレーム列を返す。"""
    generator = torch.Generator(device=device).manual_seed(config.seed)
    state = make_seed(config.grid, model.config.channels).to(device)
    damage_steps = set(config.damage_at)

    def to_frame(current: Tensor) -> Image.Image:
        return _to_image(current, config.scale, config.transparent, config.background)

    frames = [to_frame(state)]
    with torch.no_grad():
        for step in range(config.steps):
            if step in damage_steps:
                state = state * damage_masks(
                    1, config.grid, generator, device, config.damage_pattern
                )
                frames.append(to_frame(state))
            state = model(state, model.sample_noise(state, generator))
            if (step + 1) % config.stride == 0:
                frames.append(to_frame(state))
    return frames


def _to_image(state: Tensor, scale: int, transparent: bool, background: float) -> Image.Image:
    # 透過時はストレートアルファの RGBA、非透過時は background 色に合成した RGB。
    if transparent:
        array = to_rgba(state)
        mode = "RGBA"
    else:
        array = to_rgb(state, background)
        mode = "RGB"
    pixels = array[0].permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
    image = Image.fromarray(np.ascontiguousarray(pixels), mode=mode)
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def save_animation(frames: list[Image.Image], path: Path, frame_ms: int = 50) -> None:
    """フレーム列をアニメーションとして書き出す。

    保存形式は拡張子から Pillow が判定する。半透明を保つなら .webp / .apng を、
    互換性優先なら .gif を渡す（GIF はアルファが 1bit なので半透明の縁は潰れる）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=False
    )
    print(f"wrote {len(frames)} frames -> {path}")


def save_still(frames: list[Image.Image], path: Path) -> None:
    """最終フレームだけを PNG で残す（目標形状に到達できたかの確認用）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(path)
    print(f"wrote final frame -> {path}")
