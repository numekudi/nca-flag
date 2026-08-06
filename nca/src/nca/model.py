"""Growing Neural Cellular Automata の本体。

Mordvintsev et al. "Growing Neural Cellular Automata" (Distill, 2020) の再実装。

状態は [B, C, H, W] の連続値グリッド。先頭 4 チャンネルが可視の RGBA
（RGB はアルファで乗算済み = premultiplied）、残りは細胞が自由に使える隠れ状態。
更新則は全細胞で共有される 1 個の小さな MLP で、これが唯一の学習パラメータ。

境界は循環（トーラス）にしてある。ブラウザ背景としてタイリングしたときに継ぎ目が
出ないようにするためで、既存の Gray-Scott 実装（packages/turing-pattern）の周期境界と揃う。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nca.config import ModelConfig

ALPHA_CHANNEL = 3
VISIBLE_CHANNELS = 4
ALIVE_THRESHOLD = 0.1
"""3x3 近傍のアルファ最大値がこれを超える細胞だけが「生きている」とみなされる。"""


def _perception_kernel(channels: int) -> Tensor:
    """各チャンネルに [恒等, Sobel-X, Sobel-Y] を掛ける depthwise カーネルを作る。

    細胞が知覚できるのは自分の状態と、その周囲での勾配のみ。学習対象ではない固定値で、
    これにより「細胞は化学物質の濃度勾配を感じ取る」という生物学的な制約を模す。

    返り値の形状は [channels * 3, 1, 3, 3]。groups=channels の畳み込みに直接渡せる並び
    （入力チャンネル c の出力が 3c, 3c+1, 3c+2 に来る）。
    """
    identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    stacked = torch.stack([identity, sobel_x, sobel_x.T])
    return stacked.repeat(channels, 1, 1).unsqueeze(1)


def _wrap(x: Tensor) -> Tensor:
    """3x3 の窓を取るために 1 セル分だけ循環パディングする。"""
    return F.pad(x, (1, 1, 1, 1), mode="circular")


class NeuralCA(nn.Module):
    """1 ステップ分の更新則。ロールアウトは呼び出し側が繰り返して行う。

    forward が乱数を内部生成せず引数で受け取るのは、
    - ONNX へ出したときにグラフが純関数になり、JS 側の PRNG で再現できる
    - テストで発火パターンを固定できる
    ため。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("perception", _perception_kernel(config.channels), persistent=False)
        self.hidden = nn.Conv2d(config.channels * 3, config.hidden, kernel_size=1)
        self.output = nn.Conv2d(config.hidden, config.channels, kernel_size=1, bias=False)
        # 出力層をゼロ初期化する。学習開始時の状態変化が完全に 0 になり、
        # 「何も起きない CA」から少しずつ振る舞いを積み上げる形になるので学習が安定する。
        nn.init.zeros_(self.output.weight)

    def perceive(self, x: Tensor) -> Tensor:
        """各細胞の知覚ベクトル [B, C*3, H, W] を作る。"""
        return F.conv2d(_wrap(x), self.perception, groups=self.config.channels)

    @staticmethod
    def alive_mask(x: Tensor) -> Tensor:
        """生きている細胞（自分または近傍のアルファが閾値超え）の真偽マスク [B, 1, H, W]。"""
        alpha = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1]
        neighborhood_max = F.max_pool2d(_wrap(alpha), kernel_size=3, stride=1)
        return neighborhood_max > ALIVE_THRESHOLD

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        """状態を 1 ステップ進める。

        noise は [B, 1, H, W] の一様乱数 [0, 1)。fire_rate 以下の細胞だけが更新される。
        """
        if noise.shape != (x.shape[0], 1, *x.shape[2:]):
            raise ValueError(
                f"noise shape {tuple(noise.shape)} does not match state {tuple(x.shape)}"
            )

        was_alive = self.alive_mask(x)
        delta = self.output(F.relu(self.hidden(self.perceive(x))))
        fired = (noise <= self.config.fire_rate).to(x.dtype)
        x = x + delta * fired

        # 更新の前後どちらでも生きている細胞だけを残す。これにより成長は既存細胞の
        # 縁からしか起こらず、空間の離れた場所に飛び火しない。
        is_alive = was_alive & self.alive_mask(x)
        return x * is_alive.to(x.dtype)

    def sample_noise(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """forward に渡す発火抽選用の乱数を作る。"""
        return torch.rand(
            x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype, generator=generator
        )

    def rollout(self, x: Tensor, steps: int, generator: torch.Generator | None = None) -> Tensor:
        """steps 回連続で進める。"""
        for _ in range(steps):
            x = self(x, self.sample_noise(x, generator))
        return x


def make_seed(grid: int, channels: int, batch: int = 1) -> Tensor:
    """中央 1 セルだけが生きている初期状態を作る。

    RGB は 0（premultiplied なので「見えない」）、アルファ以降を 1 にする。
    ここから全ての構造が立ち上がる。
    """
    x = torch.zeros(batch, channels, grid, grid)
    x[:, ALPHA_CHANNEL:, grid // 2, grid // 2] = 1.0
    return x


def to_rgb(x: Tensor, background: float = 1.0) -> Tensor:
    """premultiplied な状態を、単色背景に合成した表示用 RGB [B, 3, H, W] にする。

    background は合成先のグレー値 [0, 1]（既定 1.0 = 白）。premultiplied なので
    out = background * (1 - alpha) + rgb。白地の目標を灰色地に置くと白地が際立つ。
    """
    rgb = x[:, :ALPHA_CHANNEL]
    alpha = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1].clamp(0.0, 1.0)
    return (background * (1.0 - alpha) + rgb).clamp(0.0, 1.0)


def to_rgba(x: Tensor) -> Tensor:
    """premultiplied な状態を、背景に合成しない表示用の straight-alpha RGBA [B, 4, H, W] にする。

    PNG / WebP はストレートアルファを前提とするため、premultiplied の RGB をアルファで
    割り戻す。アルファ 0 の細胞は完全な透明で色が定義されないので、割り算はせず 0 にする
    （ゼロ除算を避けるための場合分けであって、値を捏造しているわけではない）。
    """
    rgb = x[:, :ALPHA_CHANNEL]
    alpha = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1].clamp(0.0, 1.0)
    straight_rgb = torch.where(alpha > 0.0, rgb / alpha.clamp_min(1e-8), torch.zeros_like(rgb))
    return torch.cat([straight_rgb.clamp(0.0, 1.0), alpha], dim=1)
