"""学習済み重みを、ブラウザ推論（Rust/wasm）が読むフラットバイナリに書き出す。

`.pt` は PyTorch 専用形式なので、そのままでは wasm から読めない。ここでは学習される
3 つのテンソル（`hidden.weight`, `hidden.bias`, `output.weight`）だけを、自己記述の
小さなヘッダ付き little-endian f32 として直列化する。

知覚カーネル（恒等 / Sobel-X / Sobel-Y）は固定値なので **書き出さない**。読み手側
（`packages/nca-wasm`）が同じ定義を持つことを唯一の前提とし、二重管理を避ける。

バイナリ様式（すべて little-endian）:

    offset  type      内容
    0       4s        マジック b"NCAW"
    4       u32       フォーマット版
    8       u32       channels（状態のチャンネル数）
    12      u32       hidden（更新則 MLP の中間幅）
    16      f32       fire_rate
    20      f32[H*C3] hidden.weight   （行優先 [hidden, channels*3]）
    ...     f32[H]    hidden.bias     （[hidden]）
    ...     f32[C*H]  output.weight   （行優先 [channels, hidden]）

ここで C3 = channels * 3（知覚で 1 チャンネルが 3 チャンネルに増える）。
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

from nca.checkpoint import load_checkpoint

MAGIC = b"NCAW"
FORMAT_VERSION = 1


def _flat_f32(tensor: torch.Tensor) -> bytes:
    """テンソルを行優先の little-endian f32 バイト列にする。"""
    # contiguous を強制して view 由来のストライドを潰し、numpy 経由で明示的に <f4 にする。
    array = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
    return array.astype("<f4", copy=False).tobytes()


def export_weights(checkpoint: Path, dest: Path) -> Path:
    """チェックポイントを読み、学習パラメータをフラットバイナリとして dest に書く。

    形状が想定と食い違えば即座に失敗させる（wasm 側で黙って誤読させないため）。
    """
    loaded = load_checkpoint(checkpoint)
    state = loaded.model.state_dict()
    config = loaded.model.config

    channels = config.channels
    hidden = config.hidden
    perception_channels = channels * 3

    hidden_weight = state["hidden.weight"]  # [hidden, channels*3, 1, 1]
    hidden_bias = state["hidden.bias"]  # [hidden]
    output_weight = state["output.weight"]  # [channels, hidden, 1, 1]

    # 1x1 畳み込みの余分な空間次元 (1, 1) を落として行列に均す。
    hidden_weight = hidden_weight.reshape(hidden, perception_channels)
    output_weight = output_weight.reshape(channels, hidden)

    expected = {
        "hidden.weight": (hidden_weight.shape, (hidden, perception_channels)),
        "hidden.bias": (hidden_bias.shape, (hidden,)),
        "output.weight": (output_weight.shape, (channels, hidden)),
    }
    for name, (actual, want) in expected.items():
        if tuple(actual) != want:
            raise ValueError(f"{name} shape {tuple(actual)} != expected {want}")

    header = struct.pack("<4sIIIf", MAGIC, FORMAT_VERSION, channels, hidden, config.fire_rate)
    body = _flat_f32(hidden_weight) + _flat_f32(hidden_bias) + _flat_f32(output_weight)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(header + body)
    print(
        f"exported weights: {dest} "
        f"({len(header) + len(body)} bytes, channels={channels}, hidden={hidden})"
    )
    return dest
