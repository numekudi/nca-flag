"""目標画像の読み込みと、Noto Emoji の取得。

CA が学習するのは「この RGBA 画像に収束すること」だけなので、目標画像がこの
パッケージにおける唯一の教師データになる。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from nca.model import VISIBLE_CHANNELS

NOTO_EMOJI_URL = (
    "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{code}.png"
)


def fetch_emoji(codepoints: str, dest: Path) -> Path:
    """Noto Emoji の 128px PNG を取得して保存する。

    codepoints は "1f98e"（🦎）のような 16 進コードポイント。ZWJ 合成絵文字は
    "1f469_200d_1f4bb" のようにアンダースコア区切りで指定する。
    """
    url = NOTO_EMOJI_URL.format(code=codepoints.lower())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    dest.write_bytes(payload)
    print(f"fetched {url} -> {dest}")
    return dest


def load_target(path: Path, size: int, padding: int) -> Tensor:
    """目標 RGBA を [1, 4, grid, grid] の premultiplied テンソルとして読む。

    アルファはそのまま「細胞が存在すべき領域」の教師になるため、透過を持たない画像は
    全面が生存領域になってしまう。意図しない目標を黙って受け入れないよう明示的に弾く。
    """
    with Image.open(path) as handle:
        if handle.mode != "RGBA":
            raise ValueError(f"target must be RGBA (transparent) but {path} is {handle.mode}")
        resized = handle.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0

    image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    # premultiply。CA の状態も同じ表現なので、損失を素の差分で取れる。
    image[:, :3] *= image[:, 3:4]
    return torch.nn.functional.pad(image, (padding,) * 4)


def target_loss(state: Tensor, target: Tensor) -> Tensor:
    """サンプルごとの損失 [B]。可視 4 チャンネルの二乗誤差平均。

    隠れチャンネルには一切の教師を与えない。何を表現に使うかは CA に委ねられる。
    """
    return ((state[:, :VISIBLE_CHANNELS] - target) ** 2).mean(dim=(1, 2, 3))
