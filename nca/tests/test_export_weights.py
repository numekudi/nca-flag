"""重みエクスポートの往復・様式テスト。"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from nca.checkpoint import save_checkpoint
from nca.config import ModelConfig
from nca.export_weights import FORMAT_VERSION, MAGIC, export_weights
from nca.model import NeuralCA


def _make_checkpoint(path: Path) -> NeuralCA:
    model = NeuralCA(ModelConfig(channels=16, hidden=128, fire_rate=0.5))
    save_checkpoint(path, model, grid=72, loss=0.0)
    return model


def test_header_and_sizes_are_self_describing(tmp_path: Path) -> None:
    """ヘッダのマジック・版・次元が一致し、本体長が 3 テンソル分ちょうどになる。"""
    ckpt = tmp_path / "m.pt"
    _make_checkpoint(ckpt)
    dest = export_weights(ckpt, tmp_path / "w.bin")

    blob = dest.read_bytes()
    magic, version, channels, hidden, fire_rate = struct.unpack("<4sIIIf", blob[:20])
    assert magic == MAGIC
    assert version == FORMAT_VERSION
    assert channels == 16
    assert hidden == 128
    assert fire_rate == 0.5

    # hidden.weight[128,48] + hidden.bias[128] + output.weight[16,128] を f32 で。
    expected_floats = hidden * (channels * 3) + hidden + channels * hidden
    assert len(blob) - 20 == expected_floats * 4


def test_body_matches_state_dict_values(tmp_path: Path) -> None:
    """本体の先頭ブロックが hidden.weight の行優先 f32 と一致する。"""
    ckpt = tmp_path / "m.pt"
    model = _make_checkpoint(ckpt)
    dest = export_weights(ckpt, tmp_path / "w.bin")

    blob = dest.read_bytes()
    channels, hidden = 16, 128
    n = hidden * (channels * 3)
    got = np.frombuffer(blob[20 : 20 + n * 4], dtype="<f4")
    want = model.state_dict()["hidden.weight"].reshape(hidden, channels * 3).numpy().ravel()
    assert np.array_equal(got, want.astype("<f4"))
