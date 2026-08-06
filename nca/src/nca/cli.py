"""コマンドラインインタフェース。

uv run nca fetch-emoji 1f98e
uv run nca train
uv run nca render
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nca.checkpoint import load_checkpoint
from nca.config import (
    DamagePattern,
    ModelConfig,
    Regime,
    TargetConfig,
    TrainConfig,
    resolve_device,
)
from nca.export_weights import export_weights
from nca.render import RenderConfig, evolve, save_animation, save_still
from nca.target import fetch_emoji
from nca.train import train as run_train

app = typer.Typer(add_completion=False, help="Growing Neural Cellular Automata: train and preview.")

DEFAULT_TARGET = Path("assets/lizard.png")
DEFAULT_CHECKPOINT = Path("artifacts/nca.pt")


@app.command("fetch-emoji")
def fetch_emoji_command(
    codepoints: Annotated[
        str, typer.Argument(help="16 進コードポイント。例: 1f98e (🦎)")
    ] = "1f98e",
    dest: Annotated[Path, typer.Option(help="保存先 PNG")] = DEFAULT_TARGET,
) -> None:
    """Noto Emoji から目標画像を取得する。"""
    fetch_emoji(codepoints, dest)


@app.command()
def train(
    target: Annotated[Path, typer.Option(help="目標 RGBA PNG")] = DEFAULT_TARGET,
    size: int = 40,
    padding: int = 16,
    regime: Regime = Regime.REGENERATING,
    steps: int = 8000,
    batch_size: int = 8,
    pool_size: int = 1024,
    min_rollout: int = 64,
    max_rollout: int = 96,
    learning_rate: float = 2e-3,
    channels: int = 16,
    hidden: int = 128,
    fire_rate: float = 0.5,
    damage_count: int = 3,
    damage_pattern: Annotated[
        DamagePattern, typer.Option(help="損傷の形。mixed はサンプルごとに 3 種から選ぶ")
    ] = DamagePattern.DISC,
    seed: int = 0,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: Annotated[str | None, typer.Option(help="cuda / cpu。未指定なら自動判定")] = None,
) -> None:
    """CA を学習する。"""
    run_train(
        target_config=TargetConfig(path=target, size=size, padding=padding),
        config=TrainConfig(
            regime=regime,
            steps=steps,
            batch_size=batch_size,
            pool_size=pool_size,
            min_rollout=min_rollout,
            max_rollout=max_rollout,
            learning_rate=learning_rate,
            damage_count=damage_count,
            damage_pattern=damage_pattern,
            seed=seed,
            checkpoint=checkpoint,
        ),
        model_config=ModelConfig(channels=channels, hidden=hidden, fire_rate=fire_rate),
        device=resolve_device(device),
    )


@app.command("export-weights")
def export_weights_command(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    dest: Annotated[Path, typer.Option(help="出力するフラットバイナリ (.bin)")] = Path(
        "artifacts/nca_weights.bin"
    ),
) -> None:
    """学習済み重みを wasm 用のフラット f32 バイナリに書き出す。"""
    export_weights(checkpoint, dest)


@app.command()
def render(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output: Annotated[
        Path, typer.Option(help="アニメの保存先。拡張子で形式が決まる（.webp/.apng/.gif）")
    ] = Path("artifacts/growth.gif"),
    still: Annotated[Path, typer.Option(help="最終フレーム PNG の保存先")] = Path(
        "artifacts/final.png"
    ),
    grid: Annotated[int | None, typer.Option(help="未指定なら学習時と同じ大きさ")] = None,
    steps: int = 200,
    stride: int = 2,
    scale: int = 4,
    damage_at: Annotated[
        list[int] | None, typer.Option(help="このステップで損傷を与える。複数指定可")
    ] = None,
    damage_pattern: Annotated[
        DamagePattern, typer.Option(help="damage-at で与える損傷の形")
    ] = DamagePattern.DISC,
    transparent: Annotated[
        bool,
        typer.Option(
            help="白背景に合成せず透過 RGBA で書き出す。白地の目標を残すとき用。GIF ではなく "
            ".webp/.apng 推奨"
        ),
    ] = False,
    background: Annotated[
        float,
        typer.Option(help="非透過時に合成する背景のグレー値 [0,1]（1=白, 0.5=灰色）"),
    ] = 1.0,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda / cpu。未指定なら自動判定")] = None,
) -> None:
    """学習済み CA の時間発展をアニメーションに書き出す。"""
    resolved = resolve_device(device)
    loaded = load_checkpoint(checkpoint, resolved)
    frames = evolve(
        loaded.model,
        RenderConfig(
            grid=grid if grid is not None else loaded.grid,
            steps=steps,
            stride=stride,
            scale=scale,
            damage_at=tuple(damage_at) if damage_at is not None else (),
            damage_pattern=damage_pattern,
            transparent=transparent,
            background=background,
            seed=seed,
        ),
        resolved,
    )
    save_animation(frames, output)
    save_still(frames, still)


if __name__ == "__main__":
    app()
