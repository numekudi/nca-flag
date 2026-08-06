"""学習ループ。

1 ステップの流れ:
  1. BatchSource から初期状態を得る（毎回シード / プールから再開＋損傷）
  2. 64〜96 ステップ CA を回す（回数を毎回変えることで特定の step 数への依存を防ぐ）
  3. 可視 4 チャンネルと目標画像の MSE を取り、逆伝播
  4. 最終状態をプールに書き戻す
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from tqdm import tqdm

from nca.checkpoint import save_checkpoint
from nca.config import ModelConfig, Regime, TargetConfig, TrainConfig
from nca.model import NeuralCA, make_seed
from nca.pool import BatchSource, SamplePool, SeedSource
from nca.target import load_target, target_loss


@dataclass(frozen=True, slots=True)
class TrainResult:
    model: NeuralCA
    final_loss: float


def build_batch_source(
    config: TrainConfig,
    seed: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> BatchSource:
    """レジームに対応するバッチ供給元を組み立てる。"""
    if config.regime is Regime.GROWING:
        return SeedSource(seed=seed)
    return SamplePool(
        states=seed.repeat(config.pool_size, 1, 1, 1),
        seed=seed,
        loss_fn=partial(target_loss, target=target),
        damage_count=config.effective_damage_count,
        damage_pattern=config.damage_pattern,
        generator=generator,
    )


def train(
    target_config: TargetConfig,
    config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> TrainResult:
    """CA を学習し、最終重みをチェックポイントに保存する。"""
    generator = torch.Generator(device=device).manual_seed(config.seed)

    target = load_target(target_config.path, target_config.size, target_config.padding).to(device)
    seed = make_seed(target_config.grid, model_config.channels).to(device)
    model = NeuralCA(model_config).to(device)
    source = build_batch_source(config, seed, target, generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.5))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.lr_decay_at], gamma=config.lr_decay
    )

    loss_value = float("nan")
    progress = tqdm(range(1, config.steps + 1), desc=config.regime.value)
    for step in progress:
        index, state = source.next_batch(config.batch_size)
        rollout = int(
            torch.randint(
                config.min_rollout, config.max_rollout + 1, (1,), generator=generator, device=device
            ).item()
        )
        state = model.rollout(state, rollout, generator)

        loss = target_loss(state, target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _normalize_gradients(model)
        optimizer.step()
        scheduler.step()

        source.commit(index, state.detach())
        loss_value = float(loss.detach())
        progress.set_postfix(loss=f"{loss_value:.5f}", rollout=rollout)

        # 途中で落ちても直前までの成果が残るよう定期的に保存する。
        if step % 500 == 0:
            save_checkpoint(config.checkpoint, model, target_config.grid, loss_value)

    save_checkpoint(config.checkpoint, model, target_config.grid, loss_value)
    print(f"saved checkpoint: {config.checkpoint} (loss {loss_value:.5f})")
    return TrainResult(model=model, final_loss=loss_value)


def _normalize_gradients(model: NeuralCA) -> None:
    """パラメータごとに勾配を単位ノルムへ正規化する。

    NCA の損失は数十ステップの再帰適用を通るため勾配のスケールが桁で振れる。
    正規化しないと学習が発散する（論文でも必須とされている手当）。
    """
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"parameter {name} received no gradient")
        gradient /= gradient.norm() + 1e-8
