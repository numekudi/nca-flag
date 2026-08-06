# nca

Growing Neural Cellular Automata の学習と、時間発展のプレビュー。
Mordvintsev et al. ["Growing Neural Cellular Automata"](https://distill.pub/2020/growing-ca/) (Distill, 2020) の PyTorch 実装。

書き出した重みは同リポジトリの `../nca-wasm`（Rust/wasm）が読み、`../src` の SolidJS デモが表示する。
このパッケージ自体は Python の学習・検証までを担当する。

## セットアップ

```bash
cd nca
uv sync
```

Python 3.12。kusoapp の cloudseg と同じく、Linux では PyTorch を
CUDA 12.8 ホイールから取得する（RTX 50 系に必要）。GPU が無い環境でも CPU で動く。

## 使い方

```bash
uv run nca fetch-emoji 1f98e            # 目標画像を取得 -> assets/lizard.png
uv run nca train                        # 学習 -> artifacts/nca.pt
uv run nca render --damage-at 120       # 時間発展 -> artifacts/growth.gif, final.png
```

主なオプションは `uv run nca train --help` を参照。

### 目標画像

`--target` に任意の **RGBA（透過つき）PNG** を渡せる。アルファがそのまま
「細胞が存在すべき領域」の教師になるため、不透明な RGB 画像は読み込み時に弾かれる。

向いている画像:

- 塊としてのシルエットが明確なもの
- 細い線・小さな文字を含まないもの（自己修復時に崩れやすい）

デモは背景の市松模様の上に等倍で描かれるため、**シルエットと輝度構造** が効く。

`assets/` は基本的に git 管理外だが、デモが実際に使っている `assets/hinomaru.png` だけは
唯一の教師データなので追跡している（`.gitignore` に例外を書いてある）。これは
リポジトリ直下の `assets/flag.png`（デモ canvas を写した 72x72）の中央 40x40 を
切り出したもの。**すでにパディングが乗った画像をそのまま `--target` に渡してはいけない**
—— 学習側が `size` へ縮小してから `padding` を足すので、旗が二重に縮む。

## 学習レジーム

`--regime` で 3 種類（論文の Experiment 1-3 に対応）。違いは「どの状態から CA を回し始めるか」だけ。

| regime | バッチの供給元 | 結果 |
|---|---|---|
| `growing` | 毎回シード | 目標形状には到達するが、放置すると崩壊する |
| `persistent` | サンプルプール | 到達後も形状を保つ |
| `regenerating`（既定） | プール + 円形の損傷 | 切り取られても再生する |

**外部からノイズや破壊を注入する用途では `regenerating` が必須**。
`growing` / `persistent` で学習したモデルは、一度壊すと元に戻らない。

## モデル

| | |
|---|---|
| 状態 | `[B, 16, H, W]`。先頭 4 が可視の RGBA（premultiplied）、残り 12 が隠れ状態 |
| 知覚 | 固定の depthwise 3x3（恒等 / Sobel-X / Sobel-Y）→ 48ch。**学習しない** |
| 更新則 | `conv1x1 48→128` + ReLU → `conv1x1 128→16`（バイアス無し・ゼロ初期化） |
| 学習パラメータ | 約 8.3k（`hidden.weight`, `hidden.bias`, `output.weight` のみ） |
| 確率的更新 | 各細胞は毎ステップ `fire_rate` の確率でのみ更新される |
| 生存判定 | 3x3 近傍のアルファ最大値 > 0.1 |
| 境界 | 循環（トーラス）。タイリング時に継ぎ目が出ないため |

`forward(state, noise)` は乱数を内部生成せず引数で受け取る。ONNX へ出したときにグラフが純関数になり、
JS 側の PRNG で再現できること、テストで発火パターンを固定できることが理由。

## 構成

| モジュール | 役割 |
|---|---|
| `config.py` | 設定の dataclass、レジーム定義、デバイス解決 |
| `model.py` | NeuralCA 本体、シード生成、表示用 RGB 変換 |
| `target.py` | 目標画像の読み込み（premultiply + パディング）、Noto Emoji 取得、損失 |
| `pool.py` | バッチ供給戦略（`SeedSource` / `SamplePool`）と損傷マスク |
| `train.py` | 学習ループ、勾配正規化 |
| `checkpoint.py` | `.pt` の読み書き |
| `render.py` | 時間発展の GIF / PNG 書き出し |
| `cli.py` | Typer CLI |

## テスト

```bash
uv run pytest                  # 全件
uv run pytest -m "not slow"    # 実際に学習を回すテストを除く
uv run ruff check . && uv run ruff format --check .
```
