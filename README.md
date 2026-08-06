# nca-flag

「破いても自己修復する日章旗」— Growing Neural Cellular Automata のインタラクティブデモ。

学習（PyTorch）・推論（Rust/wasm）・表示（SolidJS）が 1 リポジトリに揃っている。
本体サイト（`primitive-ojisan.com`）とは別の Cloudflare Pages プロジェクトとして公開しており、
サイトを更新するたびに wasm をビルドし直さなくて済むよう切り離してある。

| ディレクトリ | 役割 |
|---|---|
| `nca/` | PyTorch での学習と、重みバイナリの書き出し（Python 3.12 / uv） |
| `nca-wasm/` | 学習済み重みを読んで前進計算だけ回す Rust → WebAssembly |
| `src/` | SolidJS のデモページ |
| `public/models/` | 配信する重みバイナリ（約 33KB） |

## 開発

```bash
pnpm install
pnpm dev        # wasm をビルドしてから vite dev
pnpm build      # wasm + tsc + vite build -> dist/
pnpm preview    # ビルドして wrangler pages dev
pnpm deploy     # ビルドして wrangler pages deploy
```

`pnpm build:wasm` は [wasm-pack](https://drager.github.io/wasm-pack/) を使う（Rust ツールチェーンが必要）。
出力先の `nca-wasm/pkg` は生成物なので git 管理外。`src/nca/simulation.ts` だけがそこを参照する。

## 重みの差し替え

`nca/` で学習し、`public/models/` の重みを置き換えるとデモの見た目が変わる。

```bash
cd nca
uv sync
uv run nca train --target assets/hinomaru.png --checkpoint artifacts/hinomaru.pt
uv run nca export-weights --checkpoint artifacts/hinomaru.pt \
  --dest ../public/models/nca_weights.v1.bin
```

重みバイナリの様式（`NCAW` ヘッダ + フラット f32）は `nca/src/nca/export_weights.py` と
`nca-wasm/src/lib.rs` の両方が知っている唯一の契約。片方を変えたらもう片方も必ず直すこと。

## OG 画像

`public/og-v2.png` は `assets/flag.png`（デモの canvas から取り出した 72x72 の日章旗）を
拡大してカードに組んだもの。文言や旗を変えたら作り直す。

```bash
uv run --project nca python scripts/generate_og.py
```

## デプロイ

Cloudflare Pages プロジェクト `nca-flag`（`wrangler.jsonc`）。`pnpm deploy` でローカルビルドをそのままアップロードする。
記事側（`primitive-ojisan.com/blog/self-healing-hinomaru`）からはリンクで参照している。
