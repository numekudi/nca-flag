"""OG 画像（1200x630）を生成する。

デモの canvas から取り出した盤面（`public/demo.png`、72x72 の RGBA）を
最近傍で拡大し、タイトルと施行日を添えたカードにする。
サイト本文に載せている「異常増殖」のスクリーンショットと同じ画像を使い、
カードと本文で同じ盤面を見せる。

`nca` の依存（Pillow）をそのまま借りて実行する:

    uv run --project nca python scripts/generate_og.py

出力: `public/og-v2.png`
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
# 本文の <figure> と同じ実物を使う（片方だけ差し替わって食い違うのを防ぐ）。
FLAG_PATH = ROOT / "public" / "demo.png"
OUT_PATH = ROOT / "public" / "og-v2.png"

WIDTH, HEIGHT = 1200, 630
BG = (255, 255, 255, 255)
FLAG_RED = (188, 0, 45, 255)
TEXT_MAIN = (8, 6, 13, 255)
TEXT_SUB = (107, 99, 117, 255)
# 旗の下地（白地を白背景から浮かせるための薄い面）。
PANEL = (244, 243, 236, 255)
# 下端のアクセント。日章旗の赤で、カード全体の帰属を示す。
ACCENT_BAR_HEIGHT = 12

# 旗を置く領域（左）。ここに収まるよう整数倍で拡大する。
FLAG_BOX = (72, 96, 500, 480)  # (x, y, w, h)
TEXT_X = 620

FONT_CANDIDATES = {
    "bold": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    ],
    "regular": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    ],
}


def load_font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """日本語が出せるフォントを 1 つ選ぶ。見つからなければ失敗させる（豆腐を出さない）。"""
    for path in FONT_CANDIDATES[weight]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    raise FileNotFoundError(
        f"日本語フォント（{weight}）が見つからない: {FONT_CANDIDATES[weight]}"
    )


def load_flag() -> Image.Image:
    """旗を不透明部分で切り出す。周囲の透明な余白は OG では邪魔なので落とす。"""
    flag = Image.open(FLAG_PATH).convert("RGBA")
    bbox = flag.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"{FLAG_PATH} が全面透明。デモから取り直すこと")
    return flag.crop(bbox)


def fit_nearest(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """整数倍の最近傍拡大で box に収める。1 セル 1 ピクセルの見た目を保つため補間しない。"""
    scale = max(1, min(box_w // image.width, box_h // image.height))
    return image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )


def main() -> None:
    card = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(card)

    # 旗の下地。白背景に白地の旗を置くと形が消えるので、薄い面を敷いて輪郭を出す。
    draw.rounded_rectangle(
        (FLAG_BOX[0], FLAG_BOX[1], FLAG_BOX[0] + FLAG_BOX[2], FLAG_BOX[1] + FLAG_BOX[3]),
        radius=24,
        fill=PANEL,
    )

    # 旗（左）。領域の中央に置く。
    flag = fit_nearest(load_flag(), FLAG_BOX[2] - 48, FLAG_BOX[3] - 48)
    card.alpha_composite(
        flag,
        (
            FLAG_BOX[0] + (FLAG_BOX[2] - flag.width) // 2,
            FLAG_BOX[1] + (FLAG_BOX[3] - flag.height) // 2,
        ),
    )

    # 文字（右）。上から: 施行日 → タイトル 2 行 → 技術 → URL。
    draw.text(
        (TEXT_X, 150),
        "2026.8.13 国旗損壊罪 施行",
        font=load_font("bold", 30),
        fill=FLAG_RED,
    )
    title_font = load_font("bold", 58)
    draw.text((TEXT_X, 210), "破いても", font=title_font, fill=TEXT_MAIN)
    draw.text((TEXT_X, 284), "自己修復する日章旗", font=title_font, fill=TEXT_MAIN)

    sub_font = load_font("regular", 26)
    draw.text(
        (TEXT_X, 386),
        "Growing Neural Cellular Automata",
        font=sub_font,
        fill=TEXT_SUB,
    )
    draw.text(
        (TEXT_X, 424),
        "Rust + WebAssembly でブラウザ推論",
        font=sub_font,
        fill=TEXT_SUB,
    )
    draw.text(
        (TEXT_X, 480),
        "nca-flag.primitive-ojisan.com",
        font=load_font("regular", 24),
        fill=FLAG_RED,
    )

    draw.rectangle(
        (0, HEIGHT - ACCENT_BAR_HEIGHT, WIDTH, HEIGHT),
        fill=FLAG_RED,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    card.convert("RGB").save(OUT_PATH, format="PNG", optimize=True)
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
