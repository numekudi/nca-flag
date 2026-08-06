//! Growing Neural Cellular Automata のブラウザ推論。
//!
//! `packages/nca`（PyTorch）の `NeuralCA.forward`（`src/nca/model.py:82-99`）を 1:1 で移植した
//! 1 ステップ更新則。学習はせず、`packages/nca` が `export-weights` で書き出したフラット f32
//! バイナリ（`src/nca/export_weights.py` の様式）を読んで前進計算だけを回す。
//!
//! 状態レイアウトは PyTorch と同じチャンネル優先 `[C, H, W]`:
//! チャンネル c・座標 (y, x) の値は `c * (H * W) + y * W + x`。先頭 4 チャンネルが
//! premultiplied RGBA、残りが隠れ状態。境界は循環（トーラス）。
//!
//! `packages/turing-pattern` と同じ「wasm 線形メモリを JS から直接読む」流儀に揃えてあり、
//! `rgba_ptr()` が指す `[H*W*4]` の u8 バッファを `putImageData` でそのまま描ける。

use wasm_bindgen::prelude::*;

/// 可視 RGBA のアルファチャンネル位置。
const ALPHA: usize = 3;
/// 3x3 近傍のアルファ最大値がこれを超える細胞だけが「生きている」。
const ALIVE_THRESHOLD: f32 = 0.1;
/// 重みバイナリの先頭マジックと対応フォーマット版（`export_weights.py` と一致させる）。
const MAGIC: &[u8; 4] = b"NCAW";
const FORMAT_VERSION: u32 = 1;

/// xorshift32。発火抽選の擬似乱数。PyTorch とビット一致させる必要はなく、
/// 見た目の再現（各細胞が毎ステップ fire_rate の確率でのみ更新される）だけを担う。
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// little-endian で連続する f32 を読み出すカーソル。
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    /// f32 を count 個読み進める。範囲外は None（呼び出し側で明示的に失敗させる）。
    fn take_f32(&mut self, count: usize) -> Option<Vec<f32>> {
        let end = self.offset + count * 4;
        if end > self.bytes.len() {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        let mut p = self.offset;
        for _ in 0..count {
            let word = [
                self.bytes[p],
                self.bytes[p + 1],
                self.bytes[p + 2],
                self.bytes[p + 3],
            ];
            out.push(f32::from_le_bytes(word));
            p += 4;
        }
        self.offset = end;
        Some(out)
    }
}

#[wasm_bindgen]
pub struct NcaSimulation {
    width: usize,
    height: usize,
    channels: usize,
    hidden: usize,
    fire_rate: f32,

    // 学習パラメータ。知覚カーネルは固定なので保持しない（下の perceive で再構成する）。
    hidden_weight: Vec<f32>, // 行優先 [hidden, channels*3]
    hidden_bias: Vec<f32>,   // [hidden]
    output_weight: Vec<f32>, // 行優先 [channels, hidden]

    state: Vec<f32>, // [channels * height * width]
    next: Vec<f32>,  // 更新の作業バッファ（候補状態）
    rng: u32,
}

#[wasm_bindgen]
impl NcaSimulation {
    /// 重みバイナリを読んでシミュレータを構築する。
    ///
    /// channels / hidden はバイナリのヘッダが唯一の真実。状態のサイズもそこから決まる。
    /// マジック・版・長さのいずれかが合わなければ即座に失敗させ、壊れた重みで
    /// もっともらしく動いてしまう事態を防ぐ。
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, weights: &[u8]) -> Result<NcaSimulation, JsError> {
        if width == 0 || height == 0 {
            return Err(JsError::new("width and height must be positive"));
        }
        if weights.len() < 20 || &weights[0..4] != MAGIC {
            return Err(JsError::new("weights: bad magic (expected NCAW)"));
        }
        let version = u32::from_le_bytes([weights[4], weights[5], weights[6], weights[7]]);
        if version != FORMAT_VERSION {
            return Err(JsError::new("weights: unsupported format version"));
        }
        let channels =
            u32::from_le_bytes([weights[8], weights[9], weights[10], weights[11]]) as usize;
        let hidden =
            u32::from_le_bytes([weights[12], weights[13], weights[14], weights[15]]) as usize;
        let fire_rate = f32::from_le_bytes([weights[16], weights[17], weights[18], weights[19]]);
        if channels <= ALPHA + 1 || hidden == 0 {
            return Err(JsError::new("weights: degenerate channels/hidden"));
        }

        let perception_channels = channels * 3;
        let mut cursor = Cursor::new(weights, 20);
        let hidden_weight = cursor
            .take_f32(hidden * perception_channels)
            .ok_or_else(|| JsError::new("weights: truncated hidden.weight"))?;
        let hidden_bias = cursor
            .take_f32(hidden)
            .ok_or_else(|| JsError::new("weights: truncated hidden.bias"))?;
        let output_weight = cursor
            .take_f32(channels * hidden)
            .ok_or_else(|| JsError::new("weights: truncated output.weight"))?;

        let cells = width * height;
        let mut sim = NcaSimulation {
            width,
            height,
            channels,
            hidden,
            fire_rate,
            hidden_weight,
            hidden_bias,
            output_weight,
            state: vec![0.0; channels * cells],
            next: vec![0.0; channels * cells],
            rng: 1,
        };
        sim.reset(0);
        Ok(sim)
    }

    /// 中央 1 セルだけを生かした初期状態に戻す（`make_seed` と同じ）。
    /// RGB は 0、アルファ以降の隠れチャンネルを 1 にする。
    pub fn reset(&mut self, rng_seed: u32) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
        let cx = self.width / 2;
        let cy = self.height / 2;
        let center = cy * self.width + cx;
        let cells = self.width * self.height;
        for c in ALPHA..self.channels {
            self.state[c * cells + center] = 1.0;
        }
        self.rng = if rng_seed == 0 { 0x9E3779B9 } else { rng_seed };
    }

    /// (cx, cy) を中心とする半径 radius（ピクセル）の円内を全チャンネル 0 にする。
    /// クリックで「破く」インタラクションのための損傷。学習の `damage_masks` と同義。
    pub fn damage(&mut self, cx: f32, cy: f32, radius: f32) {
        let cells = self.width * self.height;
        let r2 = radius * radius;
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r2 {
                    let idx = y * self.width + x;
                    for c in 0..self.channels {
                        self.state[c * cells + idx] = 0.0;
                    }
                }
            }
        }
    }

    /// (x, y) の 1 セルだけをシード（`reset` の中央セルと同じ状態）にする。
    /// 盤面全体は消さないので、既存の模様の隣に新しい成長の核を置ける。
    /// 範囲外の座標は何もしない（クリック位置の丸め誤差で panic させないため）。
    pub fn seed(&mut self, x: f32, y: f32) {
        if !(x >= 0.0 && y >= 0.0) {
            return;
        }
        let (sx, sy) = (x as usize, y as usize);
        if sx >= self.width || sy >= self.height {
            return;
        }
        let cells = self.width * self.height;
        let idx = sy * self.width + sx;
        // RGB は 0、アルファ以降を 1 にする（`reset` と同じ初期細胞）。
        for c in 0..ALPHA {
            self.state[c * cells + idx] = 0.0;
        }
        for c in ALPHA..self.channels {
            self.state[c * cells + idx] = 1.0;
        }
    }

    /// 状態を steps 回進める。
    pub fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    fn step_once(&mut self) {
        let w = self.width;
        let h = self.height;
        let cells = w * h;
        let c3 = self.channels * 3;

        // 各細胞の更新前の生存を先に確定させる（更新後のアルファに依存させないため）。
        // was_alive と、更新後の is_alive の論理積が最終的な生存になる。
        let mut hidden_acc = vec![0.0f32; self.hidden];
        let mut perception = vec![0.0f32; c3];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;

                // 発火抽選は 1 細胞につき 1 回（全チャンネルで共有）。noise <= fire_rate で更新。
                let noise = (xorshift32(&mut self.rng) as f32) / (u32::MAX as f32);
                let fired = if noise <= self.fire_rate { 1.0 } else { 0.0 };

                if fired == 0.0 {
                    // 更新されない細胞は状態を素通し。
                    for c in 0..self.channels {
                        self.next[c * cells + idx] = self.state[c * cells + idx];
                    }
                    continue;
                }

                self.fill_perception(x, y, &mut perception);

                // hidden = relu(W_h · perception + b_h)
                for j in 0..self.hidden {
                    let row = j * c3;
                    let mut sum = self.hidden_bias[j];
                    for i in 0..c3 {
                        sum += self.hidden_weight[row + i] * perception[i];
                    }
                    hidden_acc[j] = if sum > 0.0 { sum } else { 0.0 };
                }

                // delta = W_o · hidden（バイアス無し）。x = x + delta。
                for c in 0..self.channels {
                    let row = c * self.hidden;
                    let mut delta = 0.0f32;
                    for j in 0..self.hidden {
                        delta += self.output_weight[row + j] * hidden_acc[j];
                    }
                    self.next[c * cells + idx] = self.state[c * cells + idx] + delta;
                }
            }
        }

        // 生存マスク: 更新前・更新後どちらでも生きている細胞だけを残す。
        // was_alive は元 state、is_alive は候補 next のアルファ 3x3 近傍最大で判定する。
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let was_alive = self.alive_at(&self.state, x, y);
                let is_alive = self.alive_at(&self.next, x, y);
                if !(was_alive && is_alive) {
                    for c in 0..self.channels {
                        self.next[c * cells + idx] = 0.0;
                    }
                }
            }
        }

        std::mem::swap(&mut self.state, &mut self.next);
    }

    /// 座標 (x, y) の細胞について、固定知覚 [恒等, Sobel-X, Sobel-Y] を全チャンネル分作る。
    /// 出力の並びはチャンネル c → [3c]=恒等, [3c+1]=Sobel-X, [3c+2]=Sobel-Y（PyTorch と同じ）。
    fn fill_perception(&self, x: usize, y: usize, out: &mut [f32]) {
        let w = self.width;
        let h = self.height;
        let cells = w * h;

        // 循環境界の 3x3 近傍インデックス。
        let xm = if x == 0 { w - 1 } else { x - 1 };
        let xp = if x == w - 1 { 0 } else { x + 1 };
        let ym = if y == 0 { h - 1 } else { y - 1 };
        let yp = if y == h - 1 { 0 } else { y + 1 };

        for c in 0..self.channels {
            let base = c * cells;
            let tl = self.state[base + ym * w + xm];
            let tc = self.state[base + ym * w + x];
            let tr = self.state[base + ym * w + xp];
            let ml = self.state[base + y * w + xm];
            let mc = self.state[base + y * w + x];
            let mr = self.state[base + y * w + xp];
            let bl = self.state[base + yp * w + xm];
            let bc = self.state[base + yp * w + x];
            let br = self.state[base + yp * w + xp];

            // Sobel-X = [[-1,0,1],[-2,0,2],[-1,0,1]] / 8、Sobel-Y はその転置。
            let sobel_x = ((tr - tl) + 2.0 * (mr - ml) + (br - bl)) / 8.0;
            let sobel_y = ((bl - tl) + 2.0 * (bc - tc) + (br - tr)) / 8.0;

            out[3 * c] = mc;
            out[3 * c + 1] = sobel_x;
            out[3 * c + 2] = sobel_y;
        }
    }

    /// 座標 (x, y) が生きているか（アルファの 3x3 循環近傍の最大が閾値超え）。
    fn alive_at(&self, field: &[f32], x: usize, y: usize) -> bool {
        let w = self.width;
        let h = self.height;
        let base = ALPHA * w * h;
        let xm = if x == 0 { w - 1 } else { x - 1 };
        let xp = if x == w - 1 { 0 } else { x + 1 };
        let ym = if y == 0 { h - 1 } else { y - 1 };
        let yp = if y == h - 1 { 0 } else { y + 1 };

        let mut m = f32::MIN;
        for &ny in &[ym, y, yp] {
            for &nx in &[xm, x, xp] {
                let a = field[base + ny * w + nx];
                if a > m {
                    m = a;
                }
            }
        }
        m > ALIVE_THRESHOLD
    }

    /// 現在の状態を straight-alpha の RGBA u8 `[height*width*4]` に焼いて返す。
    /// premultiplied な RGB をアルファで割り戻す（`to_rgba` と同じ）。
    /// `putImageData` にそのまま渡せる。
    pub fn render(&self) -> Vec<u8> {
        let cells = self.width * self.height;
        let mut rgba = vec![0u8; cells * 4];
        for idx in 0..cells {
            let a = self.state[ALPHA * cells + idx].clamp(0.0, 1.0);
            let (r, g, b) = if a > 0.0 {
                (
                    (self.state[idx] / a).clamp(0.0, 1.0),
                    (self.state[cells + idx] / a).clamp(0.0, 1.0),
                    (self.state[2 * cells + idx] / a).clamp(0.0, 1.0),
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            let o = idx * 4;
            rgba[o] = (r * 255.0) as u8;
            rgba[o + 1] = (g * 255.0) as u8;
            rgba[o + 2] = (b * 255.0) as u8;
            rgba[o + 3] = (a * 255.0) as u8;
        }
        rgba
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}
