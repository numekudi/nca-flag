/**
 * wasm（`nca-wasm` クレート）への薄いアダプタ。
 *
 * wasm の初期化・重みの取得という「外の世界」との接点をここに閉じ込め、
 * UI 側（`NcaFlagDemo`）は `NcaSimulation` のインタフェースだけに依存させる。
 * 生成物 `nca-wasm/pkg` への相対パスを書くのもこのファイルだけ。
 */
import init, { NcaSimulation } from "../../nca-wasm/pkg/nca_wasm.js";

export type { NcaSimulation };

/**
 * 学習済み NCA の重み。`nca` パッケージの `export-weights` が書き出した
 * フラット f32 バイナリ（`NCAW` ヘッダ付き、約 33KB）を `public/` に同梱している。
 */
export const DEFAULT_WEIGHTS_URL = "/models/nca_weights.v1.bin";

/**
 * 重みを読み込んだシミュレータを 1 つ作る。
 * wasm の初期化と重みの取得は独立なので並行に走らせる。
 *
 * 取得・初期化に失敗した場合は握り潰さずそのまま例外を投げる。
 */
export async function createSimulation(
  width: number,
  height: number,
  weightsUrl: string,
): Promise<NcaSimulation> {
  const [, weights] = await Promise.all([init(), fetchWeights(weightsUrl)]);
  return new NcaSimulation(width, height, weights);
}

/** 重みバイナリを取得して Uint8Array で返す。HTTP エラーは例外にする。 */
async function fetchWeights(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `weights fetch failed: ${response.status} ${response.statusText}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}
