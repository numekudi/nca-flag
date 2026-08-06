import { createSignal, mergeProps, onCleanup, onMount, Show } from "solid-js";
import {
  createSimulation,
  DEFAULT_WEIGHTS_URL,
  type NcaSimulation,
} from "./simulation";
import { createDoubleTapDetector, toGridPoint } from "./pointer";
import styles from "./NcaFlagDemo.module.css";

/** 学習時のシミュレーショングリッド（size 40 + padding 16*2）。CA は解像度非依存だが既定はこれ。 */
const DEFAULT_GRID = 72;
/** 1 フレームあたりの CA ステップ数。多いほど成長が速く見える。 */
const DEFAULT_STEPS_PER_FRAME = 2;
/** 損傷（押している間破く）の半径。グリッドに対する比。 */
const DAMAGE_RADIUS_RATIO = 0.1;
/** ダブルタップ（＝シード配置）とみなす条件。 */
const DOUBLE_TAP = { intervalMs: 300, slop: 4 };

export type NcaFlagDemoProps = {
  /** 重みバイナリの URL。 */
  weightsUrl?: string;
  /** グリッドの一辺（セル数）。 */
  grid?: number;
  /** 1 フレームあたりの CA ステップ数。 */
  stepsPerFrame?: number;
};

type DemoStatus =
  | { kind: "loading" }
  | { kind: "running" }
  | { kind: "error"; message: string };

/**
 * 「破いても甦る日章旗」。学習済み NCA をブラウザ（wasm）で回し、canvas に描く。
 *
 * canvas を押している間はその位置を円形に破壊し続け、CA が自己修復する様子を体験できる。
 * ダブルタップ・ダブルクリックすると、その位置に成長のシード細胞を 1 個置ける。
 *
 * props はマウント時に一度だけ読む（シミュレータの寿命＝コンポーネントの寿命）。
 * 途中でグリッドを変えたい場合は、呼び出し側が `key` 相当の作り直しを行うこと。
 */
export function NcaFlagDemo(props: NcaFlagDemoProps) {
  const config = mergeProps(
    {
      weightsUrl: DEFAULT_WEIGHTS_URL,
      grid: DEFAULT_GRID,
      stepsPerFrame: DEFAULT_STEPS_PER_FRAME,
    },
    props,
  );

  const [status, setStatus] = createSignal<DemoStatus>({ kind: "loading" });

  let canvas!: HTMLCanvasElement;
  let simulation: NcaSimulation | null = null;
  /** 押下中のポインタ位置（グリッド座標）。null なら押していない。毎フレーム破壊に使う。 */
  let damagePoint: { x: number; y: number } | null = null;
  const doubleTap = createDoubleTapDetector(DOUBLE_TAP);

  onMount(() => {
    const grid = config.grid;
    let frame = 0;
    // アンマウント後に非同期の続きが走ってもループを始めさせないための門。
    let running = true;

    async function start() {
      const simulator = await createSimulation(grid, grid, config.weightsUrl);
      if (!running) {
        simulator.free();
        return;
      }
      simulation = simulator;

      // canvas のバッキングストアはグリッドと 1:1。切り抜かずそのまま転送し、
      // 拡大は CSS（最近傍）に任せる。
      const context = canvas.getContext("2d");
      if (!context) throw new Error("2D canvas context is unavailable");
      const imageData = context.createImageData(grid, grid);

      setStatus({ kind: "running" });

      const draw = () => {
        if (!running) return;
        // 押している間はフレームごとに破壊し続ける（ドラッグでなぞって破ける）。
        if (damagePoint) {
          simulator.damage(
            damagePoint.x,
            damagePoint.y,
            grid * DAMAGE_RADIUS_RATIO,
          );
        }
        simulator.step(config.stepsPerFrame);
        // render() は straight-alpha の RGBA [grid*grid*4] を返す。
        imageData.data.set(simulator.render());
        // putImageData は合成せず上書きするので clearRect は不要。
        context.putImageData(imageData, 0, 0);
        frame = requestAnimationFrame(draw);
      };
      frame = requestAnimationFrame(draw);
    }

    start().catch((error: unknown) => {
      // 黙って失敗させない。取得や初期化の失敗はそのまま表示する。
      running = false;
      setStatus({
        kind: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    });

    onCleanup(() => {
      running = false;
      cancelAnimationFrame(frame);
      simulation?.free();
      simulation = null;
    });
  });

  /**
   * 押下開始。ダブルタップ（短時間・近接の 2 回目）ならシードを置き、
   * そうでなければ押しっぱなし破壊を始める。
   */
  const handlePointerDown = (event: PointerEvent) => {
    if (!simulation) return;
    const point = toGridPoint(canvas, event.clientX, event.clientY, config.grid);

    if (doubleTap.tap(point, event.timeStamp)) {
      // 2 回目のタップは破壊せずシード配置に使う（破いた穴に核を打ち直せる）。
      damagePoint = null;
      simulation.seed(point.x, point.y);
      return;
    }

    damagePoint = point;
    // 指・カーソルが canvas 外へ出ても押下の追跡を続ける。
    canvas.setPointerCapture(event.pointerId);
  };

  /** 押下中の移動。破壊位置を追従させる。 */
  const handlePointerMove = (event: PointerEvent) => {
    if (!damagePoint) return;
    damagePoint = toGridPoint(
      canvas,
      event.clientX,
      event.clientY,
      config.grid,
    );
  };

  /** 押下終了（離す・キャンセル）。破壊を止める。 */
  const handlePointerEnd = () => {
    damagePoint = null;
  };

  /**
   * 長押し・右クリックのメニューを止める。
   * ここでの長押しは「破き続ける」操作なので、画像の保存メニューが出ると邪魔になる。
   */
  const handleContextMenu = (event: MouseEvent) => {
    event.preventDefault();
  };

  return (
    <div class={styles.demo}>
      {/* 白地（旗の一部）が両テーマで見えるよう、市松模様の下地を敷く。 */}
      <div class={styles.board}>
        <canvas
          ref={canvas}
          width={config.grid}
          height={config.grid}
          class={styles.canvas}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onContextMenu={handleContextMenu}
        />
      </div>
      <div class={styles.controls}>
        <Show when={status().kind === "loading"}>
          <span>モデルを読み込み中…</span>
        </Show>
        <Show
          when={(() => {
            const current = status();
            return current.kind === "error" ? current : null;
          })()}
        >
          {(error) => (
            <span class={styles.error}>
              読み込みに失敗しました: {error().message}
            </span>
          )}
        </Show>
        <button
          type="button"
          class={styles.button}
          disabled={status().kind !== "running"}
          onClick={() => simulation?.reset(0)}
        >
          リセット
        </button>
      </div>
    </div>
  );
}
