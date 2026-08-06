/**
 * ポインタ入力の純粋なロジック。DOM にも Solid にも依存させないことで、
 * 「二本指の判定がおかしい」といった不具合を UI 抜きで再現・検証できるようにする。
 */

/** グリッド座標（セル単位、小数を含む）。 */
export type GridPoint = { x: number; y: number };

/** 要素上のクライアント座標を、要素全面に貼られた grid×grid の座標へ写す。 */
export function toGridPoint(
  element: Element,
  clientX: number,
  clientY: number,
  grid: number,
): GridPoint {
  const rect = element.getBoundingClientRect();
  return {
    x: ((clientX - rect.left) / rect.width) * grid,
    y: ((clientY - rect.top) / rect.height) * grid,
  };
}

/** ダブルタップ判定のしきい値。 */
export type DoubleTapThresholds = {
  /** 2 回目の押下がこの時間（ms）以内なら連続とみなす。 */
  intervalMs: number;
  /** 1 回目からのずれの許容量（グリッドセル単位）。 */
  slop: number;
};

/**
 * 押下の系列からダブルタップを検出する。
 *
 * `dblclick` イベントではなく押下（pointerdown）の系列で判定するのは、
 * 「押しっぱなしで破く」操作と同じイベント列の上で扱いたいから。
 */
export function createDoubleTapDetector(thresholds: DoubleTapThresholds) {
  let previous: (GridPoint & { time: number }) | null = null;

  return {
    /**
     * 押下を 1 回記録し、それがダブルタップの 2 回目かを返す。
     * 2 回目と判定した時点で系列はリセットされる（3 連打が 2 回連続で当たらないように）。
     */
    tap(point: GridPoint, time: number): boolean {
      const isDoubleTap =
        previous !== null &&
        time - previous.time <= thresholds.intervalMs &&
        Math.hypot(point.x - previous.x, point.y - previous.y) <=
          thresholds.slop;

      previous = isDoubleTap ? null : { ...point, time };
      return isDoubleTap;
    },
  };
}
