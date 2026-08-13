import { createSignal, onCleanup, onMount, type Accessor } from "solid-js";

/**
 * ページ末尾まで読まれたら true になり、以降 false に戻らないシグナルを返す。
 *
 * 「一度でも下まで読んだ」ことを表す latch なので、スクロールを戻しても真のまま。
 * モーダルのような一度きりの提示に使うことを想定している。
 *
 * @param marginPx 末尾とみなす余白。慣性スクロールで数 px 足りない場合を吸収する。
 */
export function createPageBottomReached(marginPx = 32): Accessor<boolean> {
  const [reached, setReached] = createSignal(false);

  const update = () => {
    const doc = document.documentElement;
    // そもそもスクロールできない短いページで、読了扱いにしてしまわないようにする。
    if (doc.scrollHeight <= window.innerHeight + marginPx) return;

    if (window.scrollY + window.innerHeight >= doc.scrollHeight - marginPx) {
      setReached(true);
    }
  };

  onMount(() => {
    window.addEventListener("scroll", update, { passive: true });
    // 画像・canvas の読み込みで高さが変わるため、リサイズでも測り直す。
    window.addEventListener("resize", update, { passive: true });
    update();

    onCleanup(() => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    });
  });

  return reached;
}
