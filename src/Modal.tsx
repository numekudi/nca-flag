import { createEffect, type JSX } from "solid-js";
import styles from "./Modal.module.css";

export type ModalProps = {
  /** true の間だけモーダルを開く。 */
  open: boolean;
  /** Esc・閉じるボタン・背景クリックのいずれかで閉じられたときに呼ばれる。 */
  onClose: () => void;
  /** 見出し要素の id。dialog の aria-labelledby に渡す。 */
  labelledBy: string;
  children: JSX.Element;
};

/**
 * 中身を持たない汎用モーダル。
 *
 * 実体は `<dialog>` の showModal なので、フォーカストラップ・Esc での終了・
 * 背景の inert 化はブラウザ側の実装に任せる（自前実装より確実で軽い）。
 */
export function Modal(props: ModalProps) {
  let dialog!: HTMLDialogElement;

  createEffect(() => {
    // open が既に反映済みのときに再度呼ぶと InvalidStateError になるため状態を見る。
    if (props.open && !dialog.open) dialog.showModal();
    if (!props.open && dialog.open) dialog.close();
  });

  return (
    <dialog
      class={styles.dialog}
      ref={dialog}
      aria-labelledby={props.labelledBy}
      // Esc キーによる終了もここに集約される。
      onClose={() => props.onClose()}
      // ::backdrop のクリックは dialog 自身が受け取る。中身のクリックと区別する。
      onClick={(event) => {
        if (event.target === dialog) props.onClose();
      }}
    >
      <div class={styles.body}>
        <button
          class={styles.close}
          type="button"
          aria-label="閉じる"
          onClick={() => props.onClose()}
        >
          ×
        </button>
        {props.children}
      </div>
    </dialog>
  );
}
