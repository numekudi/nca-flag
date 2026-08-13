import { Modal } from "./Modal";
import { ShareButton } from "./ShareButton";
import styles from "./ShareModal.module.css";

export type ShareModalProps = {
  open: boolean;
  onClose: () => void;
  /** ツイート本文（URL は含めない）。 */
  shareText: string;
  /** 共有するページの URL。 */
  shareUrl: string;
};

/** aria-labelledby 用。1 ページに 1 つしか出さない前提の固定 id。 */
const TITLE_ID = "share-modal-title";

/**
 * 最後まで読んだ人に、作品の立ち位置を伝えてシェアを促すモーダル。
 * 見せ方（Modal）と中身（この文章）を分けて、文言だけ差し替えられるようにしている。
 */
export function ShareModal(props: ShareModalProps) {
  return (
    <Modal open={props.open} onClose={props.onClose} labelledBy={TITLE_ID}>
      <p class={styles.title} id={TITLE_ID}>
        このWebページはデジタル作品であり、
        <br />
        罰則の対象外です。
      </p>

      {/* 長文なので「事実の説明」と「お願い」で段落を割り、視線の休憩を作る。 */}
      <p>
        この作品は機械学習により実現していますが、訓練に使用したデータセットは国旗の絵文字一つのみです。AI
        というよりは <strong>AL（人工生命）</strong>{" "}
        の分野ですが、AI作品としてひと括りにされてしまうことに若干の不条理を感じています。
      </p>

      <p class={styles.text}>
        共感いただける場合は、是非 X
        でシェアいただけると幸いです。反応が励みになります。
      </p>

      <div class={styles.actions}>
        <ShareButton text={props.shareText} url={props.shareUrl} />
      </div>
    </Modal>
  );
}
