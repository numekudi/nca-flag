import styles from "./ShareButton.module.css";

export type ShareButtonProps = {
  /** ツイート本文（URL は含めない。X 側が url パラメータを末尾に付ける）。 */
  text: string;
  /** 共有するページの URL。 */
  url: string;
};

/**
 * X（旧 Twitter）の投稿画面を開くボタン。
 *
 * 素の intent URL へのリンクなので JS を持たない。ポップアップブロックや
 * SPA の状態に左右されず、リンクとして共有・右クリックもできる。
 */
export function ShareButton(props: ShareButtonProps) {
  const href = () => {
    const params = new URLSearchParams({ text: props.text, url: props.url });
    return `https://x.com/intent/post?${params}`;
  };

  return (
    <a
      class={styles.share}
      href={href()}
      target="_blank"
      rel="noopener noreferrer"
    >
      {/* X のロゴ。単色なので currentColor で塗る。 */}
      <svg
        class={styles.icon}
        viewBox="0 0 24 24"
        aria-hidden="true"
        fill="currentColor"
      >
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
      X でシェア
    </a>
  );
}
