import { NcaFlagDemo } from "./nca/NcaFlagDemo";
import { ShareButton } from "./ShareButton";
import styles from "./App.module.css";

/** 解説記事（本体サイト側）。デモから戻れるようにしておく。 */
const ARTICLE_URL = "https://primitive-ojisan.com/blog/self-healing-hinomaru";
/** 手法の元論文（Distill, 2020）。このデモの Neural Cellular Automata はここが出典。 */
const PAPER_URL = "https://distill.pub/2020/growing-ca/";
/** 共有するときの正規 URL。`index.html` の og:url / og:image と同じ場所を指すこと。 */
const SHARE_URL = "https://nca-flag.primitive-ojisan.com/";
/** ツイート本文。ページの掴みと同じ「法律 → だから自己修復」の順で書く。 */
const SHARE_TEXT =
  "2026年 8月13日、国旗損壊罪が施行されます。 #国旗損壊罪法施行";

/**
 * デモ単体のページ。
 * ここは「触れること」が主目的なので、説明は最小限にして記事へ送る。
 */
function App() {
  return (
    <main class={styles.page}>
      <h1 class={styles.title}>自己修復する国旗セルオートマトン</h1>

      {/* デモだけ見せてもコンセプトが伝わらないので、出発点の法律を先に置く。 */}
      <p class={styles.lead}>
        <strong>2026 年 8 月 13 日</strong>、日本国旗の損壊などを罰する
        <strong>国旗損壊罪</strong>が施行されます。
        だとすれば、いちばん安全な旗は
        <strong>破いても自分で元に戻る旗</strong>でしょう。
        というわけで、破くと自己修復する日章旗を作りました。
      </p>
      <p class={styles.leadSub}>
        ※壊しすぎると復活しません。まずはタップする程度でお試しください。
      </p>

      <NcaFlagDemo />

      <ul class={styles.hints}>
        <li>ドラッグ・タップしている間、その場所が破けます</li>
        <li>ダブルタップ・ダブルクリックで、その場所に成長のシードを置けます</li>
        <li>「リセット」で中央のシード 1 個からやり直します</li>
      </ul>

      {/* 学習済み重みの癖で稀に起きる崩れ方。文章だけだと伝わらないので実物を見せる。 */}
      <figure class={styles.figure}>
        <img
          class={styles.figureImage}
          src="/demo.png"
          width="72"
          height="72"
          alt="日章旗が崩れ、赤い水玉模様が盤面いっぱいに増殖したスクリーンショット"
          loading="lazy"
        />
        <figcaption class={styles.figureCaption}>
          まれに異常増殖します。
        </figcaption>
      </figure>

      <ShareButton text={SHARE_TEXT} url={SHARE_URL} />

      <p class={styles.footer}>
        仕組みの解説は <a href={ARTICLE_URL}>ブログ記事</a> にあります。
        <br />
        元論文は{" "}
        <a href={PAPER_URL} target="_blank" rel="noreferrer">
          Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020)
        </a>
        です。
      </p>
    </main>
  );
}

export default App;
