from __future__ import annotations

import base64
import binascii
import html
from pathlib import Path
import re
import shutil
from html.parser import HTMLParser

import pandas as pd

from utils import OUTPUT_DIR, ROOT_DIR, ensure_project_dirs, load_dataframe

DOCS_DIR = ROOT_DIR / "docs"
IMAGE_ASSET_DIR = DOCS_DIR / "images"
REPORT_PATHS = [DOCS_DIR / "analysis_report.html", DOCS_DIR / "index.html"]
EMBEDDED_IMAGE_PATHS = {
    "candidate_total_characters": OUTPUT_DIR / "descriptive" / "candidate_total_characters.png",
    "question_answer_length_heatmap": OUTPUT_DIR / "descriptive" / "question_answer_length_heatmap.png",
}

SHARED_FOCUS_TERMS = ["課題", "必要", "国際", "社会", "役割", "組織", "部局"]
CANDIDATE_FOCUS_TERMS = {
    "ohkoshi": ["創造", "学知", "価値", "技術", "構築", "支援"],
    "sugano": ["経営", "職員", "学長", "強化"],
    "someya": ["構成", "投資", "環境", "現場", "社会", "人材"],
    "fujigaki": ["学術", "科学", "壁", "環境", "人類", "情報"],
    "yamamoto": ["ガバナンス-governance", "部局", "プロセス-process", "組織", "課題", "法人"],
}
CANDIDATE_INTERPRETATIONS = {
    "ohkoshi": "大越慎一は、知の創造・価値創造・技術・構築・支援といった語が上位に並び、研究と教育を新しい学知や制度設計につなげる構想を前面に出している。",
    "sugano": "菅野暁は、経営・職員・学長・強化が目立ち、大学運営を人的資源とリーダーシップの設計問題として語る比重が大きい。",
    "someya": "染谷隆夫は、構成・投資・環境・現場・人材が上位に出ており、研究教育の場をどう整え、長期的に成長させるかという視点が強い。",
    "fujigaki": "藤垣裕子は、学術・科学・壁・環境・人類といった語を通じて、知の基盤や学内外のコミュニケーション障壁をどう越えるかを重視している。",
    "yamamoto": "山本隆司は、ガバナンス・部局・プロセス・組織・課題・法人が圧倒的に強く、大学の問題を統治構造と意思決定過程の設計として捉えている。",
}


class ImgSrcParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        for key, value in attrs:
            if key.lower() == "src" and value:
                self.sources.append(value)
                break


def pct(value: float) -> str:
    return f"{value:.1%}"


def fmt(value: float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.3f}"


def make_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, classes="table")


def encode_image_as_data_uri(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"report image not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        raise ValueError(f"unsupported image type for embedding: {path}")
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }[suffix]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def ensure_image_assets() -> None:
    IMAGE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for path in EMBEDDED_IMAGE_PATHS.values():
        target = IMAGE_ASSET_DIR / path.name
        if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
            shutil.copy2(path, target)


def build_embedded_image_sources() -> dict[str, str]:
    return {name: encode_image_as_data_uri(path) for name, path in EMBEDDED_IMAGE_PATHS.items()}


def build_image_sources(mode: str = "data-uri") -> dict[str, str]:
    if mode == "data-uri":
        return build_embedded_image_sources()
    if mode == "relative-files":
        ensure_image_assets()
        return {name: f"images/{path.name}" for name, path in EMBEDDED_IMAGE_PATHS.items()}
    raise ValueError(f"unsupported image source mode: {mode}")


def extract_img_sources(html_text: str) -> list[str]:
    parser = ImgSrcParser()
    parser.feed(html_text)
    return parser.sources


def validate_img_sources(html_text: str, html_path: Path) -> list[str]:
    invalid_sources: list[str] = []
    for src in extract_img_sources(html_text):
        if re.match(r"^[A-Za-z]:[\\/]", src) or src.startswith("file:///") or src.startswith("/") or src.startswith("\\"):
            invalid_sources.append(f"{src} (absolute local path is not portable)")
            continue
        if src.startswith("data:image/"):
            prefix, _, payload = src.partition(",")
            if ";base64" not in prefix or not payload:
                invalid_sources.append(f"{src[:80]}... (malformed data URI)")
                continue
            try:
                base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                invalid_sources.append(f"{src[:80]}... ({exc})")
            continue
        if re.match(r"^[a-z]+://", src):
            continue
        if src.startswith("images/") and not (html_path.parent / src).exists():
            invalid_sources.append(f"{src} (missing relative file)")
        elif not src.startswith("data:") and not src.startswith("images/") and not (html_path.parent / src).exists():
            invalid_sources.append(f"{src} (missing relative file)")
    return invalid_sources


def build_candidate_summary(candidate_stats: pd.DataFrame) -> pd.DataFrame:
    total_chars = candidate_stats["char_count"].sum()
    summary = candidate_stats.copy()
    summary["char_share"] = summary["char_count"] / total_chars
    summary = summary[
        [
            "candidate_name",
            "char_count",
            "char_share",
            "avg_chars_per_answer",
            "analysis_token_count_no_stopwords",
            "type_token_ratio",
        ]
    ].rename(
        columns={
            "candidate_name": "候補者",
            "char_count": "総文字数",
            "char_share": "文字数シェア",
            "avg_chars_per_answer": "1回答あたり平均文字数",
            "analysis_token_count_no_stopwords": "分析対象トークン数",
            "type_token_ratio": "TTR",
        }
    )
    summary["総文字数"] = summary["総文字数"].map(lambda x: f"{int(x):,}")
    summary["文字数シェア"] = summary["文字数シェア"].map(pct)
    summary["1回答あたり平均文字数"] = summary["1回答あたり平均文字数"].map(lambda x: f"{x:.1f}")
    summary["分析対象トークン数"] = summary["分析対象トークン数"].map(lambda x: f"{int(x):,}")
    summary["TTR"] = summary["TTR"].map(lambda x: f"{x:.3f}")
    return summary


def build_question_leaders(question_stats: pd.DataFrame) -> pd.DataFrame:
    leaders = (
        question_stats.sort_values(["question_id", "char_count"], ascending=[True, False], kind="stable")
        .groupby("question_id", as_index=False)
        .first()
    )
    leaders = leaders[
        ["question_id", "question_title", "candidate_name", "char_count"]
    ].rename(
        columns={
            "question_id": "設問ID",
            "question_title": "設問",
            "candidate_name": "最長回答者",
            "char_count": "文字数",
        }
    )
    leaders["文字数"] = leaders["文字数"].map(lambda x: f"{int(x):,}")
    return leaders


def build_shared_terms_table(common_terms: pd.DataFrame) -> pd.DataFrame:
    rows = common_terms[common_terms["term"].isin(SHARED_FOCUS_TERMS)].copy()
    rows = rows[
        [
            "term",
            "candidate_count",
            "total_frequency",
            "ohkoshi_frequency",
            "sugano_frequency",
            "someya_frequency",
            "fujigaki_frequency",
            "yamamoto_frequency",
        ]
    ].rename(
        columns={
            "term": "語",
            "candidate_count": "使用候補者数",
            "total_frequency": "総頻度",
            "ohkoshi_frequency": "大越",
            "sugano_frequency": "菅野",
            "someya_frequency": "染谷",
            "fujigaki_frequency": "藤垣",
            "yamamoto_frequency": "山本",
        }
    )
    for column in rows.columns[1:]:
        rows[column] = rows[column].map(lambda x: f"{int(x):,}")
    return rows


def build_focus_term_table(
    comparison: pd.DataFrame,
    log_odds_top: pd.DataFrame,
    candidate_stats: pd.DataFrame,
) -> pd.DataFrame:
    candidate_name_map = candidate_stats.set_index("candidate_id")["candidate_name"].to_dict()
    comparison_index = comparison.set_index(["candidate_name", "term"])
    log_index = log_odds_top[log_odds_top["direction"] == "positive"].set_index(["candidate_name", "term"])

    rows: list[dict[str, str]] = []
    for candidate_id, terms in CANDIDATE_FOCUS_TERMS.items():
        candidate_name = candidate_name_map[candidate_id]
        for term in terms:
            tfidf_rank = ""
            raw_frequency = ""
            z_score = ""
            log_odds_rank = ""
            if (candidate_name, term) in comparison_index.index:
                comp_row = comparison_index.loc[(candidate_name, term)]
                tfidf_rank = str(int(comp_row["tfidf_rank"]))
                raw_frequency = str(int(comp_row["raw_frequency"]))
                z_score = f"{float(comp_row['z_score']):.2f}"
                log_odds_rank = str(int(comp_row["log_odds_rank"]))
            elif (candidate_name, term) in log_index.index:
                log_row = log_index.loc[(candidate_name, term)]
                raw_frequency = str(int(log_row["target_count"]))
                z_score = f"{float(log_row['z_score']):.2f}"
                log_odds_rank = str(int(log_row["rank_within_direction"]))
            else:
                continue

            rows.append(
                {
                    "候補者": candidate_name,
                    "特徴語": term,
                    "頻度": raw_frequency,
                    "TF-IDF順位": tfidf_rank or "参考外",
                    "log-odds順位": log_odds_rank or "参考外",
                    "z-score": z_score or "参考外",
                }
            )
    return pd.DataFrame(rows)


def build_methods_table(candidate_stats: pd.DataFrame, question_stats: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    total_chars = int(candidate_stats["char_count"].sum())
    analyzed_tokens = int(candidate_stats["analysis_token_count_no_stopwords"].sum())
    question_count = int(question_stats["question_id"].nunique())
    document_count = int(question_stats.shape[0])
    yamamoto_q07 = load_dataframe(ROOT_DIR / "data" / "processed" / "candidate_answers.csv")
    yamamoto_q07_text = yamamoto_q07.loc[
        (yamamoto_q07["candidate_id"] == "yamamoto") & (yamamoto_q07["question_id"] == "q07"),
        "text",
    ].iloc[0]

    rows = [
        {"項目": "データ出典", "内容": "東京大学新聞社『総長選考2026 候補者紹介』の公開ページ"},
        {"項目": "対象", "内容": "第2次候補者5名"},
        {"項目": "設問数", "内容": f"{question_count}設問"},
        {"項目": "分析単位", "内容": f"candidate × question の {document_count} 文書"},
        {"項目": "コーパス規模", "内容": f"{total_chars:,}文字 / 分析対象トークン {analyzed_tokens:,}"},
        {"項目": "形態素解析", "内容": "MeCab + unidic-lite。主分析は名詞・動詞・形容詞を保持"},
        {"項目": "除外語の扱い", "内容": "大学・東京大学・本学・東大・研究・教育を stopword として別列で管理。レポート本文はその除外後結果を中心に使用"},
        {"項目": "共通語指標", "内容": "candidate prevalence = その語を1回以上使った候補者数 / 5"},
        {"項目": "共通課題指標", "内容": "KWIC、同文内共起、問題表現（必要・課題・問題など）との近接を併用"},
        {"項目": "特徴語指標", "内容": "TF-IDF（N = 5 documents）と weighted log-odds ratio を併用"},
        {"項目": "log-odds設定", "内容": "informative Dirichlet prior。prior_strength = 100.0"},
        {"項目": "注意点", "内容": "語彙の生データには『する』『ある』などの高頻度動詞も残るため、解釈節では内容語を選んで要約"},
        {"項目": "補足", "内容": f"山本隆司の q07 は『{html.escape(yamamoto_q07_text)}』で、自由記述は実質空欄"},
    ]
    return pd.DataFrame(rows)


def build_exec_summary(candidate_stats: pd.DataFrame) -> list[str]:
    max_row = candidate_stats.sort_values("char_count", ascending=False).iloc[0]
    min_row = candidate_stats.sort_values("char_count", ascending=True).iloc[0]
    ttr_row = candidate_stats.sort_values("type_token_ratio", ascending=False).iloc[0]
    return [
        f"分量面では山本隆司が {int(max_row['char_count']):,} 文字で最長、大越慎一が {int(min_row['char_count']):,} 文字で最短だった。",
        "設問別では山本隆司が q01・q02・q03・q04・q06 で最長回答者となり、ガバナンスや制度設計に関する設問で特に記述量が大きい。",
        "5候補に広く共有された内容語としては、課題・必要・国際・社会・役割・組織・部局が目立ち、大学の将来像を社会的課題と制度運営の両面から捉える傾向が共通していた。",
        "ただし共通語のなかでも『部局』『組織』『課題』は山本隆司への偏りが大きく、共有テーマの中にも候補者ごとの差が埋め込まれている。",
        f"語彙の多様性を示す TTR は大越慎一が {float(ttr_row['type_token_ratio']):.3f} で最も高く、短めの回答でも比較的多様な語を使っていた。",
    ]


def build_html() -> str:
    candidate_stats = load_dataframe(OUTPUT_DIR / "descriptive" / "candidate_stats.csv")
    question_stats = load_dataframe(OUTPUT_DIR / "descriptive" / "question_stats.csv")
    common_terms = load_dataframe(OUTPUT_DIR / "common_terms" / "common_terms.csv")
    comparison = load_dataframe(OUTPUT_DIR / "log_odds" / "tfidf_logodds_comparison.csv")
    log_odds_top = load_dataframe(OUTPUT_DIR / "log_odds" / "log_odds_top20_by_candidate.csv")
    tokens = load_dataframe(ROOT_DIR / "data" / "processed" / "tokens.csv")

    for column in ["char_count", "analysis_token_count_no_stopwords"]:
        candidate_stats[column] = candidate_stats[column].astype(int)
    candidate_stats["type_token_ratio"] = candidate_stats["type_token_ratio"].astype(float)
    candidate_stats["avg_chars_per_answer"] = candidate_stats["avg_chars_per_answer"].astype(float)

    question_stats["char_count"] = question_stats["char_count"].astype(int)

    candidate_summary = build_candidate_summary(candidate_stats)
    question_leaders = build_question_leaders(question_stats)
    shared_terms_table = build_shared_terms_table(common_terms)
    focus_term_table = build_focus_term_table(comparison, log_odds_top, candidate_stats)
    methods_table = build_methods_table(candidate_stats, question_stats, tokens)
    executive_summary = build_exec_summary(candidate_stats)
    image_sources = build_image_sources(mode="data-uri")

    interpretations = "".join(
        f"<li><strong>{html.escape(candidate_stats.set_index('candidate_id').loc[candidate_id, 'candidate_name'])}</strong>：{html.escape(text)}</li>"
        for candidate_id, text in CANDIDATE_INTERPRETATIONS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>総長候補者テキスト分析レポート</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --paper: #fffdf8;
      --ink: #1c2a39;
      --muted: #58677a;
      --line: #d7ccbb;
      --accent: #a43f2f;
      --accent-soft: #f2dfd7;
      --accent-deep: #7e2c20;
      --olive: #74845b;
      --shadow: 0 18px 60px rgba(28, 42, 57, 0.12);
      --radius: 20px;
      --content: 1180px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "BIZ UDPGothic", "Yu Gothic", "Hiragino Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(164, 63, 47, 0.09), transparent 32%),
        radial-gradient(circle at top right, rgba(116, 132, 91, 0.12), transparent 26%),
        linear-gradient(180deg, #faf6ef 0%, #f3ecdf 100%);
      line-height: 1.7;
    }}
    a {{ color: var(--accent-deep); }}
    .page {{
      width: min(var(--content), calc(100% - 32px));
      margin: 32px auto 56px;
    }}
    .hero {{
      background: linear-gradient(140deg, rgba(255,255,255,0.96), rgba(246,241,231,0.96));
      border: 1px solid rgba(164, 63, 47, 0.12);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(115deg, transparent 0 52%, rgba(164, 63, 47, 0.08) 52% 58%, transparent 58% 100%),
        linear-gradient(180deg, rgba(116, 132, 91, 0.06), transparent 40%);
      pointer-events: none;
    }}
    .hero-inner {{
      position: relative;
      padding: 40px 42px 34px;
    }}
    .eyebrow {{
      display: inline-block;
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-deep);
      background: var(--accent-soft);
      padding: 6px 10px;
      border-radius: 999px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.15;
      letter-spacing: -0.02em;
      font-family: "Yu Mincho", "Hiragino Mincho ProN", serif;
    }}
    .lead {{
      margin: 0;
      max-width: 840px;
      color: var(--muted);
      font-size: 1.03rem;
    }}
    .quick-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 28px;
    }}
    .stat {{
      background: rgba(255,255,255,0.84);
      border: 1px solid rgba(116, 132, 91, 0.16);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .stat .label {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat .value {{
      display: block;
      font-size: 1.55rem;
      font-weight: 700;
    }}
    .section {{
      margin-top: 28px;
      background: var(--paper);
      border: 1px solid rgba(28, 42, 57, 0.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 28px;
    }}
    .section h2 {{
      margin: 0 0 16px;
      font-size: 1.5rem;
      font-family: "Yu Mincho", "Hiragino Mincho ProN", serif;
    }}
    .section h3 {{
      margin: 24px 0 12px;
      font-size: 1.05rem;
    }}
    .section p {{
      margin: 0 0 14px;
      color: var(--ink);
    }}
    .section ul {{
      margin: 0;
      padding-left: 1.2rem;
    }}
    .note {{
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: #fbf1eb;
      border-radius: 12px;
      color: var(--muted);
      margin: 14px 0 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 14px;
    }}
    table.table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
      background: white;
      border-radius: 14px;
      overflow: hidden;
    }}
    .table th, .table td {{
      border-bottom: 1px solid #e8e0d2;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    .table th {{
      background: #f3ede2;
      color: var(--ink);
      position: sticky;
      top: 0;
    }}
    .table tr:nth-child(even) td {{
      background: #fffaf2;
    }}
    .grid {{
      display: grid;
      gap: 22px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    figure {{
      margin: 0;
    }}
    figure img {{
      width: 100%;
      display: block;
      border-radius: 16px;
      border: 1px solid rgba(28, 42, 57, 0.08);
      background: #fff;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .footer-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 18px;
      font-size: 0.95rem;
    }}
    .tag {{
      display: inline-block;
      background: #eef2e7;
      color: #2c3d23;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.8rem;
      margin-right: 6px;
    }}
    @media (max-width: 720px) {{
      .hero-inner {{ padding: 30px 22px 24px; }}
      .section {{ padding: 22px 18px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-inner">
        <div class="eyebrow">Candidate Text Analysis</div>
        <h1>総長候補者テキスト分析レポート</h1>
        <p class="lead">
          東京大学新聞社「総長選考2026 候補者紹介」に掲載された第2次候補者5名の所見文を対象に、
          共通論点と候補者ごとの差異を整理した。全文はローカルの raw HTML / raw txt / 解析 CSV に追跡できる。
        </p>
        <div class="quick-stats">
          <div class="stat"><span class="label">対象候補者</span><span class="value">5名</span></div>
          <div class="stat"><span class="label">設問数</span><span class="value">7問</span></div>
          <div class="stat"><span class="label">分析単位</span><span class="value">35文書</span></div>
          <div class="stat"><span class="label">総文字数</span><span class="value">{int(candidate_stats['char_count'].sum()):,}</span></div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>要約</h2>
      <ul>
        {"".join(f"<li>{html.escape(item)}</li>" for item in executive_summary)}
      </ul>
    </section>

    <section class="section">
      <h2>Data &amp; Method</h2>
      <p>
        データは候補者一覧ページから第2次候補者5名のみを抽出し、各候補者の詳細ページ本文を raw HTML と raw txt として保存した。
        質問単位の分析では <span class="tag">candidate × question</span> を基本単位とし、35文書の long format を作成した。
      </p>
      <div class="table-wrap">{make_table(methods_table)}</div>
      <p class="note">
        共通語の原データには、軽い正規化のみを行った結果として「する」「ある」などの一般的な動詞も残る。
        本レポートの解釈では、raw の透明性を維持したまま、内容理解に有効な名詞・複合名詞を中心に記述した。
      </p>
    </section>

    <section class="section">
      <h2>コーパスの概観</h2>
      <p>
        回答量には大きな差があり、山本隆司が最長、ついで菅野暁、藤垣裕子が続く。
        一方で大越慎一は最も短いが、TTR は5候補中で最も高く、限られた分量のなかで比較的多様な語彙を使っている。
      </p>
      <div class="table-wrap">{make_table(candidate_summary)}</div>
      <div class="grid" style="margin-top: 18px;">
        <figure>
          <img src="{image_sources['candidate_total_characters']}" alt="候補者別総文字数">
          <figcaption>候補者別総文字数。山本隆司の記述量が突出している。</figcaption>
        </figure>
        <figure>
          <img src="{image_sources['question_answer_length_heatmap']}" alt="質問×候補者 回答文字数">
          <figcaption>質問×候補者の回答文字数。q04 と q06 で山本隆司の比重が特に大きい。</figcaption>
        </figure>
      </div>
      <h3>設問別に最も長かった回答</h3>
      <div class="table-wrap">{make_table(question_leaders)}</div>
    </section>

    <section class="section">
      <h2>共通認識</h2>
      <p>
        5候補が広く共有した内容語を見ると、全員が東京大学の将来を「課題」への対応として捉え、
        その対応を「国際」「社会」「役割」と結びつけて説明していることが分かる。
        加えて「組織」「部局」が5候補すべてで確認されるため、教育研究の中身だけでなく、
        それを支える制度や運営体制も共通論点になっている。
      </p>
      <div class="table-wrap">{make_table(shared_terms_table)}</div>
      <p>
        ただし、共通語が即座に共通課題を意味するわけではない。KWIC と共起を合わせて見ると、
        「課題」は世界情勢・大学財政・研究環境・ガバナンスといった複数の局面をまたいで使われていた。
        「部局」「組織」は全候補に見られる一方、山本隆司に大きく集中しており、
        共通の語であっても焦点の当て方には明確な差がある。
      </p>
      <p>
        したがって、本データから言えるのは「5候補とも、東京大学の将来を社会的・国際的課題への対応として語る」という点と、
        「その対応を支える組織設計や部局間の関係調整が、多くの候補で重要テーマになっている」という点である。
      </p>
    </section>

    <section class="section">
      <h2>候補者ごとの差異</h2>
      <p>
        候補者差の把握には、TF-IDF と weighted log-odds の両方で上位に入る語を優先した。
        こうすることで、単に1回だけ現れたレア語ではなく、その候補に比較的安定して偏る語を抽出している。
      </p>
      <div class="table-wrap">{make_table(focus_term_table)}</div>
      <h3>読み取り</h3>
      <ul>{interpretations}</ul>
      <p>
        まとめると、5候補とも課題認識を共有しつつ、強調点はかなり異なる。
        大越は知の創造と制度構築、菅野は経営と職員・学長の役割、染谷は環境整備と投資、藤垣は学術基盤と壁の克服、
        山本はガバナンスと組織過程に最も強く重心を置いている。
      </p>
    </section>

    <section class="section">
      <h2>解釈上の留意点</h2>
      <ul>
        <li>TF-IDF の候補者単位分析は N = 5 のため、探索用途と位置づけるべきである。</li>
        <li>MeCab の辞書形には語種によって英語グロス付き表記（例: ガバナンス-governance, プロセス-process）が混ざる。</li>
        <li>共通語の raw 出力には一般動詞も残るため、厳密な読解には <a href="../output/common_issues/kwic.csv">KWIC</a> と <a href="../output/common_issues/common_issue_review_template.csv">レビュー用テンプレート</a> で原文に戻る必要がある。</li>
        <li>山本隆司の q07 は「記述なし」であり、自由記述欄の比較は他4候補と完全には対称でない。</li>
      </ul>
    </section>

    <section class="section">
      <h2>参照ファイル</h2>
      <p>このレポートはローカルの生成物から再構築できる。</p>
      <div class="footer-links">
        <a href="../data/metadata/candidates.csv">候補者一覧 CSV</a>
        <a href="../data/processed/candidate_answers.csv">質問分割済み回答 CSV</a>
        <a href="../data/processed/tokens.csv">形態素解析結果</a>
        <a href="../output/common_terms/common_terms.csv">共通語一覧</a>
        <a href="../output/common_issues/issue_candidates.csv">共通課題候補</a>
        <a href="../output/log_odds/log_odds_top20_by_candidate.csv">log-odds 上位語</a>
        <a href="../output/log_odds/tfidf_logodds_comparison.csv">TF-IDF / log-odds 比較表</a>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    ensure_project_dirs()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_image_assets()
    html_text = build_html()
    for report_path in REPORT_PATHS:
        invalid_sources = validate_img_sources(html_text, report_path)
        if invalid_sources:
            joined = "\n".join(f"- {item}" for item in invalid_sources)
            raise ValueError(f"invalid image references in {report_path}:\n{joined}")
        report_path.write_text(html_text, encoding="utf-8")
        print(f"wrote {report_path}")
    print(f"validated {len(extract_img_sources(html_text))} image references")


if __name__ == "__main__":
    main()
