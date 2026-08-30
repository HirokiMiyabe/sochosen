from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
DATA_DIR = ROOT_DIR / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_TXT_DIR = DATA_DIR / "raw" / "txt"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_ANSWERS_DIR = PROCESSED_DIR / "answers"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATE_SOURCE_URL = "https://news.todaishimbun.org/candidate"
EXPECTED_SECOND_STAGE_NAMES = [
    "大越慎一",
    "菅野暁",
    "染谷隆夫",
    "藤垣裕子",
    "山本隆司",
]
QUESTION_SECTION_MARKER = "教育、研究、運営・経営等に関する所見"
QUESTION_PREFIX = "──"
CONTENT_POS = {"名詞", "動詞", "形容詞"}
BASE_STOPWORDS = {
    "大学",
    "東京大学",
    "本学",
    "東大",
    "研究",
    "教育",
}
REVIEW_TRIGGER_WORDS = {
    "必要",
    "課題",
    "問題",
    "不足",
    "困難",
    "低下",
    "強化",
    "改善",
    "確保",
    "改革",
}
USER_AGENT = (
    "sochosen/1.0 "
    "(research script for public candidate statements; contact via local workspace)"
)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_QUESTION_PREFIX_RE = re.compile(r"^[\u2500\u2014\u2015\-]+\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[0-9０-９]+(?:[.,．，][0-9０-９]+)*$")


def ensure_project_dirs() -> None:
    for path in [
        CODE_DIR,
        RAW_HTML_DIR,
        RAW_TXT_DIR,
        PROCESSED_ANSWERS_DIR,
        METADATA_DIR,
        OUTPUT_DIR / "descriptive",
        OUTPUT_DIR / "common_terms",
        OUTPUT_DIR / "common_issues",
        OUTPUT_DIR / "tfidf",
        OUTPUT_DIR / "log_odds",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        }
    )
    return session


def fetch_html(url: str, session: requests.Session | None = None, timeout: int = 30) -> str:
    owns_session = session is None
    session = session or build_session()
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        return response.text
    finally:
        if owns_session:
            session.close()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_for_comparison(text: str) -> str:
    text = normalize_text(text).lower()
    text = _QUESTION_PREFIX_RE.sub("", text)
    return _MULTI_SPACE_RE.sub("", text)


def clean_question_title(text: str) -> str:
    return normalize_text(_QUESTION_PREFIX_RE.sub("", text))


def normalize_token(text: str) -> str:
    text = normalize_text(text).lower()
    return _MULTI_SPACE_RE.sub("", text)


def is_noise_token(token: str) -> bool:
    if not token:
        return True
    if _URL_RE.fullmatch(token):
        return True
    if _NUMBER_RE.fullmatch(token):
        return True
    if all(unicodedata.category(char)[0] in {"P", "S", "Z"} for char in token):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    rough_parts = re.split(r"(?<=[。！？!?])\s+", text)
    sentences: list[str] = []
    for part in rough_parts:
        for line in part.splitlines():
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def get_mecab_tagger():
    from pathlib import Path as _Path

    import MeCab
    import unidic_lite

    dicdir = _Path(unidic_lite.DICDIR).as_posix()
    tagger = MeCab.Tagger(f'-d "{dicdir}"')
    tagger.parse("")
    return tagger


def _choose_feature(parts: list[str], indexes: Iterable[int], fallback: str) -> str:
    for index in indexes:
        if index < len(parts):
            value = parts[index].strip()
            if value and value != "*":
                return value
    return fallback


def parse_mecab_feature(surface: str, feature: str) -> dict[str, str]:
    parts = feature.split(",")
    pos = parts[0] if parts else ""
    subpos = parts[1] if len(parts) > 1 else ""
    lemma = _choose_feature(parts, indexes=[7, 6, 10, 8], fallback=surface)
    reading = _choose_feature(parts, indexes=[9, 6], fallback="")
    return {
        "surface": surface,
        "pos": pos,
        "subpos": subpos,
        "lemma": lemma,
        "reading": reading,
        "normalized": normalize_token(lemma or surface),
    }


def tokenize_answer(text: str, tagger) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    token_index = 0
    for sentence_index, sentence_text in enumerate(split_sentences(text), start=1):
        node = tagger.parseToNode(sentence_text)
        while node:
            if node.surface:
                token_index += 1
                parsed = parse_mecab_feature(node.surface, node.feature)
                normalized = parsed["normalized"]
                is_content_pos = parsed["pos"] in CONTENT_POS
                is_noun = parsed["pos"] == "名詞"
                is_stopword = normalized in BASE_STOPWORDS
                is_noise = is_noise_token(normalized)
                rows.append(
                    {
                        "sentence_index": sentence_index,
                        "token_index": token_index,
                        "sentence_text": sentence_text,
                        "surface": parsed["surface"],
                        "lemma": parsed["lemma"],
                        "reading": parsed["reading"],
                        "pos": parsed["pos"],
                        "subpos": parsed["subpos"],
                        "normalized_form": normalized,
                        "is_content_pos": is_content_pos,
                        "is_noun": is_noun,
                        "is_stopword": is_stopword,
                        "is_noise": is_noise,
                        "analysis_term": normalized if is_content_pos and not is_noise else "",
                        "analysis_term_no_stopwords": (
                            normalized
                            if is_content_pos and not is_noise and not is_stopword
                            else ""
                        ),
                        "noun_term": normalized if is_noun and not is_noise else "",
                        "noun_term_no_stopwords": (
                            normalized if is_noun and not is_noise and not is_stopword else ""
                        ),
                    }
                )
            node = node.next
    return rows


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    preferred_fonts = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "BIZ UDGothic",
        "MS Gothic",
        "Noto Sans CJK JP",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    return plt


def truncate_text(text: str, limit: int = 80) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit - 1]}…"

