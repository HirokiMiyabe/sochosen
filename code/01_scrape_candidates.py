from __future__ import annotations

from collections import OrderedDict
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from utils import (
    CANDIDATE_SOURCE_URL,
    EXPECTED_SECOND_STAGE_NAMES,
    METADATA_DIR,
    QUESTION_SECTION_MARKER,
    RAW_HTML_DIR,
    RAW_TXT_DIR,
    build_session,
    ensure_project_dirs,
    normalize_text,
    save_dataframe,
    save_text,
    slug_from_url,
)


def extract_second_stage_candidates(listing_html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(listing_html, "html.parser")
    content = soup.select_one(".entry-content") or soup
    second_heading = content.find(
        lambda tag: tag.name in {"h2", "h3"} and "第2次候補者" in tag.get_text(" ", strip=True)
    )
    first_heading = content.find(
        lambda tag: tag.name in {"h2", "h3"} and "第1次候補者" in tag.get_text(" ", strip=True)
    )
    if second_heading is None or first_heading is None:
        raise RuntimeError("候補者一覧ページから第2次候補者セクションを特定できませんでした。")

    seen: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    current = second_heading
    while current is not None and current != first_heading:
        if getattr(current, "name", None) == "a" and current.has_attr("href"):
            candidate_text = normalize_text(current.get_text(" ", strip=True))
            if candidate_text.endswith("候補"):
                candidate_name = candidate_text.removesuffix("候補").strip()
                if candidate_name in EXPECTED_SECOND_STAGE_NAMES:
                    url = urljoin(CANDIDATE_SOURCE_URL, current["href"])
                    seen[url] = {
                        "candidate_id": slug_from_url(url),
                        "candidate_name": candidate_name,
                        "url": url,
                    }
        current = current.find_next()

    candidates = list(seen.values())
    extracted_names = {candidate["candidate_name"] for candidate in candidates}
    expected_names = set(EXPECTED_SECOND_STAGE_NAMES)
    if extracted_names != expected_names:
        raise RuntimeError(
            "第2次候補者の抽出結果が期待と一致しません。"
            f" expected={sorted(expected_names)} extracted={sorted(extracted_names)}"
        )
    return candidates


def extract_candidate_text(candidate_html: str) -> str:
    soup = BeautifulSoup(candidate_html, "html.parser")
    content = soup.select_one(".entry-content")
    if content is None:
        raise RuntimeError("候補者ページの本文領域(.entry-content)を取得できませんでした。")

    section_heading = content.find(
        lambda tag: tag.name == "h2" and QUESTION_SECTION_MARKER in tag.get_text(" ", strip=True)
    )
    if section_heading is None:
        raise RuntimeError(f"候補者ページから '{QUESTION_SECTION_MARKER}' セクションを見つけられません。")

    lines: list[str] = []
    current = section_heading.find_next("p")
    while current is not None:
        text = normalize_text(current.get_text(" ", strip=True))
        if text:
            lines.append(text)
        current = current.find_next_sibling("p")

    if not lines:
        raise RuntimeError("候補者本文が空でした。")
    return "\n\n".join(lines)


def main() -> None:
    ensure_project_dirs()
    session = build_session()
    try:
        response = session.get(CANDIDATE_SOURCE_URL, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        listing_html = response.text
        candidates = extract_second_stage_candidates(listing_html)
        save_text(RAW_HTML_DIR / "candidate_index.html", listing_html)
        save_dataframe(pd.DataFrame(candidates), METADATA_DIR / "candidates.csv")

        print(f"第2次候補者 {len(candidates)} 名を抽出しました。")
        for candidate in candidates:
            html_response = session.get(candidate["url"], timeout=30)
            html_response.raise_for_status()
            html_response.encoding = html_response.apparent_encoding or html_response.encoding or "utf-8"
            candidate_html = html_response.text
            candidate_id = candidate["candidate_id"]
            candidate_text = extract_candidate_text(candidate_html)
            save_text(RAW_HTML_DIR / f"{candidate_id}.html", candidate_html)
            save_text(RAW_TXT_DIR / f"{candidate_id}.txt", candidate_text)
            print(
                f"- {candidate['candidate_name']} ({candidate_id}): "
                f"{len(candidate_text)} 文字 / {candidate['url']}"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
