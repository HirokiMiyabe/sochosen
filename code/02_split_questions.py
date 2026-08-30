from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from utils import (
    METADATA_DIR,
    PROCESSED_ANSWERS_DIR,
    PROCESSED_DIR,
    QUESTION_SECTION_MARKER,
    RAW_HTML_DIR,
    RAW_TXT_DIR,
    clean_question_title,
    ensure_project_dirs,
    load_dataframe,
    normalize_for_comparison,
    normalize_text,
    read_text,
    save_dataframe,
    save_text,
    truncate_text,
)


def extract_question_sections_from_html(candidate_html: str) -> list[list[str]]:
    soup = BeautifulSoup(candidate_html, "html.parser")
    content = soup.select_one(".entry-content")
    if content is None:
        raise RuntimeError("候補者ページの本文領域(.entry-content)を取得できませんでした。")

    section_heading = content.find(
        lambda tag: tag.name == "h2" and QUESTION_SECTION_MARKER in tag.get_text(" ", strip=True)
    )
    if section_heading is None:
        raise RuntimeError(f"候補者ページから '{QUESTION_SECTION_MARKER}' セクションを見つけられません。")

    sections: list[list[str]] = []
    current_section: list[str] = []
    current = section_heading.find_next_sibling("p")
    while current is not None:
        text = normalize_text(current.get_text(" ", strip=True))
        if text:
            current_section.append(text)
        elif current_section:
            sections.append(current_section)
            current_section = []
        current = current.find_next_sibling("p")

    if current_section:
        sections.append(current_section)
    return sections


def parse_candidate_text(raw_text: str) -> list[dict[str, str]]:
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    records: list[dict[str, str]] = []
    current_question: str | None = None
    current_parts: list[str] = []

    def flush_current() -> None:
        if current_question is None:
            return
        answer_text = normalize_text("\n\n".join(current_parts))
        records.append(
            {
                "question_title": current_question,
                "text": answer_text,
            }
        )

    for block in blocks:
        if block.startswith("──"):
            flush_current()
            current_question = clean_question_title(block)
            current_parts = []
            continue
        if current_question is None:
            continue
        current_parts.append(block)

    flush_current()
    return records


def parse_candidate_sections(
    sections: list[list[str]],
    fallback_titles: list[str] | None = None,
) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    has_explicit_titles = any(section and section[0].startswith("──") for section in sections)
    if has_explicit_titles:
        for section in sections:
            if not section:
                continue
            if section[0].startswith("──"):
                question_title = clean_question_title(section[0])
                answer_parts = section[1:]
            else:
                question_title = ""
                answer_parts = section
            if not question_title:
                raise RuntimeError("質問見出しのないセクションが混在しているため、自動分割できませんでした。")
            parsed.append(
                {
                    "question_title": question_title,
                    "text": normalize_text("\n\n".join(answer_parts)),
                }
            )
        return parsed

    if fallback_titles is None or len(sections) != len(fallback_titles):
        raise RuntimeError("質問見出しなしセクションを既知の設問順に対応付けできませんでした。")

    for question_title, section in zip(fallback_titles, sections, strict=True):
        parsed.append(
            {
                "question_title": question_title,
                "text": normalize_text("\n\n".join(section)),
            }
        )
    return parsed


def build_question_id_map(first_candidate_records: list[dict[str, str]]) -> dict[str, str]:
    return {
        normalize_for_comparison(record["question_title"]): f"q{index:02d}"
        for index, record in enumerate(first_candidate_records, start=1)
    }


def main() -> None:
    ensure_project_dirs()
    metadata = load_dataframe(METADATA_DIR / "candidates.csv")
    if metadata.empty:
        raise RuntimeError("data/metadata/candidates.csv が空です。先に 01_scrape_candidates.py を実行してください。")

    all_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    canonical_titles: list[str] = []
    question_id_map: dict[str, str] = {}

    for candidate_index, candidate in enumerate(metadata.to_dict("records")):
        candidate_id = candidate["candidate_id"]
        candidate_name = candidate["candidate_name"]
        raw_path = RAW_TXT_DIR / f"{candidate_id}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} が見つかりません。")

        html_path = RAW_HTML_DIR / f"{candidate_id}.html"
        if html_path.exists():
            sections = extract_question_sections_from_html(read_text(html_path))
            parsed = parse_candidate_sections(
                sections,
                fallback_titles=canonical_titles if canonical_titles else None,
            )
        else:
            parsed = parse_candidate_text(read_text(raw_path))
        if not parsed:
            raise RuntimeError(f"{candidate_name} の質問分割に失敗しました。")

        if candidate_index == 0:
            canonical_titles = [record["question_title"] for record in parsed]
            question_id_map = build_question_id_map(parsed)

        questions_match_canonical = len(parsed) == len(canonical_titles)
        if questions_match_canonical:
            for order, record in enumerate(parsed):
                questions_match_canonical = questions_match_canonical and (
                    normalize_for_comparison(record["question_title"])
                    == normalize_for_comparison(canonical_titles[order])
                )

        missing_answer_count = 0
        for order, record in enumerate(parsed, start=1):
            question_title = record["question_title"]
            normalized_title = normalize_for_comparison(question_title)
            question_id = question_id_map.get(normalized_title, f"q{order:02d}")
            answer_text = normalize_text(record["text"])
            if not answer_text:
                missing_answer_count += 1
            all_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "question_id": question_id,
                    "question_order": str(order),
                    "question_title": question_title,
                    "text": answer_text,
                }
            )
            save_text(PROCESSED_ANSWERS_DIR / f"{candidate_id}_{question_id}.txt", answer_text)

        validation_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "question_count": str(len(parsed)),
                "questions_match_canonical": str(questions_match_canonical),
                "missing_answer_count": str(missing_answer_count),
                "first_answer_preview": truncate_text(parsed[0]["text"], limit=200),
                "last_answer_preview": truncate_text(parsed[-1]["text"], limit=200),
            }
        )

        print(
            f"{candidate_name}: {len(parsed)} 問, 欠損 {missing_answer_count}, "
            f"順序一致={questions_match_canonical}"
        )
        print(f"  先頭 preview: {truncate_text(parsed[0]['text'], limit=200)}")
        print(f"  末尾 preview: {truncate_text(parsed[-1]['text'], limit=200)}")

    answers_df = pd.DataFrame(all_rows).sort_values(
        by=["question_id", "candidate_id"], kind="stable"
    )
    save_dataframe(answers_df, PROCESSED_DIR / "candidate_answers.csv")
    save_dataframe(pd.DataFrame(validation_rows), PROCESSED_DIR / "question_validation.csv")

    candidate_count = answers_df["candidate_id"].nunique()
    question_counts = answers_df.groupby("candidate_id")["question_id"].nunique().to_dict()
    print(f"候補者数: {candidate_count}")
    print(f"候補者別質問数: {question_counts}")


if __name__ == "__main__":
    main()
