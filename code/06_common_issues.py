from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from utils import OUTPUT_DIR, REVIEW_TRIGGER_WORDS, ensure_project_dirs, load_dataframe, save_dataframe, truncate_text


WINDOW_SIZE = 5


def sentence_contains_trigger(sentence: str) -> bool:
    return any(trigger in sentence for trigger in REVIEW_TRIGGER_WORDS)


def main() -> None:
    ensure_project_dirs()
    common_terms = load_dataframe(OUTPUT_DIR / "common_terms" / "common_terms_4of5_or_more.csv")
    tokens = load_dataframe("data/processed/tokens.csv")
    if common_terms.empty or tokens.empty:
        raise RuntimeError("必要な入力ファイルが不足しています。先に 03〜05 のスクリプトを実行してください。")

    tokens["sentence_index"] = tokens["sentence_index"].astype(int)
    tokens["token_index"] = tokens["token_index"].astype(int)
    tokens["question_order"] = tokens["question_order"].astype(int)
    tokens["is_noise"] = tokens["is_noise"].map({"True": True, "False": False})

    selected_terms_df = common_terms[common_terms["total_frequency"].astype(int) >= 3].copy()
    if selected_terms_df.empty:
        selected_terms_df = common_terms.copy()
    selected_terms = set(selected_terms_df["term"])

    usable_tokens = tokens[~tokens["is_noise"]].copy()
    usable_tokens = usable_tokens.sort_values(
        by=["candidate_id", "question_order", "sentence_index", "token_index"],
        kind="stable",
    )

    kwic_rows: list[dict[str, str]] = []
    for _, group in usable_tokens.groupby("answer_id", sort=False):
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            term = row["analysis_term_no_stopwords"]
            if term not in selected_terms:
                continue
            left_context = " ".join(group.iloc[max(0, index - WINDOW_SIZE) : index]["surface"].tolist())
            right_context = " ".join(group.iloc[index + 1 : index + 1 + WINDOW_SIZE]["surface"].tolist())
            kwic_rows.append(
                {
                    "term": term,
                    "candidate_id": row["candidate_id"],
                    "candidate_name": row["candidate_name"],
                    "question_id": row["question_id"],
                    "question_title": row["question_title"],
                    "left_context": left_context,
                    "keyword": row["surface"],
                    "right_context": right_context,
                    "sentence": row["sentence_text"],
                }
            )

    kwic_df = pd.DataFrame(kwic_rows).sort_values(
        by=["term", "candidate_id", "question_id"], kind="stable"
    )
    save_dataframe(kwic_df, OUTPUT_DIR / "common_issues" / "kwic.csv")

    analysis_tokens = usable_tokens[usable_tokens["analysis_term_no_stopwords"] != ""].copy()
    term_sentence_frequency: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_candidate_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    grouped_sentences = analysis_tokens.groupby(
        ["candidate_id", "candidate_name", "question_id", "sentence_index", "sentence_text"],
        sort=False,
    )
    for (candidate_id, _, _, _, _), sentence_group in grouped_sentences:
        unique_terms = sorted(set(sentence_group["analysis_term_no_stopwords"].tolist()))
        selected_in_sentence = [term for term in unique_terms if term in selected_terms]
        for term in unique_terms:
            term_sentence_frequency[term] += 1
        for term in selected_in_sentence:
            for co_term in unique_terms:
                if co_term == term:
                    continue
                pair = (term, co_term)
                pair_counts[pair] += 1
                pair_candidate_sets[pair].add(candidate_id)

    cooccurrence_rows: list[dict[str, object]] = []
    for (term, co_term), count in pair_counts.items():
        dice = 2 * count / (term_sentence_frequency[term] + term_sentence_frequency[co_term])
        cooccurrence_rows.append(
            {
                "term": term,
                "co_term": co_term,
                "cooccurrence_count": count,
                "term_sentence_frequency": term_sentence_frequency[term],
                "co_term_sentence_frequency": term_sentence_frequency[co_term],
                "candidate_count": len(pair_candidate_sets[(term, co_term)]),
                "dice": round(dice, 6),
            }
        )
    cooccurrence_df = pd.DataFrame(cooccurrence_rows).sort_values(
        by=["term", "cooccurrence_count", "dice", "co_term"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    save_dataframe(cooccurrence_df, OUTPUT_DIR / "common_issues" / "cooccurrence_sentence.csv")

    heuristic_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for term_row in selected_terms_df.to_dict("records"):
        term = term_row["term"]
        term_kwic = kwic_df[kwic_df["term"] == term]
        trigger_sentence_count = int(term_kwic["sentence"].apply(sentence_contains_trigger).sum())
        top_related = cooccurrence_df[cooccurrence_df["term"] == term].head(5)
        related_terms = " / ".join(
            f"{row.co_term}({row.cooccurrence_count})" for row in top_related.itertuples()
        )
        representative_contexts = []
        for candidate_name, candidate_group in term_kwic.groupby("candidate_name", sort=False):
            sentence = candidate_group.iloc[0]["sentence"]
            representative_contexts.append(f"{candidate_name}: {truncate_text(sentence, 70)}")

        heuristic_score = (
            int(term_row["candidate_count"]) * 100
            + int(term_row["total_frequency"]) * 10
            + trigger_sentence_count
        )
        heuristic_rows.append(
            {
                "term": term,
                "candidate_count": int(term_row["candidate_count"]),
                "total_frequency": int(term_row["total_frequency"]),
                "trigger_sentence_count": trigger_sentence_count,
                "top_related_terms": related_terms,
                "heuristic_score": heuristic_score,
            }
        )
        review_rows.append(
            {
                "term": term,
                "言及候補数": int(term_row["candidate_count"]),
                "総頻度": int(term_row["total_frequency"]),
                "関連語": related_terms,
                "各候補の代表文脈": " / ".join(representative_contexts),
                "共通課題": "",
                "解釈メモ": "",
            }
        )

    issue_candidates_df = pd.DataFrame(heuristic_rows).sort_values(
        by=["heuristic_score", "total_frequency", "term"],
        ascending=[False, False, True],
        kind="stable",
    )
    review_template_df = pd.DataFrame(review_rows).sort_values(
        by=["言及候補数", "総頻度", "term"],
        ascending=[False, False, True],
        kind="stable",
    )

    save_dataframe(issue_candidates_df, OUTPUT_DIR / "common_issues" / "issue_candidates.csv")
    save_dataframe(review_template_df, OUTPUT_DIR / "common_issues" / "common_issue_review_template.csv")

    print(issue_candidates_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
