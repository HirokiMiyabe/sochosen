from __future__ import annotations

import argparse
import math
from collections import Counter

import pandas as pd

from utils import METADATA_DIR, OUTPUT_DIR, ensure_project_dirs, load_dataframe, save_dataframe


def compute_tfidf(documents: list[dict[str, object]]) -> pd.DataFrame:
    doc_counters = {doc["doc_id"]: Counter(doc["tokens"]) for doc in documents}
    document_frequency: Counter[str] = Counter()
    for counter in doc_counters.values():
        for term in counter:
            document_frequency[term] += 1

    rows: list[dict[str, object]] = []
    n_docs = len(documents)
    for document in documents:
        doc_id = document["doc_id"]
        counter = doc_counters[doc_id]
        total_terms = sum(counter.values())
        for term, raw_frequency in counter.items():
            tf = raw_frequency / total_terms if total_terms else 0.0
            idf = math.log((1 + n_docs) / (1 + document_frequency[term])) + 1.0
            rows.append(
                {
                    **{key: value for key, value in document.items() if key != "tokens"},
                    "term": term,
                    "tf": tf,
                    "idf": idf,
                    "tfidf": tf * idf,
                    "raw_frequency": raw_frequency,
                    "candidate_count": document_frequency[term],
                    "total_terms": total_terms,
                }
            )
    return pd.DataFrame(rows)


def build_candidate_documents(tokens: pd.DataFrame, metadata: pd.DataFrame, term_column: str) -> list[dict[str, object]]:
    candidate_order = metadata["candidate_id"].tolist()
    candidate_name_map = metadata.set_index("candidate_id")["candidate_name"].to_dict()
    filtered = tokens[tokens[term_column] != ""].copy()
    filtered["question_order"] = filtered["question_order"].astype(int)
    filtered["token_index"] = filtered["token_index"].astype(int)
    filtered = filtered.sort_values(by=["candidate_id", "question_order", "token_index"], kind="stable")

    documents: list[dict[str, object]] = []
    for candidate_id in candidate_order:
        candidate_tokens = filtered.loc[filtered["candidate_id"] == candidate_id, term_column].tolist()
        documents.append(
            {
                "doc_id": candidate_id,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name_map[candidate_id],
                "level": "candidate",
                "term_column": term_column,
                "tokens": candidate_tokens,
            }
        )
    return documents


def build_question_documents(tokens: pd.DataFrame, answers: pd.DataFrame, term_column: str) -> list[dict[str, object]]:
    filtered = tokens[tokens[term_column] != ""].copy()
    filtered["question_order"] = filtered["question_order"].astype(int)
    filtered["token_index"] = filtered["token_index"].astype(int)
    filtered = filtered.sort_values(
        by=["question_id", "candidate_id", "token_index"],
        kind="stable",
    )

    question_metadata = (
        answers[["question_id", "question_order", "question_title"]]
        .drop_duplicates(subset=["question_id"])
        .assign(question_order=lambda df: df["question_order"].astype(int))
        .sort_values("question_order")
    )

    documents: list[dict[str, object]] = []
    for question in question_metadata.to_dict("records"):
        question_tokens = filtered[filtered["question_id"] == question["question_id"]]
        for (candidate_id, candidate_name), group in question_tokens.groupby(
            ["candidate_id", "candidate_name"], sort=False
        ):
            documents.append(
                {
                    "doc_id": f"{candidate_id}_{question['question_id']}",
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "question_id": question["question_id"],
                    "question_order": question["question_order"],
                    "question_title": question["question_title"],
                    "level": "candidate_question",
                    "term_column": term_column,
                    "tokens": group[term_column].tolist(),
                }
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--term-column",
        default="analysis_term_no_stopwords",
        choices=["analysis_term", "analysis_term_no_stopwords", "noun_term", "noun_term_no_stopwords"],
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    ensure_project_dirs()
    metadata = load_dataframe(METADATA_DIR / "candidates.csv")
    answers = load_dataframe("data/processed/candidate_answers.csv")
    tokens = load_dataframe("data/processed/tokens.csv")

    if answers.empty or tokens.empty:
        raise RuntimeError("入力データが不足しています。先に 01〜03 のスクリプトを実行してください。")

    candidate_documents = build_candidate_documents(tokens, metadata, args.term_column)
    candidate_tfidf = compute_tfidf(candidate_documents).sort_values(
        by=["candidate_id", "tfidf", "raw_frequency", "term"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    candidate_tfidf["rank"] = candidate_tfidf.groupby("candidate_id").cumcount() + 1

    question_documents = build_question_documents(tokens, answers, args.term_column)
    question_frames: list[pd.DataFrame] = []
    for question_id, docs in pd.Series(question_documents).groupby(
        [doc["question_id"] for doc in question_documents], sort=False
    ):
        question_df = compute_tfidf(list(docs))
        question_frames.append(question_df)
    question_tfidf = pd.concat(question_frames, ignore_index=True).sort_values(
        by=["question_order", "candidate_id", "tfidf", "raw_frequency", "term"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    question_tfidf["rank_within_question_candidate"] = question_tfidf.groupby(
        ["question_id", "candidate_id"]
    ).cumcount() + 1

    output_dir = OUTPUT_DIR / "tfidf"
    save_dataframe(candidate_tfidf, output_dir / "tfidf_all.csv")
    save_dataframe(candidate_tfidf[candidate_tfidf["rank"] <= args.top_k], output_dir / "tfidf_top20_by_candidate.csv")
    save_dataframe(question_tfidf, output_dir / "tfidf_by_question.csv")

    print(candidate_tfidf[candidate_tfidf["rank"] <= 10][["candidate_name", "term", "tfidf", "raw_frequency"]].to_string(index=False))


if __name__ == "__main__":
    main()
