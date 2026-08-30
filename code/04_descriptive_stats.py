from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import METADATA_DIR, OUTPUT_DIR, configure_matplotlib, ensure_project_dirs, load_dataframe, save_dataframe


def main() -> None:
    ensure_project_dirs()
    metadata = load_dataframe(METADATA_DIR / "candidates.csv")
    answers = load_dataframe(Path("data/processed/candidate_answers.csv"))
    tokens = load_dataframe(Path("data/processed/tokens.csv"))

    if answers.empty or tokens.empty:
        raise RuntimeError("入力データが不足しています。01〜03 のスクリプトを先に実行してください。")

    candidate_order = metadata["candidate_id"].tolist()
    candidate_name_map = metadata.set_index("candidate_id")["candidate_name"].to_dict()

    answers["char_count"] = answers["text"].str.len()
    tokens["token_index"] = tokens["token_index"].astype(int)
    answers["question_order"] = answers["question_order"].astype(int)

    candidate_stats = metadata.copy()
    answer_counts = answers.groupby("candidate_id").size().rename("answer_count")
    char_counts = answers.groupby("candidate_id")["char_count"].sum().rename("char_count")
    total_tokens = tokens.groupby("candidate_id").size().rename("total_token_count")
    analysis_tokens = tokens[tokens["analysis_term"] != ""].groupby("candidate_id").size().rename(
        "analysis_token_count"
    )
    analysis_tokens_no_stopwords = (
        tokens[tokens["analysis_term_no_stopwords"] != ""]
        .groupby("candidate_id")
        .size()
        .rename("analysis_token_count_no_stopwords")
    )
    noun_tokens_no_stopwords = (
        tokens[tokens["noun_term_no_stopwords"] != ""]
        .groupby("candidate_id")
        .size()
        .rename("noun_token_count_no_stopwords")
    )
    unique_terms = (
        tokens[tokens["analysis_term"] != ""]
        .groupby("candidate_id")["analysis_term"]
        .nunique()
        .rename("unique_terms")
    )
    unique_terms_no_stopwords = (
        tokens[tokens["analysis_term_no_stopwords"] != ""]
        .groupby("candidate_id")["analysis_term_no_stopwords"]
        .nunique()
        .rename("unique_terms_no_stopwords")
    )

    candidate_stats = (
        candidate_stats.merge(answer_counts, on="candidate_id", how="left")
        .merge(char_counts, on="candidate_id", how="left")
        .merge(total_tokens, on="candidate_id", how="left")
        .merge(analysis_tokens, on="candidate_id", how="left")
        .merge(analysis_tokens_no_stopwords, on="candidate_id", how="left")
        .merge(noun_tokens_no_stopwords, on="candidate_id", how="left")
        .merge(unique_terms, on="candidate_id", how="left")
        .merge(unique_terms_no_stopwords, on="candidate_id", how="left")
        .fillna(0)
    )

    for column in [
        "answer_count",
        "char_count",
        "total_token_count",
        "analysis_token_count",
        "analysis_token_count_no_stopwords",
        "noun_token_count_no_stopwords",
        "unique_terms",
        "unique_terms_no_stopwords",
    ]:
        candidate_stats[column] = candidate_stats[column].astype(int)

    candidate_stats["type_token_ratio"] = (
        candidate_stats["unique_terms_no_stopwords"]
        / candidate_stats["analysis_token_count_no_stopwords"].replace(0, pd.NA)
    ).fillna(0.0)
    candidate_stats["avg_chars_per_answer"] = (
        candidate_stats["char_count"] / candidate_stats["answer_count"].replace(0, pd.NA)
    ).fillna(0.0)
    candidate_stats["avg_tokens_per_answer"] = (
        candidate_stats["analysis_token_count_no_stopwords"]
        / candidate_stats["answer_count"].replace(0, pd.NA)
    ).fillna(0.0)

    question_token_counts = (
        tokens.groupby(["candidate_id", "question_id"]).size().rename("total_token_count").reset_index()
    )
    question_analysis_counts = (
        tokens[tokens["analysis_term_no_stopwords"] != ""]
        .groupby(["candidate_id", "question_id"])
        .size()
        .rename("analysis_token_count_no_stopwords")
        .reset_index()
    )
    question_unique_terms = (
        tokens[tokens["analysis_term_no_stopwords"] != ""]
        .groupby(["candidate_id", "question_id"])["analysis_term_no_stopwords"]
        .nunique()
        .rename("unique_terms_no_stopwords")
        .reset_index()
    )

    question_stats = (
        answers[["candidate_id", "candidate_name", "question_id", "question_order", "question_title", "char_count"]]
        .merge(question_token_counts, on=["candidate_id", "question_id"], how="left")
        .merge(question_analysis_counts, on=["candidate_id", "question_id"], how="left")
        .merge(question_unique_terms, on=["candidate_id", "question_id"], how="left")
        .fillna(0)
        .sort_values(by=["question_order", "candidate_id"], kind="stable")
    )
    for column in ["total_token_count", "analysis_token_count_no_stopwords", "unique_terms_no_stopwords"]:
        question_stats[column] = question_stats[column].astype(int)

    word_frequency_by_candidate = (
        tokens[tokens["analysis_term_no_stopwords"] != ""]
        .groupby(["candidate_id", "candidate_name", "analysis_term_no_stopwords"])
        .size()
        .rename("frequency")
        .reset_index()
        .rename(columns={"analysis_term_no_stopwords": "term"})
    )
    term_prevalence = (
        word_frequency_by_candidate.groupby("term")["candidate_id"]
        .nunique()
        .rename("candidate_count")
        .reset_index()
    )
    total_term_frequency = (
        word_frequency_by_candidate.groupby("term")["frequency"].sum().rename("total_frequency").reset_index()
    )
    word_frequency_by_candidate = (
        word_frequency_by_candidate.merge(term_prevalence, on="term", how="left")
        .merge(total_term_frequency, on="term", how="left")
        .sort_values(by=["candidate_id", "frequency", "term"], ascending=[True, False, True], kind="stable")
    )
    word_frequency_by_candidate["candidate_prevalence"] = (
        word_frequency_by_candidate["candidate_count"] / len(candidate_order)
    )

    output_dir = OUTPUT_DIR / "descriptive"
    save_dataframe(candidate_stats, output_dir / "candidate_stats.csv")
    save_dataframe(question_stats, output_dir / "question_stats.csv")
    save_dataframe(word_frequency_by_candidate, output_dir / "word_frequency_by_candidate.csv")

    plt = configure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered_stats = candidate_stats.set_index("candidate_id").loc[candidate_order].reset_index()
    ax.bar(ordered_stats["candidate_name"], ordered_stats["char_count"], color="#3A6EA5")
    ax.set_title("候補者別 総文字数")
    ax.set_ylabel("文字数")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_total_characters.png")
    plt.close(fig)

    heatmap_source = question_stats.copy()
    heatmap_source["candidate_name"] = heatmap_source["candidate_id"].map(candidate_name_map)
    pivot = (
        heatmap_source.pivot(index="question_id", columns="candidate_name", values="char_count")
        .fillna(0)
        .astype(int)
    )
    question_titles = (
        heatmap_source.drop_duplicates(subset=["question_id"])
        .sort_values("question_order")
        .set_index("question_id")["question_title"]
        .to_dict()
    )
    pivot = pivot.reindex(index=sorted(pivot.index), columns=[candidate_name_map[cid] for cid in candidate_order])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{qid}: {question_titles[qid]}" for qid in pivot.index])
    ax.set_title("質問×候補者 回答文字数")
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            ax.text(column_index, row_index, pivot.iloc[row_index, column_index], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="文字数")
    fig.tight_layout()
    fig.savefig(output_dir / "question_answer_length_heatmap.png")
    plt.close(fig)

    print(candidate_stats[["candidate_name", "char_count", "analysis_token_count_no_stopwords"]].to_string(index=False))


if __name__ == "__main__":
    main()
