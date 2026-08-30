from __future__ import annotations

import pandas as pd

from utils import METADATA_DIR, OUTPUT_DIR, ensure_project_dirs, load_dataframe, save_dataframe


def build_prevalence_table(
    tokens: pd.DataFrame,
    term_column: str,
    candidate_order: list[str],
) -> pd.DataFrame:
    filtered = tokens[tokens[term_column] != ""].copy()
    filtered = filtered.rename(columns={term_column: "term"})
    candidate_term_counts = (
        filtered.groupby(["term", "candidate_id"]).size().unstack(fill_value=0).reindex(columns=candidate_order, fill_value=0)
    )
    total_frequency = candidate_term_counts.sum(axis=1)
    candidate_count = (candidate_term_counts > 0).sum(axis=1)

    table = pd.DataFrame(
        {
            "term": candidate_term_counts.index,
            "candidate_count": candidate_count.values,
            "candidate_prevalence": candidate_count.values / len(candidate_order),
            "total_frequency": total_frequency.values,
        }
    )
    for candidate_id in candidate_order:
        table[f"{candidate_id}_frequency"] = candidate_term_counts[candidate_id].values
    table = table.sort_values(
        by=["candidate_count", "total_frequency", "term"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return table


def main() -> None:
    ensure_project_dirs()
    metadata = load_dataframe(METADATA_DIR / "candidates.csv")
    tokens = load_dataframe("data/processed/tokens.csv")
    if tokens.empty:
        raise RuntimeError("tokens.csv が空です。先に 03_tokenize_mecab.py を実行してください。")

    candidate_order = metadata["candidate_id"].tolist()
    output_dir = OUTPUT_DIR / "common_terms"

    default_table = build_prevalence_table(tokens, "analysis_term_no_stopwords", candidate_order)
    with_stopwords_table = build_prevalence_table(tokens, "analysis_term", candidate_order)

    save_dataframe(default_table, output_dir / "common_terms.csv")
    save_dataframe(with_stopwords_table, output_dir / "common_terms_with_stopwords.csv")
    save_dataframe(default_table[default_table["candidate_count"] == len(candidate_order)], output_dir / "common_terms_5of5.csv")
    save_dataframe(default_table[default_table["candidate_count"] >= len(candidate_order) - 1], output_dir / "common_terms_4of5_or_more.csv")

    print(default_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
