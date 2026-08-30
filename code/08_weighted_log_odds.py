from __future__ import annotations

import argparse
import math
from collections import Counter

import pandas as pd

from utils import METADATA_DIR, OUTPUT_DIR, ensure_project_dirs, load_dataframe, save_dataframe


def build_candidate_counters(tokens: pd.DataFrame, candidate_order: list[str], term_column: str) -> dict[str, Counter[str]]:
    filtered = tokens[tokens[term_column] != ""].copy()
    counters: dict[str, Counter[str]] = {}
    for candidate_id in candidate_order:
        counters[candidate_id] = Counter(filtered.loc[filtered["candidate_id"] == candidate_id, term_column].tolist())
    return counters


def compute_weighted_log_odds(
    candidate_counters: dict[str, Counter[str]],
    candidate_name_map: dict[str, str],
    prior_strength: float,
    term_column: str,
) -> pd.DataFrame:
    overall_counter: Counter[str] = Counter()
    for counter in candidate_counters.values():
        overall_counter.update(counter)

    total_tokens = sum(overall_counter.values())
    if total_tokens == 0:
        raise RuntimeError("weighted log-odds を計算するための token がありません。")

    vocabulary = sorted(overall_counter)
    candidate_count_by_term = {
        term: sum(1 for counter in candidate_counters.values() if counter.get(term, 0) > 0) for term in vocabulary
    }

    # Monroe, Colaresi & Quinn (2008) の informative Dirichlet prior:
    # alpha_w = alpha_0 * p(w), p(w) は全候補結合コーパスでの語頻度比率。
    # delta_w = log((y_i + alpha_w) / (n_i + alpha_0 - y_i - alpha_w))
    #         - log((y_j + alpha_w) / (n_j + alpha_0 - y_j - alpha_w))
    # var_w   = 1 / (y_i + alpha_w) + 1 / (y_j + alpha_w)
    # z_w     = delta_w / sqrt(var_w)
    prior_counts = {term: prior_strength * (overall_counter[term] / total_tokens) for term in vocabulary}
    alpha_0 = sum(prior_counts.values())

    rows: list[dict[str, object]] = []
    for candidate_id, target_counter in candidate_counters.items():
        rest_counter = overall_counter - target_counter
        target_total = sum(target_counter.values())
        rest_total = sum(rest_counter.values())
        for term in vocabulary:
            prior_count = prior_counts[term]
            target_count = target_counter.get(term, 0)
            rest_count = rest_counter.get(term, 0)
            numerator_target = target_count + prior_count
            numerator_rest = rest_count + prior_count
            denominator_target = target_total + alpha_0 - numerator_target
            denominator_rest = rest_total + alpha_0 - numerator_rest
            log_odds = math.log(numerator_target / denominator_target) - math.log(numerator_rest / denominator_rest)
            variance = (1 / numerator_target) + (1 / numerator_rest)
            z_score = log_odds / math.sqrt(variance)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name_map[candidate_id],
                    "term": term,
                    "target_count": target_count,
                    "rest_count": rest_count,
                    "prior_count": prior_count,
                    "log_odds": log_odds,
                    "variance": variance,
                    "z_score": z_score,
                    "candidate_count": candidate_count_by_term[term],
                    "term_column": term_column,
                    "prior_strength": prior_strength,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--term-column",
        default="analysis_term_no_stopwords",
        choices=["analysis_term", "analysis_term_no_stopwords", "noun_term", "noun_term_no_stopwords"],
    )
    parser.add_argument("--prior-strength", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    ensure_project_dirs()
    metadata = load_dataframe(METADATA_DIR / "candidates.csv")
    tokens = load_dataframe("data/processed/tokens.csv")
    tfidf_all = load_dataframe(OUTPUT_DIR / "tfidf" / "tfidf_all.csv")
    if tokens.empty or tfidf_all.empty:
        raise RuntimeError("入力データが不足しています。先に 03 と 07 のスクリプトを実行してください。")

    candidate_order = metadata["candidate_id"].tolist()
    candidate_name_map = metadata.set_index("candidate_id")["candidate_name"].to_dict()

    candidate_counters = build_candidate_counters(tokens, candidate_order, args.term_column)
    log_odds_df = compute_weighted_log_odds(
        candidate_counters,
        candidate_name_map,
        prior_strength=args.prior_strength,
        term_column=args.term_column,
    ).sort_values(
        by=["candidate_id", "z_score", "target_count", "term"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    log_odds_df["rank"] = log_odds_df.groupby("candidate_id").cumcount() + 1

    top_rows: list[pd.DataFrame] = []
    for candidate_id, group in log_odds_df.groupby("candidate_id", sort=False):
        positive = group.head(args.top_k).copy()
        positive["direction"] = "positive"
        positive["rank_within_direction"] = range(1, len(positive) + 1)

        negative = group.sort_values(by=["z_score", "target_count", "term"], ascending=[True, False, True], kind="stable").head(args.top_k).copy()
        negative["direction"] = "negative"
        negative["rank_within_direction"] = range(1, len(negative) + 1)

        top_rows.extend([positive, negative])
        frequency_one_count = int((positive["target_count"] == 1).sum())
        print(f"{candidate_name_map[candidate_id]}: Top{args.top_k} 中 target_count=1 は {frequency_one_count} 語")

    top_df = pd.concat(top_rows, ignore_index=True)

    tfidf_all["tfidf"] = tfidf_all["tfidf"].astype(float)
    tfidf_all["raw_frequency"] = tfidf_all["raw_frequency"].astype(int)
    tfidf_all = tfidf_all.sort_values(
        by=["candidate_id", "tfidf", "raw_frequency", "term"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    tfidf_all["tfidf_rank"] = tfidf_all.groupby("candidate_id").cumcount() + 1

    log_odds_ranks = log_odds_df.sort_values(
        by=["candidate_id", "z_score", "target_count", "term"],
        ascending=[True, False, False, True],
        kind="stable",
    ).copy()
    log_odds_ranks["log_odds_rank"] = log_odds_ranks.groupby("candidate_id").cumcount() + 1

    comparison_df = tfidf_all[
        [
            "candidate_id",
            "candidate_name",
            "term",
            "tfidf_rank",
            "tfidf",
            "raw_frequency",
            "candidate_count",
        ]
    ].merge(
        log_odds_ranks[
            [
                "candidate_id",
                "candidate_name",
                "term",
                "z_score",
                "log_odds_rank",
            ]
        ],
        on=["candidate_id", "candidate_name", "term"],
        how="inner",
    )[
        [
            "candidate_name",
            "term",
            "tfidf_rank",
            "log_odds_rank",
            "tfidf",
            "z_score",
            "raw_frequency",
            "candidate_count",
        ]
    ].sort_values(
        by=["candidate_name", "tfidf_rank", "log_odds_rank", "term"],
        kind="stable",
    )

    output_dir = OUTPUT_DIR / "log_odds"
    save_dataframe(log_odds_df, output_dir / "log_odds_all.csv")
    save_dataframe(top_df, output_dir / "log_odds_top20_by_candidate.csv")
    save_dataframe(comparison_df, output_dir / "tfidf_logodds_comparison.csv")


if __name__ == "__main__":
    main()
