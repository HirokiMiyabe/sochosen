from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DIR, ensure_project_dirs, get_mecab_tagger, load_dataframe, save_dataframe, tokenize_answer


def main() -> None:
    ensure_project_dirs()
    answers_path = PROCESSED_DIR / "candidate_answers.csv"
    answers = load_dataframe(answers_path)
    if answers.empty:
        raise RuntimeError("candidate_answers.csv が空です。先に 02_split_questions.py を実行してください。")

    tagger = get_mecab_tagger()
    token_rows: list[dict[str, object]] = []
    for answer in answers.to_dict("records"):
        answer_id = f"{answer['candidate_id']}_{answer['question_id']}"
        rows = tokenize_answer(answer["text"], tagger)
        for row in rows:
            token_rows.append(
                {
                    "answer_id": answer_id,
                    "candidate_id": answer["candidate_id"],
                    "candidate_name": answer["candidate_name"],
                    "question_id": answer["question_id"],
                    "question_order": answer["question_order"],
                    "question_title": answer["question_title"],
                    **row,
                }
            )
        print(
            f"{answer['candidate_name']} {answer['question_id']}: "
            f"{len(rows)} token / {sum(1 for row in rows if row['analysis_term'])} analysis token"
        )

    tokens_df = pd.DataFrame(token_rows)
    save_dataframe(tokens_df, PROCESSED_DIR / "tokens.csv")

    sample = tokens_df[
        ["candidate_name", "question_id", "surface", "lemma", "pos", "analysis_term_no_stopwords"]
    ].head(20)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
