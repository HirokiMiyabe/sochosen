# 総長候補者テキスト分析

東京大学新聞社の公開ページ「総長選考2026 候補者紹介」から、第2次候補者5名の所見テキストを取得し、再実行可能なスクリプトとして整形・形態素解析・記述統計・特徴語分析を行うプロジェクトです。

## 分析目的

以下の5点を再現可能な形で出力します。

1. 誰がどれだけ語ったかの基礎記述
2. 5候補に共有される語彙
3. 共通語の文脈から見た共通課題候補
4. 候補者ごとの特徴語（TF-IDF）
5. 候補者が他4候補より相対的に強く使う語（weighted log-odds ratio）

## データ出典

- 候補者一覧: https://news.todaishimbun.org/candidate
- 候補者詳細: 上記一覧から抽出した各候補者ページ

対象は第2次候補者5名のみです。

- 大越慎一
- 菅野暁
- 染谷隆夫
- 藤垣裕子
- 山本隆司

## ディレクトリ構造

```text
sochosen/
├─ .gitignore
├─ README.md
├─ Todo.md
├─ requirements.txt
├─ code/
│  ├─ 01_scrape_candidates.py
│  ├─ 02_split_questions.py
│  ├─ 03_tokenize_mecab.py
│  ├─ 04_descriptive_stats.py
│  ├─ 05_common_terms.py
│  ├─ 06_common_issues.py
│  ├─ 07_tfidf.py
│  ├─ 08_weighted_log_odds.py
│  └─ utils.py
├─ data/
│  ├─ raw/
│  │  ├─ html/
│  │  └─ txt/
│  ├─ processed/
│  │  ├─ answers/
│  │  ├─ candidate_answers.csv
│  │  ├─ question_validation.csv
│  │  └─ tokens.csv
│  └─ metadata/
│     └─ candidates.csv
└─ output/
   ├─ descriptive/
   ├─ common_terms/
   ├─ common_issues/
   ├─ tfidf/
   └─ log_odds/
```

## セットアップ

既存の `.venv` を使います。

```powershell
C:\Users\hiroj\MyProject\sochosen\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`MeCab` は `mecab-python3 + unidic-lite` を利用します。辞書パスは Windows 環境向けに `Path(...).as_posix()` を通して指定しています。

## 実行順序

```powershell
python code\01_scrape_candidates.py
python code\02_split_questions.py
python code\03_tokenize_mecab.py
python code\04_descriptive_stats.py
python code\05_common_terms.py
python code\06_common_issues.py
python code\07_tfidf.py
python code\08_weighted_log_odds.py
```

## 前処理方法

- raw HTML と raw txt を両方保存します。
- `02_split_questions.py` は `raw HTML` を優先して設問単位に分割します。
- 明示的な質問見出しがある候補は見出し行を使って分割します。
- 山本隆司ページは他候補と HTML 構造が異なり、質問見出しが明示されないため、空段落区切りで 7 セクションに分け、他候補の設問順に対応付けています。
- 形態素解析は MeCab で行い、主分析は名詞・動詞・形容詞を対象にします。
- `tokens.csv` には stopword 除外前後の列を両方保存します。
- stopword は `大学 / 東京大学 / 本学 / 東大 / 研究 / 教育` を初期値として明示管理し、自動的に消した語と元の語列を比較できるようにしています。
- CSV は `utf-8-sig` で保存します。

## 分析 1: 基礎記述

`code/04_descriptive_stats.py` は以下を出力します。

- `output/descriptive/candidate_stats.csv`
- `output/descriptive/question_stats.csv`
- `output/descriptive/word_frequency_by_candidate.csv`
- `output/descriptive/candidate_total_characters.png`
- `output/descriptive/question_answer_length_heatmap.png`

## 分析 2: 共通語

`code/05_common_terms.py` は candidate prevalence を計算し、以下を出力します。

- `output/common_terms/common_terms.csv`
- `output/common_terms/common_terms_with_stopwords.csv`
- `output/common_terms/common_terms_5of5.csv`
- `output/common_terms/common_terms_4of5_or_more.csv`

`common_terms_with_stopwords.csv` により、stopword 除外前後の比較ができます。

## 分析 3: 共通課題候補

`code/06_common_issues.py` は高 prevalence 語について KWIC と共起を作り、人手レビュー用の表も出力します。

- `output/common_issues/kwic.csv`
- `output/common_issues/cooccurrence_sentence.csv`
- `output/common_issues/issue_candidates.csv`
- `output/common_issues/common_issue_review_template.csv`

`common_issue_review_template.csv` は次の列を持ちます。

- `term`
- `言及候補数`
- `総頻度`
- `関連語`
- `各候補の代表文脈`
- `共通課題`
- `解釈メモ`

## 分析 4: TF-IDF

`code/07_tfidf.py` は候補者全体を `N = 5 documents` として集約した版と、設問別の版を出力します。

- `output/tfidf/tfidf_all.csv`
- `output/tfidf/tfidf_top20_by_candidate.csv`
- `output/tfidf/tfidf_by_question.csv`

### TF-IDF の限界

- 候補者単位の文書数は 5 しかないため、IDF の粒度が粗いです。
- そのため、TF-IDF は確定的な特徴量ではなく探索用の指標として扱います。
- 原文確認や weighted log-odds との突き合わせが前提です。

## 分析 5: weighted log-odds

`code/08_weighted_log_odds.py` は one-vs-rest 比較を行い、informative Dirichlet prior を使った weighted log-odds を計算します。

- `output/log_odds/log_odds_all.csv`
- `output/log_odds/log_odds_top20_by_candidate.csv`
- `output/log_odds/tfidf_logodds_comparison.csv`

prior は 5 候補全体を結合した語頻度分布から構成し、既定値の `prior_strength` は `100.0` です。式の説明は `code/08_weighted_log_odds.py` 内のコメントに残しています。

## 参考文献

- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict.

## 出力ファイル一覧

- `data/metadata/candidates.csv`: 第2次候補者一覧
- `data/raw/html/*.html`: 候補者ページ raw HTML
- `data/raw/txt/*.txt`: 候補者ごとの raw txt
- `data/processed/candidate_answers.csv`: `candidate × question` の long 形式データ
- `data/processed/answers/*.txt`: 設問ごとの個別 txt
- `data/processed/question_validation.csv`: 設問数と preview の検証用表
- `data/processed/tokens.csv`: 形態素解析済み token 列
- `output/descriptive/*`: 基礎記述
- `output/common_terms/*`: 共通語
- `output/common_issues/*`: KWIC, 共起, レビュー補助表
- `output/tfidf/*`: TF-IDF
- `output/log_odds/*`: weighted log-odds と比較表

## 再現方法

- `requirements.txt` をインストールする
- 上記 01〜08 の順でスクリプトを実行する
- 出力 CSV と PNG を `output/` 以下で確認する
- 共通課題の解釈は `kwic.csv` と `common_issue_review_template.csv` を使って人手で行う

## 補足

- 候補者 ID は一覧ページから取得した URL slug を使っています。そのため `菅野暁` の ID はサンプル表記の `kanno` ではなく実 URL に合わせた `sugano` です。
- スクレイパと分割処理は実装時点の公開 HTML 構造を前提としています。サイト構造が変わった場合は `code/01_scrape_candidates.py` と `code/02_split_questions.py` の見直しが必要です。
