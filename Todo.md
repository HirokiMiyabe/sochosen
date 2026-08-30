# 総長候補者テキスト分析 TODO

## 0. 目的

東京大学新聞社「総長選考2026 候補者紹介」の第2次候補者5名について、各候補者の所見テキストを取得・整形し、以下の①〜⑤を再現可能な形で分析する。

1. 基礎記述：誰が何をどれだけ語ったか
2. 共通語：5候補に共有される語彙
3. 共通課題：5候補に共有される問題認識
4. 特徴語：各候補固有の語彙（TF-IDF）
5. 相対的強調：各候補が他4候補より強く用いる語（weighted log-odds ratio with informative Dirichlet prior）

対象ページ：
- https://news.todaishimbun.org/candidate

対象者（第2次候補者のみ）：
- 大越慎一
- 菅野暁
- 染谷隆夫
- 藤垣裕子
- 山本隆司

---

## 1. 作業場所・基本方針

現在の作業ディレクトリ：

```powershell
PS C:\Users\hiroj\MyProject\sochosen> C:\Users\hiroj\MyProject\sochosen\.venv\Scripts\Activate.ps1
(.venv) PS C:\Users\hiroj\MyProject\sochosen>
```

この `C:\Users\hiroj\MyProject\sochosen` をプロジェクトルートとする。

### 必須方針

- 既存の `.venv` を使う。新しい仮想環境は作らない。
- MeCab はすでに利用可能なので、まず既存環境から呼び出せるか確認する。
- Pythonコードはすべて `code/` に置く。
- 取得・整形したデータはすべて `data/` に置く。
- 分析結果は `output/` に置く。
- 元データを上書きしない。raw → processed → analysis の流れを保つ。
- 乱数を使う処理があれば seed を固定する。
- すべての処理をスクリプトとして再実行可能にする。Notebookだけに処理を閉じ込めない。
- 文字コードは原則 UTF-8。CSVを書き出す場合は Excel でも開きやすい `utf-8-sig` を使う。
- スクレイピング時には過剰アクセスを避け、候補者ページは各1回程度の取得で済むようにする。
- HTMLを取得した場合、可能なら raw HTML も保存し、後から抽出処理を検証できるようにする。

---

## 2. プロジェクト構成を作る

以下の構成を作成する。

```text
sochosen/
├─ .git/
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

PowerShell例：

```powershell
New-Item -ItemType Directory -Force code, data, output
New-Item -ItemType Directory -Force data\raw\html, data\raw\txt, data\processed\answers, data\metadata
New-Item -ItemType Directory -Force output\descriptive, output\common_terms, output\common_issues, output\tfidf, output\log_odds
```

---

## 3. Git 管理を開始する

### TODO

- [ ] `git status` を確認する。
- [ ] まだGit管理されていなければ `git init` を実行する。
- [ ] `.gitignore` を作る。
- [ ] 初期構成をコミットする。

`.gitignore` には最低限以下を入れる。

```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.vscode/
.DS_Store
Thumbs.db
```

今回は分析の再現性を優先し、`data/` は原則Git管理対象にする。ただし、raw HTMLが大きくなりすぎた場合は後で方針を見直す。

初期コミット例：

```powershell
git add .
git commit -m "Initialize candidate text analysis project"
```

---

## 4. requirements.txt を作る

まず現在の `.venv` で必要ライブラリが入っているか確認し、不足分だけインストールする。

最低限の候補：

```txt
beautifulsoup4
requests
pandas
numpy
scikit-learn
matplotlib
mecab-python3
unidic-lite
scipy
```

ただし、MeCabがシステム側で既に動いており `mecab-python3` / `unidic-lite` が不要または競合する場合は、無理に入れない。最初に以下を確認すること。

```powershell
python -c "import MeCab; print(MeCab.Tagger().parse('東京大学の研究と教育'))"
```

動く構成を優先し、その結果に合わせて `requirements.txt` を確定する。

必要なら：

```powershell
pip install -r requirements.txt
```

---

# DATA COLLECTION

## 5. 候補者一覧・URLを取得する

`code/01_scrape_candidates.py` を作成する。

入口：

- https://news.todaishimbun.org/candidate

第2次候補者5名だけを対象にする。

候補者一覧ページから各候補者詳細ページへのリンクを抽出し、`data/metadata/candidates.csv` に保存する。

想定形式：

```csv
candidate_id,candidate_name,url
ohkoshi,大越慎一,https://...
kanno,菅野暁,https://...
someya,染谷隆夫,https://...
fujigaki,藤垣裕子,https://...
yamamoto,山本隆司,https://...
```

### 注意

- 「第1次候補者」は分析対象に含めない。
- URLや候補者名をコード中に完全ハードコードするより、一覧ページから取得したうえで、第2次候補者部分を明示的に選択する実装を優先する。
- ただしサイト構造が不安定な場合は、5人の期待値を検証用リストとして持ってよい。

---

## 6. 候補者ごとの全文 `.txt` をまず作る

各候補者詳細ページについて、**所見・所信表明の回答本文**を取得する。

まずは質問単位に分割せず、候補者1人につき1ファイルとして保存する。

保存先：

```text
data/raw/txt/ohkoshi.txt
data/raw/txt/kanno.txt
data/raw/txt/someya.txt
data/raw/txt/fujigaki.txt
data/raw/txt/yamamoto.txt
```

必要ならHTMLも：

```text
data/raw/html/ohkoshi.html
...
```

### txt に含めるもの

- 候補者本人の回答本文
- 質問見出しは、後の分割に使うため残してよい

### txt から除外するもの

- サイト共通ヘッダー・フッター
- ナビゲーション
- 候補者プロフィール・経歴（今回の「所見」分析に不要なら除外）
- 東京大学新聞社側の説明文
- SNSリンク等

**重要：質問文そのものは後の語彙分析対象に含めない。**

ただし質問への分割に使うため、raw段階では質問見出しを保持してよい。

---

## 7. 質問への回答ごとに分割する

`code/02_split_questions.py` を作る。

候補者ごとの全文txtまたはHTMLから、質問単位で回答を切り出す。

### データモデル

分析の基本単位は long format の

```text
candidate × question
```

とする。

`data/processed/candidate_answers.csv` を作成する。

例：

```csv
candidate_id,candidate_name,question_id,question_title,text
fujigaki,藤垣裕子,q01,大学の役割,"..."
fujigaki,藤垣裕子,q02,教育,"..."
...
```

また、目視確認しやすいように個別txtも保存する。

```text
data/processed/answers/fujigaki_q01.txt
data/processed/answers/fujigaki_q02.txt
...
```

### 重要な検証

- [ ] 5候補すべてについて質問数を確認する。
- [ ] 質問タイトル・順序が候補者間で共通か確認する。
- [ ] 欠損回答があるか確認する。
- [ ] 同じ質問に対する回答が同じ `question_id` になるようにする。
- [ ] 質問本文が回答 `text` に混入していないことを目視確認する。
- [ ] 最初と最後の200文字程度をログ表示し、分割ミスを確認する。

質問タイトルはサイト上の表記を保存するが、分析用に `q01`, `q02`, ... の安定したIDを別に付与する。

---

# PREPROCESSING

## 8. MeCab で形態素解析する

`code/03_tokenize_mecab.py` を作る。

入力：

```text
data/processed/candidate_answers.csv
```

出力：

```text
data/processed/tokens.csv
```

### 最初に保持する品詞

主分析ではまず：

- 名詞
- 動詞
- 形容詞

を候補とする。

ただし「特徴語」の解釈がしやすいよう、**名詞のみの分析も別途出力できる実装**にする。

### 正規化

- 表層形ではなく、可能なら辞書形・原形を使用する。
- 全角半角等の軽いUnicode正規化を行う。
- 記号、空白、URL、数字だけのtokenを除外する。
- 1文字語は機械的に全削除しない。意味のある漢字1文字語が存在するため。

### ストップワード

最初から恣意的に大量削除しない。

まず raw に近い結果を出し、その後、分析上自明な語を `code/utils.py` 内などで明示的に管理する。

候補例：

- 大学
- 東京大学
- 本学
- 東大
- 研究
- 教育

ただしこれらは**自動的に除外せず、除外前・除外後を比較できるようにする。**

質問見出し由来の語は回答本文に含めない。

### 複合語

「国際卓越研究大学」「若手研究者」「産学連携」「研究時間」「学術研究」などがMeCabで分割されすぎる可能性がある。

初回分析では通常の形態素を使う。

その後、結果を確認して必要なら：

- 名詞連接の抽出
- bi-gram / tri-gram

を追加する。

最初から複合語辞書を恣意的に作り込みすぎない。

---

# ANALYSIS ①〜⑤

## 9. ① 基礎記述

`code/04_descriptive_stats.py`

候補者ごと、質問ごとに以下を算出する。

### 候補者単位

- 文字数
- 総token数
- 分析対象token数
- 異なり語数
- Type-Token Ratio（参考値として。文書長依存であることを明記）
- 1回答あたり平均文字数
- 1回答あたり平均token数

### 質問単位

- 各候補者の文字数
- token数
- 異なり語数

### 出力

```text
output/descriptive/candidate_stats.csv
output/descriptive/question_stats.csv
output/descriptive/word_frequency_by_candidate.csv
```

可視化：

- 候補者別総文字数 bar plot
- 質問×候補者の回答長 heatmap または grouped bar plot

図はPNGで保存する。

---

## 10. ② 共通語：candidate prevalence

`code/05_common_terms.py`

ここでは単純総頻度だけでなく、**何人の候補者がその語を使ったか**を重視する。

単語 `w` について：

```text
candidate_prevalence(w) = その語を1回以上使用した候補者数 / 5
```

以下を出力する。

```text
term
candidate_count
candidate_prevalence
total_frequency
ohkoshi_frequency
kanno_frequency
someya_frequency
fujigaki_frequency
yamamoto_frequency
```

### 主に見るもの

- 5/5候補が使用した語
- 4/5候補が使用した語

ただし以下のような質問構造上ほぼ必然的な語は「共通課題」と即断しない。

- 大学
- 研究
- 教育
- 東京大学

### 出力

```text
output/common_terms/common_terms.csv
output/common_terms/common_terms_5of5.csv
output/common_terms/common_terms_4of5_or_more.csv
```

候補者間で頻度差が大きい語も確認できるようにする。

---

## 11. ③ 共通課題：共通語の文脈分析

`code/06_common_issues.py`

**共通語 = 共通課題ではない**ので、candidate prevalence の高い語について文脈を見る。

### Step A：KWICを出す

5/5または4/5で使われた重要語について、各出現の前後文脈を保存する。

日本語なので「前後30 token程度」または「その語を含む文」を基本単位にする。

出力例：

```csv
term,candidate_name,question_id,left_context,keyword,right_context,sentence
財務,大越慎一,q04,...,財務,...,...
```

保存：

```text
output/common_issues/kwic.csv
```

### Step B：共起語を見る

共通語 `w` の周囲 ±5 token または同一文内で共起する語を集計する。

候補者別・全体の双方を出す。

例：

```text
財務 → 基盤 / 多様化 / 自立 / 基金 / 運営 ...
若手 → 研究者 / 雇用 / 支援 / キャリア ...
```

可能なら Dice係数またはPMIも計算する。ただし小規模コーパスなので、**共起回数そのものも必ず併記**する。

### Step C：「課題候補」を抽出する

自動的に「これは課題である」と断定しない。

以下の条件を満たす語・フレーズを、人間が原文確認するための候補としてランキングする。

- candidate prevalence が高い
- 総頻度が一定以上
- 「必要」「課題」「問題」「不足」「困難」「低下」「強化」「改善」「確保」「改革」等の問題・対応表現と近接している

これは補助的ヒューリスティックと明記する。

最終的な「共通課題」のラベル付けはKWICを読んで人間が判断する。

### 目標となる最終表

```text
共通課題 | 言及候補数 | 関連語 | 各候補の代表文脈 | 解釈メモ
```

自動生成CSVに加え、人間が後から記入できるテンプレートも作る。

```text
output/common_issues/common_issue_review_template.csv
```

---

## 12. ④ TF-IDF：候補者ごとの特徴語

`code/07_tfidf.py`

### 文書単位

主分析では、各候補者の全質問回答を結合し、

```text
N = 5 documents
```

としてTF-IDFを計算する。

目的：

> その候補者では頻出し、他候補者ではあまり使われない語を探索する。

### 重要

N=5しかないため、IDFの段階が粗く、1人だけが使ったレア語が上位に出やすい。

**TF-IDFだけを「候補者の特徴」の確定的指標として扱わない。探索用と位置づける。**

### 出力

候補者ごとTop20またはTop30：

```text
output/tfidf/tfidf_all.csv
output/tfidf/tfidf_top20_by_candidate.csv
```

列：

```text
candidate_name
term
tf
idf
tfidf
raw_frequency
candidate_count
```

### 追加分析：質問別

`candidate × question` を文書としてTF-IDFを算出する版も出す。

特に同一質問内で候補者を比較できるようにする。

例：

```text
「研究」に対する回答だけで5候補を比較
```

出力：

```text
output/tfidf/tfidf_by_question.csv
```

---

## 13. ⑤ Weighted log-odds ratio with informative Dirichlet prior

`code/08_weighted_log_odds.py`

各候補者について one-vs-rest で比較する。

例：

```text
藤垣裕子 vs 他4候補
大越慎一 vs 他4候補
...
```

### 目的

TF-IDFより直接的に、

> 「他候補と比べて、この候補が相対的に強く使用している語」

を抽出する。

### 実装方針

Monroe, Colaresi & Quinn (2008) の informative Dirichlet prior を用いた weighted log-odds の一般的実装に従う。

各語について：

- target candidate frequency
- rest frequency
- prior count
- log odds difference
- variance
- z-score

を算出する。

informative prior は**5候補全体のコーパス語頻度**から作る。

実装式についてコード内にコメントを書く。

### 重要

- 0頻度を単純な無限大にしない。
- prior の強さ（alpha_0）の設定をコード上で明示する。
- prior strength を変えた感度分析ができる関数設計にする。
- 主ランキングは `z-score` の降順とする。

### 出力

```text
output/log_odds/log_odds_all.csv
output/log_odds/log_odds_top20_by_candidate.csv
```

列：

```text
candidate_name
term
target_count
rest_count
prior_count
log_odds
variance
z_score
candidate_count
```

候補者ごと：

- 正のz-score Top20 = その候補に特徴的
- 負のz-score Bottom20 = 他候補側に特徴的

を保存する。

---

# INTEGRATION

## 14. TF-IDF と weighted log-odds を統合する

可能なら `code/08_weighted_log_odds.py` または別処理で以下の比較表も作る。

```text
candidate_name
term
tfidf_rank
log_odds_rank
tfidf
z_score
raw_frequency
candidate_count
```

保存：

```text
output/log_odds/tfidf_logodds_comparison.csv
```

特に、

- TF-IDFでも高い
- weighted log-oddsでも高い

語を「頑健な特徴語候補」として確認する。

ただし機械的に閾値を決めず、原文KWICに戻って解釈する。

---

## 15. 最終的に回答したい分析問い

### RQ1：共通認識

> 5人の総長候補が共通して重視・問題視している東京大学の課題は何か？

使うもの：

- candidate prevalence
- 総頻度
- KWIC
- 共起
- 原文確認

### RQ2：候補者間の差

> 共通する問題状況の中で、各候補者は何を相対的に強調しているか？

使うもの：

- TF-IDF
- weighted log-odds
- 質問別比較
- KWIC

最終的には、例えば以下の形式で解釈できることを目指す。

```text
5候補とも X を課題として認識している。
ただし、大越候補は A の語彙で、菅野候補は B、染谷候補は C、藤垣候補は D、山本候補は E の語彙でその課題を位置づけている。
```

---

# QUALITY CHECK

## 16. 必須チェック

- [ ] 第2次候補者5名だけが対象になっている。
- [ ] 候補者プロフィール・経歴が分析本文に混入していない。
- [ ] 質問文そのものが分析tokenに混入していない。
- [ ] 5人のraw txtを目視確認した。
- [ ] 質問単位への分割を各候補最低2箇所目視確認した。
- [ ] `candidate_answers.csv` の候補者数 = 5。
- [ ] 質問数・欠損を報告した。
- [ ] MeCabの辞書形が妥当に取得できている。
- [ ] stopword除外前後の結果を比較できる。
- [ ] TF-IDFではN=5の制約を明記した。
- [ ] log-oddsの prior と式をコードコメントに残した。
- [ ] 頻度1の語だけでランキングが占有されていないか確認した。
- [ ] 特徴語について原文KWICへ戻れる。

---

## 17. README.md に最終的に書くこと

READMEには最低限以下を記載する。

1. 分析目的
2. データ出典とURL
3. 対象候補者
4. ディレクトリ構造
5. セットアップ方法
6. 実行順序
7. 前処理方法
8. 分析①〜⑤の説明
9. TF-IDFの限界
10. weighted log-odds の説明・参考文献
11. 出力ファイル一覧
12. 再現方法

実行順序は例えば：

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

---

# DEFINITION OF DONE

以下がすべて揃ったら今回のタスク完了。

- [ ] Git管理されたプロジェクトになっている。
- [ ] `code/`, `data/`, `output/` が存在する。
- [ ] `requirements.txt` が存在し、環境を再現できる。
- [ ] 5候補者それぞれのraw `.txt` が存在する。
- [ ] 質問単位に分割されたデータが存在する。
- [ ] 形態素解析済みデータが存在する。
- [ ] ①基礎記述のCSV・図が出る。
- [ ] ②共通語ランキングが出る。
- [ ] ③共通課題検討用KWIC・共起・レビュー表が出る。
- [ ] ④候補者別TF-IDFランキングが出る。
- [ ] ⑤候補者別weighted log-oddsランキングが出る。
- [ ] TF-IDFとlog-oddsを横断して比較できる。
- [ ] すべての出力から元の候補者・質問・原文へ追跡できる。
- [ ] READMEの手順だけで最初から再実行できる。

---

## Codexへの実装上の指示

このTODOは上から順に実装すること。

- 各ステップ終了時に実際にスクリプトを実行し、エラーがないことを確認する。
- サイトのHTML構造は実物を確認してからパーサを書く。CSS selectorを推測だけで決めない。
- 抽出結果を必ず目視検証する。
- 不明点があっても処理を止めず、合理的なデフォルトを採用し、その判断をREADMEまたはコードコメントに残す。
- 分析結果を都合よく見せるためのstopword追加や閾値調整をしない。
- preprocessing・パラメータ・除外語はすべて明示・保存する。
- まず最小限の再現可能な実装を完成させ、その後必要なら改善する。
