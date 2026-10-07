# Fitsched

Googleカレンダーの過去の予定から「トレーニング」をしやすい時間帯を学習し、
これからの空き時間にトレーニング枠を提案するツールです。

## セットアップ

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
```

## アプリ（画面）で使う

```sh
.venv/bin/streamlit run code/app.py
```

または Finder で `Fitsched.command` をダブルクリック。ブラウザで http://localhost:8501 が開き、
期間・月ごとの目標回数・トレーニングしてよい時間帯を変えると、その場で提案が更新されます。
`http://localhost:8501/?start=2026-10-01` のように URL で提案の開始日を指定することもできます。

## コマンドラインで使う

```sh
# data/credentials.json が無ければ、data/2023_calendar_events.csv で学習（提案期間は予定なし扱い）
.venv/bin/python code/Fitshed.py

# 期間を指定
.venv/bin/python code/Fitshed.py --start 2026-10-01 --months 3

# 提案期間の予定をCSVで渡す（start,end,summary 列）
.venv/bin/python code/Fitshed.py --future-csv data/future_events.csv
```

主なオプション: `--source {auto,google,csv}` / `--history-year` / `--rest-days`（練習の間に空ける休息日数）/ `--plot`（時間帯のヒストグラムを表示）

月ごとの目標回数・1日の上限・避けたい時間帯は `code/Fitshed.py` 冒頭の定数で変更できます。

## Googleカレンダーから取得する場合

1. Google Cloud Console で Calendar API を有効化し、OAuth クライアント（デスクトップアプリ）を作成
2. ダウンロードした JSON を `data/credentials.json` として置く
3. 実行するとブラウザで認証が開き、`data/token.json` が保存される

`data/` は `.gitignore` 済みです（個人の予定・認証情報を含むため）。

## モデルの評価

```sh
.venv/bin/python code/evaluate.py
```

旧モデルと新モデルを、実データ（2023年）と疑似データ（`code/synth.py`）で比べます。
疑似データは「朝型・授業直後・週末夕方」という決まった習慣でトレーニングを入れているため、
モデルが本当の習慣をどこまで再現できるか（正解との相関）を測れます。

カレンダーにすでに入っている「トレーニング」は目標回数に数え、その前後の日は休息日として避けます。
