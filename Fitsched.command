#!/bin/zsh
# ダブルクリックで Fitsched の画面を起動する
cd "$(dirname "$0")"
exec .venv/bin/streamlit run code/app.py
