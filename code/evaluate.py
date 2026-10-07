"""旧モデルと新モデルを、実データと疑似データで比べる（実行: .venv/bin/python code/evaluate.py）"""
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import Fitshed as fs
import synth

YEAR = 2023
LEGACY_FEATURES = ['hour', 'day', 'weekday', 'month']


def legacy_reported_auc(history_df):
    """旧コードの評価そのもの：全24時間の枠をランダムに分けて測る"""
    slots = fs.create_slots(f'{YEAR}-01-01', f'{YEAR + 1}-01-01')
    slots['is_training'] = fs.overlap_mask(slots, history_df[fs.is_training_event(history_df)]).astype(int)
    slots['hour'] = slots['start'].dt.hour
    slots['day'] = slots['start'].dt.day
    slots['weekday'] = slots['start'].dt.weekday
    slots['month'] = slots['start'].dt.month
    X_train, X_test, y_train, y_test = train_test_split(
        slots[LEGACY_FEATURES], slots['is_training'], test_size=0.2, random_state=42, stratify=slots['is_training'])
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def legacy_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)


def scores(table, oof, truth=None):
    row = {
        'ROC-AUC': roc_auc_score(table['is_training'], oof),
        'AP': average_precision_score(table['is_training'], oof),
    }
    if truth is not None:
        # 本当の確率と予測の並び順がどれだけ一致しているか（1が完全一致）
        row['正解との相関'] = spearmanr(oof, truth).statistic
    return row


def evaluate(name, history_df, truth_df=None):
    table = fs.build_training_table(history_df, YEAR)
    table['day'] = table['start'].dt.day
    table['month'] = table['start'].dt.month
    truth = None
    if truth_df is not None:
        truth = table[['start']].merge(truth_df, on='start', how='left')['true_prob'].fillna(0).to_numpy()

    rows = {
        '旧モデル（旧評価のまま）': {'ROC-AUC': legacy_reported_auc(history_df)},
        '旧特徴量': scores(table, fs.cross_validate(table, LEGACY_FEATURES, legacy_model), truth),
        '新モデル': scores(table, fs.cross_validate(table), truth),
    }
    if truth is not None:
        rows['正解の確率（上限）'] = scores(table, truth, truth)
    df = pd.DataFrame(rows).T
    print(f'\n## {name}（トレーニング {int(table["is_training"].sum())}件 / 候補 {len(table)}枠）')
    print(df.to_string(float_format=lambda v: f'{v:.3f}', na_rep='-'))
    return df


def main():
    print('旧評価以外は、候補枠（6〜22時・他の予定なし）で月ごとに分けた交差検証。AP の当てずっぽう値はトレーニング枠の割合。')
    evaluate('実データ 2023年', fs.load_events_csv(fs.HISTORY_CSV))
    for seed in range(3):
        events, truth = synth.generate(YEAR, seed=seed)
        evaluate(f'疑似データ seed={seed}', events, truth)


if __name__ == '__main__':
    main()
