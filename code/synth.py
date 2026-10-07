"""モデル検証用の疑似カレンダーを作る

生活パターン（学校・塾バイト・私用）と「トレーニングを入れる習慣」を決め打ちで生成する。
習慣のルールが分かっているので、各時間枠の「本当のトレーニング確率」も一緒に返せる。
モデルがこの確率の並び順をどこまで再現できるかで、特徴量や評価方法の良し悪しを確かめる。
"""
import numpy as np
import pandas as pd

# 曜日ごとの授業時間（学期中の平日のみ）
SCHOOL_HOURS = {0: (9, 15), 1: (10, 16), 2: (9, 13), 3: (13, 18), 4: (10, 15)}
SEMESTER_MONTHS = {1, 4, 5, 6, 7, 10, 11, 12}
TRAIN_HOURS = range(6, 22)  # トレーニングの候補にする開始時刻


def habit_weight(hour, weekday, gap_before, gap_after):
    """トレーニングを入れる習慣（正解のルール）。大きいほどその枠を選びやすい"""
    if 8 <= hour <= 10:
        w = 3.0      # 朝型
    elif weekday >= 5 and 15 <= hour <= 17:
        w = 3.0      # 週末の夕方
    elif 19 <= hour <= 20:
        w = 1.5      # 夜
    else:
        w = 0.5
    if gap_before is not None and gap_before <= 1:
        w *= 3.0     # 授業やバイトの直後にそのまま行く
    if gap_after is not None and gap_after < 1:
        w *= 0.2     # 直後に予定があると避ける
    return w


def _overlaps(busy, s, e):
    return any(s < be and e > bs for bs, be in busy)


def generate(year=2023, seed=0):
    """(events, truth) を返す。events は start,end,summary、truth は各時間枠の本当の確率"""
    rng = np.random.default_rng(seed)
    events, truth = [], []

    for day in pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D'):
        wd = day.weekday()
        busy = []

        def add(s, e, summary):
            busy.append((s, e))
            events.append((day + pd.Timedelta(hours=s), day + pd.Timedelta(hours=e), summary))

        if day.month in SEMESTER_MONTHS and wd in SCHOOL_HOURS:
            add(*SCHOOL_HOURS[wd], '学校')
        if wd in (1, 3) and rng.random() < 0.8:
            add(18, 21, '塾バイト')
        if wd == 5 and rng.random() < 0.5:
            add(13, 17, '塾バイト')
        if rng.random() < 0.3:
            s = int(rng.integers(10, 20))
            e = min(s + int(rng.integers(1, 4)), 23)
            if not _overlaps(busy, s, e):
                add(s, e, '予定')

        # その日の空き枠に習慣の重みを付け、トレーニングする日なら重みに比例して1枠選ぶ
        free, weights = [], []
        for h in TRAIN_HOURS:
            if _overlaps(busy, h, h + 1):
                continue
            ends = [be for _, be in busy if be <= h]
            starts = [bs for bs, _ in busy if bs >= h + 1]
            gap_before = h - max(ends) if ends else None
            gap_after = min(starts) - (h + 1) if starts else None
            free.append(h)
            weights.append(habit_weight(h, wd, gap_before, gap_after))
        if not free:
            continue

        p_day = 0.35 if wd >= 5 else 0.25
        probs = p_day * np.array(weights) / sum(weights)
        truth.extend((day + pd.Timedelta(hours=h), p) for h, p in zip(free, probs))
        if rng.random() < p_day:
            h = int(rng.choice(free, p=probs / probs.sum()))
            add(h, h + 1, 'トレーニング')

    events_df = pd.DataFrame(events, columns=['start', 'end', 'summary']).sort_values('start', ignore_index=True)
    events_df['duration'] = (events_df['end'] - events_df['start']).dt.total_seconds() / 60.0
    truth_df = pd.DataFrame(truth, columns=['start', 'true_prob'])
    return events_df, truth_df


if __name__ == '__main__':
    ev, tr = generate()
    print(ev['summary'].value_counts())
