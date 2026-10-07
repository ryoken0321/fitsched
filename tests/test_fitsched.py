import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import fitsched as fs  # noqa: E402


def events(rows):
    return pd.DataFrame({
        'start': pd.to_datetime([r[0] for r in rows], format='ISO8601'),
        'end': pd.to_datetime([r[1] for r in rows], format='ISO8601'),
        'summary': [r[2] for r in rows],
    })


def test_overlap_mask_marks_partially_covered_slots():
    slots = fs.create_slots('2023-01-01 08:00', '2023-01-01 12:00')
    mask = fs.overlap_mask(slots, events([('2023-01-01 09:30', '2023-01-01 10:30', '授業')]))
    assert mask.tolist() == [False, True, True, False]


def test_drop_all_day_keeps_timed_events():
    df = events([('2023-01-01', '2023-01-02', '誕生日'), ('2023-01-02 10:00', '2023-01-02 11:00', '授業')])
    assert fs.drop_all_day(df)['summary'].tolist() == ['授業']


def test_events_to_dataframe_skips_date_only_events():
    df = fs.events_to_dataframe([
        {'start': {'date': '2023-01-01'}, 'end': {'date': '2023-01-02'}, 'summary': '祝日'},
        {'start': {'dateTime': '2023-01-02T10:00:00+09:00'}, 'end': {'dateTime': '2023-01-02T11:00:00+09:00'},
         'summary': '授業'},
    ])
    assert df['summary'].tolist() == ['授業']
    assert df['start'].iloc[0] == pd.Timestamp('2023-01-02 10:00')


def test_proposal_end_aligns_to_calendar_months():
    assert fs.proposal_end(pd.Timestamp('2026-12-15'), 3) == pd.Timestamp('2027-03-01')


def test_goal_is_prorated_for_partial_month():
    start, end = pd.Timestamp('2026-11-16'), pd.Timestamp('2026-12-01')
    assert fs.goal_in_period({}, 2026, 11, start, end, default=8) == 4


def test_positive_proba_without_positive_class():
    class OneClass:
        classes_ = np.array([0])
    assert fs.positive_proba(OneClass(), pd.DataFrame({'x': [1, 2]})).tolist() == [0, 0]


class ConstantModel:
    """どの枠も同じ確率を返すモデル（提案の制約だけを確かめるため）"""
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


def test_propose_slots_respects_goal_daily_limit_and_rest_days():
    start, end = pd.Timestamp('2026-11-01'), pd.Timestamp('2026-12-01')
    slots = fs.propose_slots(ConstantModel(), fs.empty_events(), start, end, goals={11: 8},
                             daily_limit=1, rest_days=1)
    days = slots['start'].dt.normalize()
    assert len(slots) == 8
    assert days.is_unique
    assert (days.diff().dropna() >= pd.Timedelta(days=2)).all()


def test_propose_slots_counts_scheduled_training_and_avoids_busy_hours():
    start, end = pd.Timestamp('2026-11-01'), pd.Timestamp('2026-12-01')
    future = events([('2026-11-02 10:00', '2026-11-02 11:00', 'トレーニング'),
                     ('2026-11-05 06:00', '2026-11-05 22:00', '旅行')])
    slots = fs.propose_slots(ConstantModel(), future, start, end, goals={11: 3}, rest_days=1)
    assert len(slots) == 2
    assert not (slots['start'].dt.normalize() == pd.Timestamp('2026-11-05')).any()


def test_cross_validate_needs_two_months():
    table = pd.DataFrame({'start': pd.date_range('2023-01-01', periods=10, freq='h'),
                          'is_training': [0, 1] * 5, **{f: 0 for f in fs.FEATURES}})
    with pytest.raises(ValueError):
        fs.cross_validate(table)
