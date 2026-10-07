import argparse
import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

# パスはこのファイルからの相対で解決する（Fitsched/data/ 以下）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
CLIENT_SECRET_FILE = DATA_DIR / 'credentials.json'
TOKEN_FILE = DATA_DIR / 'token.json'
HISTORY_CSV = DATA_DIR / '2023_calendar_events.csv'

# Google Calendar APIの設定
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
LOCAL_TZ = 'Asia/Tokyo'

TRAINING_KEYWORD = 'トレーニング'
# 時刻・曜日に加え、同じ日の前後の予定（トレーニング以外）との間隔を使う
FEATURES = ['hour', 'weekday', 'is_weekend', 'gap_before', 'gap_after', 'day_busy_hours']
NO_EVENT_GAP = 24  # その日に前後の予定が無いときの間隔（時間）
HOUR = pd.Timedelta(hours=1)

# ユーザーが希望する月ごとのトレーニング回数（キーは月）
MONTHLY_TRAINING_GOAL = {
    1: 10,  # 1月に10回トレーニング
    2: 8,   # 2月に8回トレーニング
    3: 12,  # 3月に12回トレーニング
    4: 8,
    5: 6,
    6: 8,
    7: 11,
    8: 8,
}
DEFAULT_MONTHLY_GOAL = 8  # 上に無い月の回数

# 日ごとのトレーニング回数の上限
DAILY_TRAINING_LIMIT = 1

# トレーニングの間に空ける休息日数（1なら連日にしない）
REST_DAYS = 1

# トレーニングをしたくない時間帯
UNWANTED_HOURS = list(range(0, 6)) + list(range(22, 24))


def get_calendar_service():
    """Google Calendar APIクライアントを作成する（認証情報がある場合のみ呼ぶ）"""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
        TOKEN_FILE.chmod(0o600)
    return build('calendar', 'v3', credentials=creds)


def get_calendar_events(service, calendar_id, time_min, time_max):
    """Googleカレンダーからイベントを取得する（ページングも辿る）"""
    events = []
    page_token = None
    while True:
        events_result = service.events().list(
            calendarId=calendar_id, timeMin=time_min, timeMax=time_max, maxResults=2500,
            singleEvents=True, orderBy='startTime', pageToken=page_token).execute()
        events.extend(events_result.get('items', []))
        page_token = events_result.get('nextPageToken')
        if not page_token:
            return events


def to_local_naive(value):
    """dateTime（タイムゾーン付き）は日本時間に揃え、date（終日）はそのまま日付として扱う"""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(LOCAL_TZ).tz_localize(None)
    return ts


def events_to_dataframe(events):
    """イベントデータをDataFrameに変換する"""
    data = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        summary = event.get('summary', 'No Title')
        data.append([to_local_naive(start), to_local_naive(end), summary])

    df = pd.DataFrame(data, columns=['start', 'end', 'summary'])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['duration'] = (df['end'] - df['start']).dt.total_seconds() / 60.0
    return df


def load_events_csv(path):
    """保存済みCSVからイベントを読み込む"""
    df = pd.read_csv(path)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df


def create_slots(start_date, end_date, freq='60min'):
    """指定期間内のスロットを作成する"""
    slots = pd.date_range(start=start_date, end=end_date, freq=freq)
    return pd.DataFrame({'start': slots[:-1], 'end': slots[1:]})


def is_training_event(events):
    return events['summary'].str.contains(TRAINING_KEYWORD, na=False)


def overlap_mask(slots, events):
    """1時間刻みで連続したスロットのうち、いずれかの予定と重なるものを True にする"""
    n = len(slots)
    if n == 0 or events.empty:
        return np.zeros(n, dtype=bool)
    t0 = slots['start'].iloc[0]
    first = np.floor((events['start'] - t0) / HOUR).clip(0, n).astype(int)
    last = np.ceil((events['end'] - t0) / HOUR).clip(0, n).astype(int)
    mark = np.zeros(n + 1, dtype=int)
    np.add.at(mark, first.to_numpy(), 1)
    np.add.at(mark, last.to_numpy(), -1)
    return np.cumsum(mark[:-1]) > 0


def split_events_by_day(events):
    """日をまたぐ予定を日ごとに分ける"""
    rows = []
    for s, e in zip(events['start'], events['end']):
        day = s.normalize()
        while day < e:
            nxt = day + pd.Timedelta(days=1)
            rows.append((day, max(s, day), min(e, nxt)))
            day = nxt
    df = pd.DataFrame(rows, columns=['date', 'start', 'end'])
    return df.astype('datetime64[ns]')


def add_features(slots, events):
    """時刻・曜日と、同じ日の前後の予定（トレーニング以外）との間隔を特徴量にする"""
    slots['hour'] = slots['start'].dt.hour
    slots['weekday'] = slots['start'].dt.weekday
    slots['is_weekend'] = (slots['weekday'] >= 5).astype(int)

    ev = split_events_by_day(events[~is_training_event(events)])
    keys = pd.DataFrame({
        'start': slots['start'].astype('datetime64[ns]').to_numpy(),
        'end': slots['end'].astype('datetime64[ns]').to_numpy(),
        'date': slots['start'].dt.normalize().astype('datetime64[ns]').to_numpy(),
        'order': np.arange(len(slots)),
    })
    if ev.empty:
        gap_before = gap_after = np.full(len(slots), float(NO_EVENT_GAP))
        busy_hours = np.zeros(len(slots))
    else:
        prev = pd.merge_asof(
            keys.sort_values('start'), ev[['date', 'end']].rename(columns={'end': 'prev_end'}).sort_values('prev_end'),
            left_on='start', right_on='prev_end', by='date', direction='backward').sort_values('order')
        nxt = pd.merge_asof(
            keys.sort_values('end'), ev[['date', 'start']].rename(columns={'start': 'next_start'}).sort_values('next_start'),
            left_on='end', right_on='next_start', by='date', direction='forward').sort_values('order')
        gap_before = ((prev['start'] - prev['prev_end']) / HOUR).fillna(NO_EVENT_GAP).to_numpy()
        gap_after = ((nxt['next_start'] - nxt['end']) / HOUR).fillna(NO_EVENT_GAP).to_numpy()
        per_day = ((ev['end'] - ev['start']) / HOUR).groupby(ev['date']).sum()
        busy_hours = keys['date'].map(per_day).fillna(0).to_numpy()

    slots['gap_before'] = gap_before
    slots['gap_after'] = gap_after
    slots['day_busy_hours'] = busy_hours
    return slots


def plot_training_hours(training_events):
    """トレーニングイベントの時間帯分布を表示する"""
    import matplotlib.pyplot as plt

    plt.hist(training_events['start'].dt.hour, bins=24, range=(0, 24))
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Training Events')
    plt.title('Distribution of Training Events by Hour')
    plt.show()


def build_training_table(history_df, year, unwanted_hours=UNWANTED_HOURS):
    """1時間枠ごとの学習用データ。トレーニングを入れる余地があった枠（練習してよい時間帯で、
    ほかの予定と重ならない枠）だけを残す。深夜や授業中は最初から候補にならないので学習させない"""
    slots = create_slots(f'{year}-01-01', f'{year + 1}-01-01')
    training = is_training_event(history_df)
    slots['is_training'] = overlap_mask(slots, history_df[training]).astype(int)
    busy_other = overlap_mask(slots, history_df[~training])
    slots = add_features(slots, history_df)
    keep = ~slots['hour'].isin(unwanted_hours) & (~busy_other | (slots['is_training'] == 1))
    return slots[keep].reset_index(drop=True)


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1)


def cross_validate(table, features=FEATURES, model_factory=make_model):
    """月ごとに分けた交差検証。評価する月のデータは学習に使わないので、隣の時間枠からの“答え漏れ”が起きない"""
    groups = table['start'].dt.to_period('M')
    n_splits = min(6, groups.nunique())
    oof = np.zeros(len(table))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(table, groups=groups):
        model = model_factory().fit(table.iloc[train_idx][features], table.iloc[train_idx]['is_training'])
        oof[test_idx] = model.predict_proba(table.iloc[test_idx][features])[:, 1]
    return oof


def train_model(history_df, year, unwanted_hours=UNWANTED_HOURS):
    """過去のトレーニング実績から、時間帯ごとのトレーニングしやすさを学習する"""
    training_events = history_df[is_training_event(history_df)]
    table = build_training_table(history_df, year, unwanted_hours)
    y = table['is_training']
    if y.sum() == 0:
        raise SystemExit(f'「{TRAINING_KEYWORD}」を含む予定が{year}年に見つかりませんでした。')

    # トレーニング枠は候補のうち数%しかないので、Accuracy ではなく並び順の指標で見る
    oof = cross_validate(table)
    metrics = {
        'training_count': len(training_events),
        'slot_count': len(table),
        'positive_rate': y.mean(),
        'roc_auc': roc_auc_score(y, oof),
        'avg_precision': average_precision_score(y, oof),
    }
    model = make_model().fit(table[FEATURES], y)
    return model, training_events, metrics


def print_metrics(metrics):
    print(f"学習データ: {metrics['training_count']}件のトレーニング / 候補 {metrics['slot_count']}スロット")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f} (0.5 = 当てずっぽう)")
    print(f"Average Precision: {metrics['avg_precision']:.3f} "
          f"(当てずっぽうなら {metrics['positive_rate']:.3f})")


def monthly_goal(goals, year, month):
    """目標回数は (年, 月) → 月 → 既定値 の順で探す"""
    return goals.get((year, month), goals.get(month, DEFAULT_MONTHLY_GOAL))


def scheduled_trainings(future_df, start, end):
    """提案期間内に、すでにカレンダーに入っているトレーニング"""
    in_period = (future_df['start'] >= start) & (future_df['start'] < end)
    return future_df[in_period & is_training_event(future_df)].sort_values('start')


def propose_slots(model, future_df, start, end, goals=MONTHLY_TRAINING_GOAL,
                  daily_limit=DAILY_TRAINING_LIMIT, unwanted_hours=UNWANTED_HOURS, rest_days=REST_DAYS):
    """空いている時間帯から、トレーニング確率の高い順に月ごとの目標回数まで選ぶ。

    - カレンダーに入っているトレーニングは目標回数に数え、休息日の判定にも使う
    - トレーニング日の前後 rest_days 日は避ける。それでは目標に届かない月だけ、休息日を1日ずつ減らして埋める
    """
    future_slots = create_slots(start, end)
    # 予定がある時間帯を使用不可にする
    future_slots['available'] = (~overlap_mask(future_slots, future_df)).astype(int)
    future_slots = add_features(future_slots, future_df)
    future_slots['proposed_training'] = model.predict_proba(future_slots[FEATURES])[:, 1]

    candidates = future_slots[
        (future_slots['available'] == 1) & ~future_slots['hour'].isin(unwanted_hours)
    ].sort_values('proposed_training', ascending=False, kind='stable')

    # 日付 → その日のトレーニング回数（予定済み + 提案済み）
    day_count = scheduled_trainings(future_df, start, end)['start'].dt.normalize().value_counts().to_dict()

    def too_close(day, gap):
        return any(day_count.get(day + sign * pd.Timedelta(days=k), 0) > 0
                   for k in range(1, gap + 1) for sign in (1, -1))

    selected = []
    for (year, month), month_slots in candidates.groupby(
            [candidates['start'].dt.year, candidates['start'].dt.month], sort=False):
        already = sum(n for d, n in day_count.items() if (d.year, d.month) == (year, month))
        remaining = monthly_goal(goals, year, month) - already
        for gap in range(rest_days, -1, -1):
            for idx, slot in month_slots.iterrows():
                if remaining <= 0:
                    break
                day = slot['start'].normalize()
                if idx in selected or day_count.get(day, 0) >= daily_limit or too_close(day, gap):
                    continue
                day_count[day] = day_count.get(day, 0) + 1
                selected.append(idx)
                remaining -= 1

    return future_slots.loc[selected].sort_values('start')


def load_from_google(year, start, end):
    """Googleカレンダーから学習用の予定と提案期間の予定を取得する"""
    service = get_calendar_service()
    calendar_id = 'primary'
    history_df = events_to_dataframe(get_calendar_events(
        service, calendar_id, f'{year}-01-01T00:00:00Z', f'{year + 1}-01-01T00:00:00Z'))
    # 学習データを保存
    history_df.to_csv(DATA_DIR / f'{year}_calendar_events.csv', index=False)
    future_df = events_to_dataframe(get_calendar_events(
        service, calendar_id,
        start.tz_localize(LOCAL_TZ).isoformat(), end.tz_localize(LOCAL_TZ).isoformat()))
    return history_df, future_df


def default_source():
    return 'google' if CLIENT_SECRET_FILE.exists() or TOKEN_FILE.exists() else 'csv'


def in_month(df, period):
    return df[(df['start'].dt.year == period.year) & (df['start'].dt.month == period.month)]


def print_proposals(optimal_slots, scheduled, start, months):
    """結果を月ごとに表示し、希望の回数を満たせない場合に通知する"""
    for period in pd.period_range(start=start, periods=months, freq='M'):
        goal = monthly_goal(MONTHLY_TRAINING_GOAL, period.year, period.month)
        month_slots = in_month(optimal_slots, period)
        already = len(in_month(scheduled, period))
        label = f'{calendar.month_name[period.month]} {period.year}'
        note = f'（予定済み {already}回 + 提案 {len(month_slots)}回）' if already else ''
        if already + len(month_slots) < goal:
            print(f'\n{label}: 希望のトレーニング回数 {goal}回に対して、{already + len(month_slots)}回です{note}。')
        else:
            print(f'\n{label}: 提案されたトレーニングスロット{note}')
        if not month_slots.empty:
            print(month_slots[['start', 'end', 'proposed_training']].to_string(index=False))


def parse_args():
    today = date.today()
    next_month = date(today.year + today.month // 12, today.month % 12 + 1, 1)
    parser = argparse.ArgumentParser(description='過去のカレンダーから最適なトレーニング時間を提案する')
    parser.add_argument('--source', choices=['auto', 'google', 'csv'], default='auto',
                        help='auto: data/credentials.json があればGoogleカレンダー、無ければCSVを使う')
    parser.add_argument('--history-year', type=int, default=2023, help='学習に使う年')
    parser.add_argument('--history-csv', type=Path, default=HISTORY_CSV,
                        help='CSVモードで学習に使う予定データ')
    parser.add_argument('--future-csv', type=Path,
                        help='CSVモードで提案期間の予定として使うCSV（省略時は予定なしとして扱う）')
    parser.add_argument('--start', default=next_month.isoformat(), help='提案期間の開始日 (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=8, help='提案する月数')
    parser.add_argument('--rest-days', type=int, default=REST_DAYS,
                        help='トレーニングの間に空ける休息日数（0で連日も可）')
    parser.add_argument('--plot', action='store_true', help='トレーニング時間帯のヒストグラムを表示する')
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    if source == 'auto':
        source = default_source()

    start = pd.Timestamp(args.start).normalize()
    end = start + pd.DateOffset(months=args.months)
    year = args.history_year

    if source == 'google':
        history_df, future_df = load_from_google(year, start, end)
    else:
        print(f'CSVモード: {args.history_csv} から学習します')
        history_df = load_events_csv(args.history_csv)
        if args.future_csv:
            future_df = load_events_csv(args.future_csv)
        else:
            print('提案期間の予定は未指定のため、すべて空いているものとして扱います')
            future_df = pd.DataFrame(columns=['start', 'end', 'summary'])

    model, training_events, metrics = train_model(history_df, year)
    print_metrics(metrics)
    if args.plot:
        plot_training_hours(training_events)

    optimal_slots = propose_slots(model, future_df, start, end, rest_days=args.rest_days)
    print_proposals(optimal_slots, scheduled_trainings(future_df, start, end), start, args.months)


if __name__ == '__main__':
    main()
