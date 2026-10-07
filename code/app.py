"""Fitsched の Web 画面（起動: streamlit run code/app.py）"""
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import fitsched as fs

WEEKDAYS_JA = '月火水木金土日'
PROPOSAL_COLOR = '#2a78d6'
BUSY_COLOR = '#b5b4ad'
SCHEDULED_COLOR = '#1baf7a'

st.set_page_config(page_title='Fitsched', page_icon='🏋️', layout='wide')


@st.cache_data(ttl=600, show_spinner='カレンダーを読み込んでいます…')
def load_events(source, year, start, end):
    if source == 'google':
        return fs.load_from_google(year, start, end)
    if source == 'demo':
        return fs.load_demo(year, start, end)
    return fs.load_events_csv(fs.HISTORY_CSV), fs.empty_events()


@st.cache_resource(show_spinner='過去のトレーニングから学習しています…')
def train(history_df, year):
    return fs.train_model(history_df, year)


def next_month_start():
    today = date.today()
    return date(today.year + today.month // 12, today.month % 12 + 1, 1)


def day_label(ts):
    return f'{ts.month}/{ts.day}({WEEKDAYS_JA[ts.weekday()]})'


def split_by_day(df, start, end, hour_lo, hour_hi):
    """予定を日ごとの [開始時, 終了時) に分け、表示範囲の時間帯に切り詰める"""
    rows = []
    for _, ev in df.iterrows():
        s, e = max(ev['start'], start), min(ev['end'], end)
        day = s.normalize()
        while day < e:
            lo = max(s, day + pd.Timedelta(hours=hour_lo))
            hi = min(e, day + pd.Timedelta(hours=hour_hi))
            if lo < hi:
                rows.append({
                    'date': day, 'label': day_label(day),
                    'y0': (lo - day) / pd.Timedelta(hours=1),
                    'y1': (hi - day) / pd.Timedelta(hours=1),
                    'title': ev.get('summary', ''),
                    'time': f"{lo:%H:%M}–{hi:%H:%M}",
                })
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=['date', 'label', 'y0', 'y1', 'title', 'time'])


def month_chart(busy, scheduled, proposals, month_start, hour_lo, hour_hi):
    """1か月分の 日×時刻 のグリッドに、予定（グレー）・予定済みトレーニング（緑）・提案（青）を並べる"""
    days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq='D')
    order = [day_label(d) for d in days]
    busy = busy.assign(kind='予定', prob='')
    scheduled = scheduled.assign(kind='予定済みトレーニング', prob='')
    proposals = proposals.assign(kind='提案')
    data = pd.concat([busy, scheduled, proposals], ignore_index=True)

    return alt.Chart(data).mark_bar(cornerRadius=3).encode(
        x=alt.X('label:O', sort=order, scale=alt.Scale(domain=order, paddingInner=0.15),
                title=None, axis=alt.Axis(labelAngle=-60, labelOverlap=False, labelFontSize=10)),
        y=alt.Y('y0:Q', scale=alt.Scale(domain=[hour_hi, hour_lo], nice=False),
                title='時刻', axis=alt.Axis(values=list(range(hour_lo, hour_hi + 1, 2)), format='d')),
        y2='y1:Q',
        color=alt.Color('kind:N', title=None,
                        scale=alt.Scale(domain=['予定', '予定済みトレーニング', '提案'],
                                        range=[BUSY_COLOR, SCHEDULED_COLOR, PROPOSAL_COLOR]),
                        legend=alt.Legend(orient='top')),
        tooltip=[alt.Tooltip('label:N', title='日付'), alt.Tooltip('time:N', title='時間'),
                 alt.Tooltip('title:N', title='内容'), alt.Tooltip('prob:N', title='トレーニング確率')],
    ).properties(height=380)


# ---- サイドバー: 条件 ----
with st.sidebar:
    st.header('条件')
    SOURCE_LABELS = {'google': 'Googleカレンダー', 'csv': '保存済みCSV（予定なし扱い）', 'demo': 'デモ（疑似データ）'}
    sources = [s for s, ok in [('google', fs.CLIENT_SECRET_FILE.exists() or fs.TOKEN_FILE.exists()),
                               ('csv', fs.HISTORY_CSV.exists()), ('demo', True)] if ok]
    source = st.radio('データ', sources, format_func=SOURCE_LABELS.get,
                      help='data/credentials.json を置くとGoogleカレンダーを使えます')
    year = st.number_input('学習に使う年', min_value=2015, max_value=date.today().year,
                           value=2023)
    # URL の ?start=YYYY-MM-DD で開始日を指定できる（期間を固定した画面を共有するとき用）
    try:
        default_start = date.fromisoformat(st.query_params.get('start', ''))
    except ValueError:
        default_start = next_month_start()
    start_date = st.date_input('提案の開始日', value=default_start)
    months = st.slider('提案する月数', 1, 12, 3)
    hour_lo, hour_hi = st.slider('トレーニングしてよい時間帯', 0, 24, (6, 22), format='%d時')
    daily_limit = st.number_input('1日の上限（回）', 1, 3, fs.DAILY_TRAINING_LIMIT)
    rest_days = st.number_input('トレーニングの間の休息日数', 0, 6, fs.REST_DAYS,
                                help='1なら連日にしない。休息日を守ると目標に届かない月は、自動で休息日を減らして埋めます。')

    start = pd.Timestamp(start_date)
    end = start + pd.DateOffset(months=months)
    periods = pd.period_range(start=start, periods=months, freq='M')

    st.subheader('月ごとの目標回数')
    goals = {}
    cols = st.columns(2)
    for i, p in enumerate(periods):
        goals[(p.year, p.month)] = cols[i % 2].number_input(
            f'{p.year}年{p.month}月', 0, 31, fs.monthly_goal(fs.MONTHLY_TRAINING_GOAL, p.year, p.month),
            key=f'goal-{p}')

    if st.button('カレンダーを再読み込み', width='stretch'):
        load_events.clear()

# ---- 本体 ----
st.title('Fitsched')
st.caption('過去の予定から「トレーニングしやすい時間帯」を学習し、空き時間にトレーニング枠を提案します。')

try:
    history_df, future_df = load_events(source, int(year), start, end)
except Exception as e:  # 認証切れ・ネットワークなど
    st.error(f'カレンダーを読み込めませんでした: {e}')
    if source == 'google':
        st.info('認証が切れている場合は data/token.json を削除して再読み込みすると、ブラウザで再認証できます。')
    st.stop()

try:
    model, _, metrics = train(history_df, int(year))
except ValueError as e:
    st.error(str(e))
    st.stop()

unwanted_hours = [h for h in range(24) if not hour_lo <= h < hour_hi]
proposals = fs.propose_slots(model, future_df, start, end, goals, int(daily_limit), unwanted_hours, int(rest_days))
scheduled = fs.scheduled_trainings(future_df, start, end)

total_goal = sum(goals.values())
c1, c2, c3 = st.columns(3)
c1.metric('予定済み + 提案 / 目標', f'{len(scheduled) + len(proposals)} / {total_goal}回',
          help=f'カレンダーに入っているトレーニング {len(scheduled)}回 + 提案 {len(proposals)}回')
c2.metric(f'{year}年のトレーニング実績', f"{metrics['training_count']}回")
c3.metric('モデルの判別力 (ROC-AUC)', f"{metrics['roc_auc']:.2f}",
          help='0.5で当てずっぽう、1.0で完璧。月ごとに分け、学習に使っていない月で測った値です。')

if source == 'demo':
    st.info('デモモードです。synth.py が作った疑似カレンダー（授業・バイト・トレーニングの習慣）で学習・提案しています。')
elif source == 'csv':
    st.warning('CSVモードでは提案期間の予定が分からないため、すべて空いているものとして提案しています。')

busy_all = split_by_day(future_df[~fs.is_training_event(future_df)], start, end, hour_lo, hour_hi)
scheduled_all = split_by_day(scheduled, start, end, hour_lo, hour_hi)
prop_rows = proposals.assign(summary='トレーニング（提案）')
prop_all = split_by_day(prop_rows, start, end, hour_lo, hour_hi)
prop_all['prob'] = [f'{p:.0%}' for p in proposals['proposed_training']]

tabs = st.tabs([f'{p.year}年{p.month}月' for p in periods])
for tab, p in zip(tabs, periods):
    with tab:
        m_start = p.start_time
        in_month = lambda df: df[(df['date'] >= m_start) & (df['date'] <= p.end_time)]
        m_props = in_month(prop_all)
        m_sched = in_month(scheduled_all)
        already = len(fs.in_month(scheduled, p))
        goal = goals[(p.year, p.month)]
        if already:
            st.info(f'カレンダーに入っているトレーニング {already}回を目標に数えています。')
        if already + len(m_props) < goal:
            st.warning(f'目標 {goal}回に対して、予定済みと提案を合わせて {already + len(m_props)}回です。')

        st.altair_chart(month_chart(in_month(busy_all), m_sched, m_props, m_start, hour_lo, hour_hi),
                        width='stretch')

        table = pd.DataFrame({
            '日付': m_props['label'],
            '時間': m_props['time'],
            'トレーニング確率': m_props['prob'],
        })
        st.dataframe(table, hide_index=True, width='stretch')
