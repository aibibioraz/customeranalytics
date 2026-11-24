"""
FreedomProfile — дашборд результатов (Streamlit).

Запуск из корня workspace: streamlit run freedomprofile/notebooks/app.py
Или из freedomprofile/:    streamlit run notebooks/app.py
Зависимости:               streamlit plotly duckdb pandas numpy scikit-learn
Данные: build_user_features.py и ноутбуки 04-08; затем из freedomprofile запустите
python notebooks/09_model_improvements.py для churn/Freedom Score и
python notebooks/10_propensity_improvements.py для калибровки propensity и охватных метрик.
Другая папка:              SAPP_PROCESSED=/путь/к/processed streamlit run freedomprofile/notebooks/app.py
"""
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

st.set_page_config(page_title="FreedomProfile", page_icon="📊", layout="wide")

DIR = Path(os.environ.get("SAPP_PROCESSED", Path(__file__).resolve().parents[1] / "data" / "processed"))
REQUIRED = ["model_table.parquet", "freedom_score.parquet", "propensity_scores.parquet", "nba_actions.parquet",
            "churn_predictions_test.parquet", "churn_feature_importance.csv", "freedom_score_importance.csv",
            "propensity_summary.csv", "channel_summary.csv", "channel_raw_profile.csv"]
missing = [f for f in REQUIRED if not (DIR / f).exists()]
if missing:
    st.error(f"В папке {DIR} не хватает файлов: {', '.join(missing)}.\n\n"
             "Сначала запустите build_user_features.py и ноутбуки 04–08 (они сохраняют результаты в data/processed/).")
    st.stop()

# Совместимость со старыми и новыми версиями Streamlit (use_container_width заменили на width="stretch")
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
W = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

st.markdown("""
<style>
:root { --fp-green:#08c96c; --fp-blue:#7399ed; --fp-ink:#15212b; --fp-muted:#647382; }
.stApp { background: radial-gradient(ellipse at 2% 96%, #e2f6ff 0, transparent 27%),
                     radial-gradient(ellipse at 98% 96%, #dff9ef 0, transparent 28%), #fbfcfd; }
[data-testid="stHeader"] { background:rgba(255,255,255,.88); border-bottom:1px solid #e9edf1; }
[data-testid="stSidebar"] { display:none; }
.block-container { max-width:1540px; padding:1.15rem 4.2rem 4rem; }
h1,h2,h3,p,div,span { font-family: Inter, "Segoe UI", Arial, sans-serif; }
h1 { color:#17232d; letter-spacing:-.045em; font-weight:750; }
h2,h3 { color:#192932; letter-spacing:-.025em; }
.fp-brand { display:flex; align-items:center; gap:9px; height:48px; color:#111a22; font-size:20px; font-weight:800; letter-spacing:-.04em; }
.fp-shield { width:25px; height:30px; display:grid; place-items:center; color:#073c25; font-size:14px; font-weight:900;
  background:linear-gradient(145deg,#08d46c,#00a953); clip-path:polygon(0 0,100% 0,100% 68%,82% 88%,50% 100%,18% 88%,0 68%); }
.fp-stamp { margin-top:11px; display:inline-block; border:1px solid #e5edf1; border-radius:999px; padding:8px 14px;
  color:#536677; background:#fff; font-size:13px; white-space:nowrap; }
div[data-testid="stRadio"] > div { gap:5px; }
div[data-testid="stRadio"] label { border-radius:999px; padding:7px 12px; color:#657483; font-size:15px; }
div[data-testid="stRadio"] label:has(input:checked) { color:#10241a; background:#e9fbf1; font-weight:650; }
div[data-testid="stRadio"] label > div:first-child { display:none; }
div[data-testid="stMetric"] { min-height:116px; padding:21px 22px; border:1px solid #e8eef1; border-radius:19px;
  background:rgba(255,255,255,.92); box-shadow:0 10px 32px rgba(26,53,66,.045); }
div[data-testid="stMetricLabel"] { color:#71808d; font-size:14px; }
div[data-testid="stMetricValue"] { color:#142a22; font-size:30px; font-weight:750; }
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] { border:1px solid #e8eef1; border-radius:18px;
  background:#fff; padding:10px; box-shadow:0 10px 30px rgba(26,53,66,.035); }
div[data-testid="stAlert"] { border-radius:15px; }
div.stButton > button, div[data-testid="stDownloadButton"] button { border:0; border-radius:13px; color:#082b1a;
  background:linear-gradient(105deg,#7b9bed 0%,#18ce8b 48%,#0bd467 100%); font-weight:700; }
.fp-hero { position:relative; overflow:hidden; min-height:380px; margin:24px 0 22px; padding:48px 5.5%;
  display:grid; grid-template-columns:1.2fr .8fr; align-items:center; border:1px solid #edf1f3; border-radius:28px;
  background:radial-gradient(ellipse at 85% 95%,rgba(62,225,166,.18),transparent 34%),
             radial-gradient(ellipse at 92% 20%,rgba(112,157,244,.17),transparent 36%),#fff;
  box-shadow:0 24px 65px rgba(38,68,82,.06); }
.fp-eyebrow { margin-bottom:18px; color:#688075; font-size:12px; font-weight:750; letter-spacing:.18em; text-transform:uppercase; }
.fp-hero h1 { margin:0; font-size:clamp(44px,5.2vw,76px); line-height:1.04; }
.fp-gradient-text { margin-top:9px; font-size:clamp(31px,3.8vw,54px); line-height:1.12; font-weight:800; letter-spacing:-.045em;
  background:linear-gradient(100deg,#7898e9 0%,#12c98e 54%,#03bf59 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.fp-hero-copy { max-width:590px; margin:23px 0 0; color:#53616b; font-size:19px; line-height:1.65; }
.fp-pills { display:flex; flex-wrap:wrap; gap:9px; margin-top:24px; }
.fp-pill { border:1px solid #e4eee8; border-radius:999px; padding:9px 13px; color:#4d6659; background:#ffffffc9; font-size:13px; }
.fp-art { position:relative; height:280px; display:grid; place-items:center; }
.fp-glow { position:absolute; width:250px; height:250px; border-radius:50%; background:linear-gradient(135deg,#92acf7,#75edb1); opacity:.25; filter:blur(22px); }
.fp-phone { position:relative; width:174px; height:245px; padding:17px 15px; border:1px solid #b9c8c5; border-radius:27px;
  background:linear-gradient(160deg,#fff,#f2fff8); box-shadow:0 22px 50px #315b4b30; transform:rotate(-5deg); }
.fp-notch { width:55px; height:8px; margin:-8px auto 17px; border-radius:99px; background:#253c3d; }
.fp-phone-mark { width:54px; height:62px; margin:15px auto 13px; display:grid; place-items:center; color:#034c29; font-size:25px; font-weight:900;
  background:linear-gradient(140deg,#00d867,#00a651); clip-path:polygon(0 0,100% 0,100% 68%,82% 88%,50% 100%,18% 88%,0 68%); }
.fp-score-label { text-align:center; color:#76867f; font-size:11px; letter-spacing:.1em; text-transform:uppercase; }
.fp-score-value { text-align:center; color:#182d23; font-size:33px; font-weight:800; }
.fp-score-track { height:7px; margin:12px 3px; border-radius:9px; background:linear-gradient(90deg,#7298ed,#11ce88,#0bc45c); }
.fp-mini-bars { height:41px; display:flex; align-items:end; gap:6px; margin:10px 4px 0; }
.fp-mini-bars i { flex:1; border-radius:4px 4px 0 0; background:linear-gradient(#22d78c,#baf3d7); }
.fp-float { position:absolute; padding:12px 16px; border:1px solid #e8f0ec; border-radius:15px; background:#ffffffed;
  color:#3e5c4d; font-size:13px; font-weight:650; box-shadow:0 13px 28px #345f4b1a; }
.fp-float-one { top:29px; right:2%; }.fp-float-two { bottom:24px; left:1%; }.fp-float-three { top:111px; left:0; }
@media(max-width:900px) { .block-container{padding:1rem 1.1rem 3rem}.fp-hero{grid-template-columns:1fr;padding:32px 7%}.fp-art{height:240px}.fp-stamp{display:none} }
</style>
""", unsafe_allow_html=True)

SEG_ORDER = ["High", "Medium", "Low"]
ACTION_HINTS = {
    "1. Активация": "помочь сделать первую или новую операцию (пуш, SMS)",
    "2. Удержание (приоритет)": "кэшбэк или звонок: ценного клиента жалко терять",
    "3. Недорогое удержание": "только дешёвый канал (пуш, SMS)",
    "4. Cross-sell": "персональное предложение продукта в приложении",
    "5. Лояльность": "VIP-привилегии, без продаж",
    "6. Без воздействия": "не тратить бюджет",
}


def pq(name):
    return (DIR / name).as_posix()


# ----------------------------------------------------------------------------- загрузка данных
@st.cache_data(show_spinner=False)
def load_nba():
    return duckdb.sql(f"SELECT * FROM read_parquet('{pq('nba_actions.parquet')}')").df()


@st.cache_data(show_spinner=False)
def load_customer_features(customer_id):
    return duckdb.sql(f"""
        SELECT * FROM read_parquet('{pq('model_table.parquet')}')
        WHERE customer_id = {int(customer_id)}
        LIMIT 1
    """).df()


@st.cache_data(show_spinner=False)
def load_fs():
    return duckdb.sql(f"""
        SELECT f.customer_id, f.freedom_score, f.segment,
               CAST(m.y_value_next AS DOUBLE) AS y_value_next,
               CAST(m.y_churn AS DOUBLE) AS y_churn
        FROM read_parquet('{pq('freedom_score.parquet')}') f
        JOIN read_parquet('{pq('model_table.parquet')}') m USING (customer_id)""").df()


@st.cache_data(show_spinner=False)
def load_churn(file_modified=None):
    return duckdb.sql(f"SELECT * FROM read_parquet('{pq('churn_predictions_test.parquet')}')").df()


@st.cache_data(show_spinner=False)
def load_csv(name, index_col=None):
    return pd.read_csv(DIR / name, index_col=index_col)


@st.cache_data(show_spinner="Собираю профиль источников данных…")
def load_eda_profile():
    cache = DIR.parent / "cache"
    sources = {
        "Пользователи": ("users.parquet", "reg_date",
                         "Пропусков в ключевых полях профиля: 0%"),
        "Транзакции": ("tx.parquet", "transaction_date",
                       "Сумма: 9,6% пропусков; terminal_type: 10,5%"),
        "Покупки партнёров": ("partner.parquet", "purchase_date",
                              "События покупок у партнёров"),
        "Процессы": ("proc.parquet", "started_at",
                     "События процессов; есть старые записи до периода транзакций"),
        "Привлечение": ("acq.parquet", None,
                        "Может быть несколько источников на одного клиента; категория бывает пустой"),
    }
    rows = []
    for label, (filename, date_col, note) in sources.items():
        path = cache / filename
        if not path.exists():
            continue
        file_sql = path.as_posix()
        date_select = (f"min({date_col}) AS first_date, max({date_col}) AS last_date"
                       if date_col else "NULL AS first_date, NULL AS last_date")
        stats = duckdb.sql(f"""
            SELECT count(*) AS records, count(DISTINCT customer_id) AS customers,
                   {date_select}
            FROM read_parquet('{file_sql}')
        """).df().iloc[0]
        period = (f"{pd.to_datetime(stats.first_date).date()} — {pd.to_datetime(stats.last_date).date()}"
                  if pd.notna(stats.first_date) else "даты в источнике нет")
        rows.append({"Источник": label, "Строк": int(stats.records),
                     "Уникальных клиентов": int(stats.customers), "Период": period,
                     "Особенности данных": note})

    quality_rows = []
    tx_path = cache / "tx.parquet"
    if tx_path.exists():
        f = tx_path.as_posix()
        q = duckdb.sql(f"""
            SELECT 100 * avg((transaction_sum IS NULL)::INT) AS amount_null,
                   100 * avg((terminal_type IS NULL)::INT) AS terminal_null,
                   min(transaction_date) AS first_date, max(transaction_date) AS last_date
            FROM read_parquet('{f}')
        """).df().iloc[0]
        quality_rows.append({"Проверка": "Транзакции", "Результат":
                             f"сумма отсутствует у {q.amount_null:.1f}%; терминал — у {q.terminal_null:.1f}%",
                             "Вывод": "Пропуски обработаны при агрегации; важно не считать их нулевой суммой автоматически."})

    acq_path = cache / "acq.parquet"
    if acq_path.exists():
        f = acq_path.as_posix()
        q = duckdb.sql(f"""
            SELECT 100 * avg((secondary_category_filled IS NULL OR
                              lower(trim(coalesce(secondary_category_filled, ''))) IN ('', 'null', 'nan'))::INT) AS missing
            FROM read_parquet('{f}')
        """).df().iloc[0]
        quality_rows.append({"Проверка": "Привлечение", "Результат": f"источник неизвестен у {q.missing:.1f}% строк",
                             "Вывод": "Каналы пересекаются; сравнение не равно оценке CAC или причинного эффекта."})

    users_path = cache / "users.parquet"
    conflict = None
    if users_path.exists():
        f = users_path.as_posix()
        conflict = duckdb.sql(f"""
            WITH p AS (
                SELECT customer_id, count(*) AS n,
                       count(DISTINCT city) AS cities, count(DISTINCT gender) AS genders,
                       count(DISTINCT customer_age) AS ages
                FROM read_parquet('{f}') GROUP BY customer_id
            )
            SELECT count(*) AS customers, avg(n) AS rows_per_customer,
                   100 * avg((cities > 1)::INT) AS city_conflict,
                   100 * avg((genders > 1)::INT) AS gender_conflict,
                   100 * avg((ages > 1)::INT) AS age_conflict
            FROM p
        """).df().iloc[0]
        quality_rows.append({"Проверка": "Профили пользователей",
                             "Результат": f"{conflict.rows_per_customer:.2f} строки на клиента; конфликт возраста у {conflict.age_conflict:.1f}% клиентов",
                             "Вывод": f"Различаются город у {conflict.city_conflict:.1f}% и пол у {conflict.gender_conflict:.1f}% клиентов с повторными профилями."})

    # Approximate source overlap from distinct IDs; each source can contribute multiple rows.
    overlap = None
    available = [cache / f for f in ("users.parquet", "tx.parquet", "partner.parquet", "proc.parquet", "acq.parquet") if (cache / f).exists()]
    if len(available) >= 2:
        union_sql = " UNION ALL ".join(
            f"SELECT DISTINCT customer_id, '{p.stem}' AS source FROM read_parquet('{p.as_posix()}')"
            for p in available)
        overlap = duckdb.sql(f"""
            WITH ids AS ({union_sql}), per_user AS (
                SELECT customer_id, count(DISTINCT source) AS n_sources FROM ids GROUP BY 1
            )
            SELECT count(*) AS total, count(*) FILTER (WHERE n_sources > 1) AS multi_source
            FROM per_user
        """).df().iloc[0]

    return pd.DataFrame(rows), pd.DataFrame(quality_rows), conflict, overlap


def importance(name, top=15):
    s = load_csv(name, index_col=0).iloc[:, 0].sort_values(ascending=False)
    return (s / s.sum()).head(top)[::-1]


def choose_cost_threshold(y_true, probability, false_positive_cost, false_negative_cost):
    """Choose a cutoff on validation predictions using only the supplied error costs."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    positives_below = np.r_[0, np.cumsum(y_sorted)]
    thresholds = np.r_[0.0, np.unique(p_sorted), np.nextafter(p_sorted[-1], np.inf)]
    split = np.searchsorted(p_sorted, thresholds, side="left")
    fn = positives_below[split]
    fp = (len(y) - split) - (int(y.sum()) - fn)
    costs = fp * false_positive_cost + fn * false_negative_cost
    curve = pd.DataFrame({"threshold": thresholds, "false_positives": fp,
                          "false_negatives": fn, "cost": costs})
    best = curve.loc[curve.cost.idxmin()]
    return float(best.threshold), curve


def choose_f1_threshold(y_true, probability):
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


# ----------------------------------------------------------------------------- верхняя навигация
NAV = ["Обзор", "Churn", "Freedom Score", "Propensity", "Next Best Action", "Каналы привлечения"]
head_logo, head_nav, head_stamp = st.columns([1.35, 6.2, 1.45])
with head_logo:
    st.markdown('<div class="fp-brand"><span class="fp-shield">F</span> FREEDOM</div>', unsafe_allow_html=True)
with head_nav:
    page = st.radio("Раздел", NAV, horizontal=True, label_visibility="collapsed", key="fp_nav")
with head_stamp:
    st.markdown('<div class="fp-stamp">АНАЛИТИКА&nbsp; · &nbsp;SAPP</div>', unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:#edf0f2;margin:3px 0 14px"></div>', unsafe_allow_html=True)

# ============================================================================= ОБЗОР
if page == "Обзор":
    st.markdown("""
    <section class="fp-hero">
      <div>
        <div class="fp-eyebrow">единый профиль клиента · аналитика экосистемы</div>
        <h1>FreedomProfile</h1>
        <div class="fp-gradient-text">От данных — к решениям</div>
        <p class="fp-hero-copy">Персонализация, удержание и рост на основе поведения клиентов. Все ключевые сигналы экосистемы — в одном профиле.</p>
        <div class="fp-pills"><span class="fp-pill">Персонализация</span><span class="fp-pill">Удержание</span><span class="fp-pill">Рост продуктов</span></div>
      </div>
      <div class="fp-art" aria-hidden="true">
        <div class="fp-glow"></div>
        <div class="fp-float fp-float-one">Next Best Action</div>
        <div class="fp-float fp-float-two">клиентский профиль</div>
        <div class="fp-float fp-float-three">Churn&nbsp; · &nbsp;Value</div>
        <div class="fp-phone"><div class="fp-notch"></div><div class="fp-score-label">Freedom Score</div>
          <div class="fp-phone-mark">F</div><div class="fp-score-value">84</div><div class="fp-score-track"></div>
          <div class="fp-mini-bars"><i style="height:35%"></i><i style="height:58%"></i><i style="height:43%"></i><i style="height:77%"></i><i style="height:92%"></i><i style="height:67%"></i><i style="height:100%"></i></div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    nba = load_nba()
    base = nba.p_churn.notna()
    source_profile, quality_profile, user_profile, source_overlap = load_eda_profile()

    st.markdown("#### Состав таблиц")
    st.dataframe(source_profile, **W, hide_index=True)

    st.markdown("#### Качество и ограничения данных")
    if not quality_profile.empty:
            st.dataframe(quality_profile, **W, hide_index=True)
    else:
        st.info("Не найдены cache parquet-файлы для EDA. Сначала сформируйте data/cache из ноутбука 02_eda.ipynb.")

    st.subheader("Охват модели")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Клиентов", f"{len(nba):,}".replace(",", " "))
    c2.metric("В базе churn (были операции за 90 дней)", f"{int(base.sum()):,}".replace(",", " "))
    c3.metric("Доля сегмента High", f"{(nba.segment == 'High').mean():.1%}")
    c4.metric("Средний риск ухода (в базе)", f"{nba.loc[base, 'p_churn'].mean():.1%}")

    st.subheader("Что получает каждый клиент (Next Best Action)")
    cnt = nba.action.value_counts().sort_index().rename_axis("action").reset_index(name="clients")
    fig = px.bar(cnt, x="clients", y="action", orientation="h", text="clients")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, **W)
    st.info("цифры показывают, как модели ранжируют клиентов по прошлым данным")

# ============================================================================= CHURN
elif page == "Churn":
    st.title("Churn: кто уйдёт в ближайшие 30 дней")
    st.caption("Клиент считается ушедшим, если после 12 апреля у него нет ни одной успешной операции за 30 дней. ")
    pred = load_churn((DIR / "churn_predictions_test.parquet").stat().st_mtime)
    y = pred.y_churn.values.astype(int)
    p_col = "churn_prob_lgbm_calibrated" if "churn_prob_lgbm_calibrated" in pred.columns else "churn_prob_lgbm"
    p = pred[p_col].values.astype(float)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Клиентов в тесте", f"{len(y):,}".replace(",", " "))
    c2.metric("Доля ушедших", f"{y.mean():.1%}")
    c3.metric("AUC-ROC", f"{roc_auc_score(y, p):.3f}")
    c4.metric("PR-AUC", f"{average_precision_score(y, p):.3f}")

    coverage_path = DIR / "churn_data_coverage.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path).iloc[0]
        st.caption(f"Проверка покрытия: транзакции {coverage['первая_дата']} — {coverage['последняя_дата']}; "
                   f"метка использует горизонт {int(coverage['горизонт_метки_дней'])} дней. "
                   "Для сравнения с 60-дневным churn данных после даты среза недостаточно.")
        status_path = DIR / "churn_status_audit.csv"
        if status_path.exists():
            with st.expander("Проверка статусов успешных операций"):
                st.dataframe(pd.read_csv(status_path).round(3), **W, hide_index=True)

    st.subheader("Выбор порога")
    st.caption("Сравните два способа выбрать порог. Порог подбирается на отдельной validation-выборке, качество оценивается на отложенном test.")
    validation_path = DIR / "churn_threshold_validation.parquet"
    if validation_path.exists():
        val_pred = duckdb.sql(f"SELECT * FROM read_parquet('{pq('churn_threshold_validation.parquet')}')").df()
        p_val = val_pred.churn_prob_calibrated.to_numpy(dtype=float)
        threshold_mode = st.radio("Метод выбора порога", ["Стоимость ошибок", "Максимум F1"],
                                  horizontal=True, key="churn_threshold_mode")
        cost_curve = None
        if threshold_mode == "Стоимость ошибок":
            v1, v2 = st.columns(2)
            false_positive_cost = v1.slider("Лишний контакт (условная стоимость)", 1, 100, 10)
            false_negative_cost = v2.slider("Пропущенный уход (условная стоимость)", 1, 100, 50)
            thr, cost_curve = choose_cost_threshold(val_pred.y_churn, p_val,
                                                    false_positive_cost, false_negative_cost)
            st.metric("Порог, выбранный на validation", f"{thr:.3f}",
                      f"стоимости: контакт {false_positive_cost} · пропуск {false_negative_cost}", delta_color="off")
        else:
            thr, f1_val = choose_f1_threshold(val_pred.y_churn, p_val)
            st.metric("Порог по максимуму F1 на validation", f"{thr:.3f}",
                      f"F1 на validation: {f1_val:.3f}", delta_color="off")
    else:
        st.warning("Сначала перезапустите 04_churn_model.ipynb: для подбора порога нужен отдельный validation-файл.")
        thr = st.slider("Порог вероятности ухода", 0.05, 0.95, 0.38, 0.01)

    sel = p >= thr
    tp = int((sel & (y == 1)).sum())
    fp = int((sel & (y == 0)).sum())
    fn = int((~sel & (y == 1)).sum())
    prec = tp / max(int(sel.sum()), 1)
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{prec:.1%}", help="Из названных рискованными, какая доля действительно ушла")
    m2.metric("Recall", f"{rec:.1%}", help="Из всех ушедших, какую долю нашли")
    m3.metric("F1", f"{f1:.3f}")
    m4.metric("Размер списка", f"{int(sel.sum()):,}".replace(",", " "), f"{sel.mean():.0%} клиентов")
    if validation_path.exists() and threshold_mode == "Стоимость ошибок":
        test_cost = fp * false_positive_cost + fn * false_negative_cost
        m5.metric("Стоимость ошибок на test", f"{test_cost:,.0f}".replace(",", " "), "условные единицы")
    elif validation_path.exists():
        m5.metric("Способ выбора порога", "F1")
    else:
        m5.metric("Стоимость ошибок на test", "—")

    pr, rc, th = precision_recall_curve(y, p)
    step = max(1, len(th) // 600)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th[::step], y=pr[:-1][::step], name="Precision"))
    fig.add_trace(go.Scatter(x=th[::step], y=rc[:-1][::step], name="Recall"))
    fig.add_vline(x=thr, line_dash="dash", line_color="gray")
    fig.update_layout(height=340, xaxis_title="порог", yaxis_title="значение", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, **W)

    if validation_path.exists() and threshold_mode == "Стоимость ошибок":
        curve_plot = cost_curve.sort_values("threshold")
        if len(curve_plot) > 800:
            curve_plot = curve_plot.iloc[np.linspace(0, len(curve_plot) - 1, 800).astype(int)]
        fig_cost = px.line(curve_plot, x="threshold", y="cost", labels={"threshold": "порог", "cost": "стоимость ошибок"})
        fig_cost.add_vline(x=thr, line_dash="dash", line_color="#08c96c", annotation_text="минимум на validation")
        fig_cost.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cost, **W)

    st.subheader("Список с учётом бюджета удержания")
    budget_share = st.slider("Какую долю клиентов можем обработать", 5, 50, 20, 1, format="%d%%") / 100
    k = max(1, int(len(y) * budget_share))
    top_idx = np.argsort(-p)[:k]
    found = int(y[top_idx].sum())
    precision_k = found / k
    recall_k = found / max(int(y.sum()), 1)
    lift_k = precision_k / y.mean() if y.mean() else 0
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Клиентов в списке", f"{k:,}".replace(",", " "))
    b2.metric("Churn-клиентов найдено", f"{found:,}".replace(",", " "))
    b3.metric("Precision списка", f"{precision_k:.1%}")
    b4.metric("Доля всех churn / lift", f"{recall_k:.1%} / {lift_k:.2f}×")

    capacity_path = DIR / "churn_capacity_metrics.csv"
    if capacity_path.exists():
        st.subheader("Сравнение охвата на тестовой выборке")
        capacity = pd.read_csv(capacity_path)
        st.caption("Доверительные интервалы рассчитаны bootstrap-ом. Выше lift — больше найденных churn-клиентов на тот же объём контактов.")
        st.dataframe(capacity.round(3), **W, hide_index=True)

    calibration_metrics_path = DIR / "churn_calibration_metrics.csv"
    calibration_path = DIR / "churn_calibration.csv"
    if calibration_metrics_path.exists() and calibration_path.exists():
        st.subheader("Калибровка вероятностей")
        st.caption("Калибратор обучен на validation. Тестовая кривая показывает, соответствует ли прогнозируемый риск фактической доле churn.")
        st.dataframe(pd.read_csv(calibration_metrics_path).round(5), **W, hide_index=True)
        calibration = pd.read_csv(calibration_path)
        fig_cal = px.line(calibration, x="predicted_probability", y="observed_rate", markers=True,
                          hover_data=["customers"], labels={"predicted_probability": "предсказанная вероятность",
                                                              "observed_rate": "фактическая доля churn"})
        upper = max(calibration.predicted_probability.max(), calibration.observed_rate.max())
        fig_cal.add_shape(type="line", x0=0, y0=0, x1=upper, y1=upper, line=dict(color="gray", dash="dash"))
        fig_cal.update_layout(height=320, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cal, **W)

    cohort_path = DIR / "churn_cohort_metrics.csv"
    if cohort_path.exists():
        st.subheader("Устойчивость по когортам регистрации")
        st.caption("Это проверка групп на одном тестовом срезе; она не заменяет тест на более позднем периоде.")
        cohorts = pd.read_csv(cohort_path)
        cohort_col = next(c for c in ("cohort", "когорта регистрации", "когорта_регистрации") if c in cohorts.columns)
        clients_col = next(c for c in ("customers", "клиентов") if c in cohorts.columns)
        churn_col = next(c for c in ("churn_rate", "доля churn", "доля_churn") if c in cohorts.columns)
        if "mean_predicted_risk" in cohorts.columns:
            cohort_lines = cohorts.melt(id_vars=cohort_col, value_vars=[churn_col, "mean_predicted_risk"],
                                        var_name="показатель", value_name="риск")
            fig_cohort = px.line(cohort_lines, x=cohort_col, y="риск", color="показатель", markers=True)
            fig_cohort.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="когорта регистрации",
                                     margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cohort, **W)
        display_cohorts = cohorts.rename(columns={cohort_col: "когорта регистрации", clients_col: "клиентов",
                                                   churn_col: "фактический churn"})
        st.dataframe(display_cohorts.round(3), **W, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Lift по децилям риска")
        d = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
        d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11))
        t = d.groupby("decile", observed=True).agg(clients=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"))
        t["captured"] = t.churners.cumsum() / t.churners.sum()
        t["lift"] = t.churn_rate / d.y.mean()
        fig2 = px.bar(t.reset_index(), x="decile", y="churn_rate", labels={"decile": "дециль (1 = самый рискованный)", "churn_rate": "доля ушедших"})
        fig2.add_hline(y=y.mean(), line_dash="dash", line_color="gray")
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, **W)
        st.dataframe(t.round(3), **W)
    with right:
        st.subheader("Что влияет на риск ухода")
        imp = importance("churn_feature_importance.csv")
        fig3 = px.bar(imp, x=imp.values, y=imp.index, orientation="h", labels={"x": "вклад (доля gain)", "y": ""})
        fig3.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, **W)
        st.caption("Главные признаки — недавняя активность: число операций за 30 дней, дней с последней операции, число активных дней.")

# ============================================================================= FREEDOM SCORE
elif page == "Freedom Score":
    st.title("Freedom Score: ценность клиента (0-100)")
    st.caption("Одноэтапная модель сразу прогнозирует сумму трат. Двухэтапная оценивает вероятность трат и сумму среди потративших; итоговый прогноз = вероятность × сумма.")
    comparison_path = DIR / "freedom_score_model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Сравнение моделей на отложенной выборке")
        st.dataframe(pd.read_csv(comparison_path).round(3), **W, hide_index=True)
    fs = load_fs()
    g = fs.groupby("segment").agg(clients=("customer_id", "size"), value=("y_value_next", "sum"),
                                  avg_spend=("y_value_next", "mean"), churn=("y_churn", "mean"),
                                  avg_score=("freedom_score", "mean")).reindex(SEG_ORDER)
    g["Доля клиентов"] = g.clients / g.clients.sum()
    g["Доля будущей ценности"] = g.value / g.value.sum()

    cols = st.columns(3)
    for col, seg in zip(cols, SEG_ORDER):
        col.metric(f"{seg}", f"{g.loc[seg, 'Доля клиентов']:.1%} клиентов",
                   f"{g.loc[seg, 'Доля будущей ценности']:.1%} ценности", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Доля клиентов и доля ценности")
        st.caption("Доля будущей ценности сегмента = сумма будущих трат клиентов сегмента / сумма будущих трат всех клиентов")
        st.caption("Сумма успешных покупок по карте за 30 дней после 12 апреля")
        long = g[["Доля клиентов", "Доля будущей ценности"]].reset_index().melt(id_vars="segment", var_name="показатель", value_name="доля")
        fig = px.bar(long, x="segment", y="доля", color="показатель", barmode="group", category_orders={"segment": SEG_ORDER})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
        st.plotly_chart(fig, **W)
    with right:
        st.subheader("Распределение баллов")
        fig2 = px.histogram(fs.sample(min(100_000, len(fs)), random_state=1), x="freedom_score", color="segment", nbins=50,
                            category_orders={"segment": SEG_ORDER})
        fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), bargap=0.02)
        st.plotly_chart(fig2, **W)

    st.subheader("Профиль сегментов")
    view = g[["clients", "Доля клиентов", "Доля будущей ценности", "avg_score", "avg_spend", "churn"]].rename(
        columns={"clients": "клиентов", "avg_score": "средний балл", "avg_spend": "средние будущие траты", "churn": "churn (только база churn)"})
    st.dataframe(view.round(3), **W)

    st.subheader("Что влияет на балл")
    imp = importance("freedom_score_importance.csv")
    fig3 = px.bar(imp, x=imp.values, y=imp.index, orientation="h", labels={"x": "вклад (доля gain)", "y": ""})
    fig3.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, **W)

# ============================================================================= PROPENSITY
elif page == "Propensity":
    st.title("Propensity: кому какой продукт предложить")
    st.caption("Вероятность первой покупки продукта за 30 дней среди клиентов, которые раньше его не покупали (cross-sell). ")
    sm = load_csv("propensity_summary.csv")
    show_product = pd.Series(True, index=sm.index)
    if "key" in sm.columns:
        show_product &= sm["key"] != "open_card"
    if "продукт" in sm.columns:
        show_product &= sm["продукт"] != "Открытие карты"
    sm = sm[show_product].copy()
    left, right = st.columns(2)
    with left:
        fig = px.bar(sm, x="продукт", y="lift топ-10%", text=sm["lift топ-10%"].round(1))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="Во сколько раз больше покупателей в топ-10% клиентов")
        st.plotly_chart(fig, **W)
    with right:
        fig = px.bar(sm, x="продукт", y="AUC", text=sm["AUC"].round(2))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="AUC по продуктам", yaxis_range=[0.5, 1])
        st.plotly_chart(fig, **W)
    st.dataframe(sm.round(3), **W, hide_index=True)
    st.subheader("Кому предложить: выберите продукт")
    names = {"flights": "Авиабилеты", "ticketon": "Ticketon", "groceries": "Продукты", "train_tickets": "Ж/д билеты",
             "open_deposit": "Открытие депозита"}
    key = st.selectbox("Продукт", list(names), format_func=lambda k: names[k])
    calibration_path = DIR / "propensity_calibration.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        calibration = calibration[calibration.key == key]
        if not calibration.empty:
            st.subheader("Калибровка по децилям тестовой выборки")
            st.caption("Точки ближе к диагонали означают, что предсказанная конверсия ближе к фактической.")
            fig_cal = px.line(calibration, x="предсказанная_конверсия", y="фактическая_конверсия",
                              markers=True, hover_data=["клиентов"],
                              labels={"предсказанная_конверсия": "предсказанная",
                                      "фактическая_конверсия": "фактическая"})
            upper = max(calibration.предсказанная_конверсия.max(), calibration.фактическая_конверсия.max())
            fig_cal.add_shape(type="line", x0=0, y0=0, x1=upper, y1=upper,
                              line=dict(color="gray", dash="dash"))
            fig_cal.update_layout(height=320, xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                                  margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cal, **W)
    reach_path = DIR / "propensity_reach_metrics.csv"
    if reach_path.exists():
        reach = pd.read_csv(reach_path)
        reach_product = reach[reach.key == key]
        if not reach_product.empty:
            reach_product = reach_product.sort_values("охват").copy()
            reach_product["охват_label"] = reach_product["охват"].map(lambda value: f"{value:.0%}")
            st.subheader("Конверсия на тесте при разном охвате")
            fig_reach = px.bar(reach_product, x="охват_label", y="конверсия в топе", text_auto=".1%",
                               category_orders={"охват_label": reach_product["охват_label"].tolist()},
                               labels={"охват_label": "охват списка", "конверсия в топе": "конверсия"})
            fig_reach.add_hline(y=reach_product["базовая конверсия"].iloc[0], line_dash="dash",
                                line_color="gray", annotation_text="средняя конверсия")
            fig_reach.update_layout(height=300, yaxis_tickformat=".1%", xaxis_type="category",
                                    margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_reach, **W)
            st.dataframe(reach_product[["охват", "контактируем", "покупок в тесте", "конверсия в топе",
                                        "базовая конверсия", "lift", "recall"]].round(3), **W, hide_index=True)
    top = st.slider("Охват: доля клиентов с самой высокой вероятностью", 1, 50, 15, 1, format="%d%%")
    sc = duckdb.sql(f"""SELECT p_{key} AS p FROM read_parquet('{pq('propensity_scores.parquet')}')
                        WHERE elig_{key} = 1""").df().p
    cut = float(np.quantile(sc, 1 - top / 100))
    a, b, c = st.columns(3)
    a.metric("Клиентов-кандидатов", f"{len(sc):,}".replace(",", " "))
    b.metric("Клиентов в охвате", f"{int((sc >= cut).sum()):,}".replace(",", " "))
    c.metric("Минимальная вероятность в охвате", f"{cut:.1%}", f"среднее по кандидатам: {sc.mean():.1%}", delta_color="off")
    fig = px.histogram(sc, x="p", nbins=60)
    fig.add_vline(x=cut, line_dash="dash", line_color="red")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="вероятность покупки", showlegend=False)
    st.plotly_chart(fig, **W)

# ============================================================================= NBA
elif page == "Next Best Action":
    st.title("Next Best Action: что делать с каждым клиентом")
    st.caption("Правила поверх трёх моделей: риск ухода (churn), ценность (Freedom Score) и вероятность покупки (propensity).")
    nba = load_nba()
    g = nba.groupby("action").agg(clients=("customer_id", "size"), avg_score=("freedom_score", "mean"),
                                  avg_churn=("p_churn", "mean")).sort_index()
    g["доля"] = g.clients / g.clients.sum()
    g["что делаем"] = [ACTION_HINTS.get(a, "") for a in g.index]
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(g.reset_index(), x="clients", y="action", orientation="h", text="clients")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, **W)
    with right:
        cs = nba[nba.action == "4. Cross-sell"].product_name.value_counts().rename_axis("продукт").reset_index(name="клиентов")
        fig2 = px.pie(cs, names="продукт", values="клиентов", hole=0.45, title="Какие продукты предлагаем (cross-sell)")
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, **W)
    st.dataframe(g.rename(columns={"clients": "клиентов", "avg_score": "средний Freedom Score", "avg_churn": "средний риск ухода"}).round(3),
                 **W)

    st.subheader("Список клиентов по действию")
    act = st.selectbox("Действие", sorted(nba.action.unique()))
    sub = nba[nba.action == act].sort_values(["freedom_score"], ascending=False).head(200)
    st.dataframe(sub, **W, hide_index=True)
    st.download_button("Скачать список (CSV)", sub.to_csv(index=False).encode("utf-8"), file_name="nba_list.csv", mime="text/csv")

    st.subheader("Карточка клиента")
    st.caption("Выберите ID из показанного списка действий или введите любой customer_id вручную.")
    visible_ids = [""] + sub.customer_id.astype("int64").astype(str).tolist()
    selected_id = st.selectbox("Клиент из списка выше", visible_ids,
                               format_func=lambda value: "Выберите клиента…" if value == "" else value,
                               key="nba_customer_select")
    typed_id = st.text_input("Или введите customer_id", placeholder="Например, 1234567", key="nba_customer_id")
    cid_text = typed_id.strip() or selected_id
    if cid_text.strip():
        if not cid_text.strip().isdigit():
            st.warning("Введите customer_id целым числом.")
        else:
            cid = int(cid_text.strip())
            row = nba[nba.customer_id == cid]
            feature_row = load_customer_features(cid)
            if row.empty or feature_row.empty:
                st.warning("Клиент с таким customer_id не найден в витрине.")
            else:
                r = row.iloc[0]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Рекомендованное действие", r.action.split(". ", 1)[1])
                k2.metric("Freedom Score", f"{r.freedom_score:.0f}", r.segment, delta_color="off")
                k3.metric("Риск ухода", "—" if pd.isna(r.p_churn) else f"{r.p_churn:.1%}")
                k4.metric("Продукт для предложения", r.product_name if isinstance(r.product_name, str) else "—")

                pr_result = duckdb.sql(
                    f"SELECT * FROM read_parquet('{pq('propensity_scores.parquet')}') WHERE customer_id = {cid}"
                ).df()
                if not pr_result.empty:
                    pr = pr_result.iloc[0]
                    product_names = {"flights": "Авиабилеты", "ticketon": "Ticketon", "groceries": "Продукты",
                                     "train_tickets": "Ж/д билеты", "open_deposit": "Открытие депозита"}
                    rows = [{"Продукт": name, "Вероятность покупки": pr[f"p_{key}"],
                             "Статус предложения": "Можно предложить" if bool(pr[f"elig_{key}"]) else "Уже есть продукт"}
                            for key, name in product_names.items()]
                    propensity_view = pd.DataFrame(rows)
                    st.markdown("#### Склонность к покупке продуктов")
                    fig = px.bar(propensity_view, x="Вероятность покупки", y="Продукт", orientation="h",
                                 color="Статус предложения", text_auto=".1%",
                                 color_discrete_map={"Можно предложить": "#0875D1", "Уже есть продукт": "#AFCFEA"})
                    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                      yaxis=dict(autorange="reversed"), xaxis_tickformat=".1%")
                    st.plotly_chart(fig, **W)

                labels = {
                    "customer_id": "ID клиента", "city": "Город", "gender": "Пол", "age": "Возраст",
                    "reg_date": "Дата регистрации", "days_since_reg": "Дней с регистрации",
                    "profile_conflict": "Профиль содержит конфликтующие значения", "n_profile_rows": "Строк профиля",
                    "has_tx": "Есть успешная транзакция", "tx_n_all": "Транзакций за всю историю",
                    "tx_n_30d": "Транзакций за 30 дней", "tx_n_90d": "Транзакций за 90 дней",
                    "tx_n_base": "Транзакций за базовое окно", "tx_n_prev30d": "Транзакций за предыдущие 30 дней",
                    "tx_trend_30d": "Изменение активности за 30 дней", "spend_all": "Сумма покупок за всю историю",
                    "spend_30d": "Сумма покупок за 30 дней", "spend_90d": "Сумма покупок за 90 дней",
                    "purchase_avg": "Средняя сумма покупки", "purchase_max": "Максимальная сумма покупки",
                    "purchase_std": "Разброс сумм покупок", "active_days": "Дней с активностью",
                    "n_mcc": "Категорий торговых точек", "recency_days": "Дней с последней транзакции",
                    "tx_history_days": "Дней между первой транзакцией и срезом", "fail_rate": "Доля неуспешных операций",
                    "fails_30d": "Неуспешных операций за 30 дней", "n_returns": "Возвратов",
                    "pp_n_all": "Покупок у партнёров за всю историю", "pp_n_90d": "Покупок у партнёров за 90 дней",
                    "pp_spend_all": "Сумма покупок у партнёров", "pp_cashback_all": "Сумма cashback у партнёров",
                    "pp_n_apps": "Число партнёрских сервисов", "pp_recency_days": "Дней с последней покупки у партнёра",
                    "proc_n_all": "Процессов за всю историю", "proc_n_30d": "Процессов за 30 дней",
                    "proc_n_types": "Типов процессов", "proc_completion_rate": "Доля завершённых процессов",
                    "proc_error_rate": "Доля процессов с ошибкой", "proc_decline_rate": "Доля отклонённых процессов",
                    "proc_avg_duration_s": "Средняя длительность процесса, секунд",
                    "proc_recency_days": "Дней с последнего процесса", "acq_n_sources": "Число источников привлечения",
                }

                def feature_group(name):
                    if name in {"city", "gender", "age", "reg_date", "days_since_reg", "n_profile_rows", "profile_conflict"}:
                        return "Профиль"
                    if name == "has_tx" or name.startswith(("tx_", "spend_", "purchase_", "share_", "active_", "fail", "n_mcc", "n_returns", "recency_days")):
                        return "Транзакции"
                    if name.startswith("pp_"):
                        return "Партнёры"
                    if name.startswith(("proc_", "has_", "rating_")):
                        return "Продукты и процессы"
                    if name.startswith("acq_"):
                        return "Привлечение"
                    return "Другие признаки"

                def feature_value(value, name):
                    if pd.isna(value):
                        return "Нет данных"
                    if isinstance(value, (bool, np.bool_)):
                        return "Да" if value else "Нет"
                    if name.startswith(("has_", "acq_", "rating_")) and value in (0, 1):
                        return "Да" if value else "Нет"
                    if isinstance(value, (float, np.floating)):
                        if name.endswith(("_rate",)) or name.startswith("share_"):
                            return f"{value:.1%}"
                        return f"{value:,.3f}".replace(",", " ")
                    return str(value)

                feature_values = feature_row.iloc[0]
                detail_rows = []
                for name, value in feature_values.items():
                    if name == "customer_id" or name.startswith("y_"):
                        continue
                    shown_name = labels.get(name, name.replace("_", " "))
                    if name.startswith("pp_") and name.endswith("_n"):
                        shown_name = "Покупок у партнёра: " + name[3:-2].replace("_", " ")
                    elif name.startswith("acq_") and name != "acq_n_sources":
                        shown_name = "Источник привлечения: " + name[4:].replace("_", " ")
                    detail_rows.append({"Группа": feature_group(name), "Показатель": shown_name,
                                        "Поле витрины": name, "Значение": feature_value(value, name)})
                st.markdown("#### Профиль и признаки, использованные моделями")
                detail = pd.DataFrame(detail_rows).sort_values(["Группа", "Показатель"])
                st.dataframe(detail, **W, hide_index=True)
    else:
        st.info("Выберите customer_id из таблицы списка клиентов выше и вставьте его сюда.")

# ============================================================================= КАНАЛЫ
else:
    st.title("Каналы привлечения: кто приводит ценных клиентов")
    tab_adj, tab_combo, tab_basic = st.tabs(["При прочих равных", "Комбинации источников", "Первичный обзор"])

    # ---------------------------------------------------------------- вкладка 1: регрессия (ноутбук 08b)
    with tab_adj:
        reg_path, matrix_path = DIR / "acquisition_regression.csv", DIR / "acquisition_matrix.csv"
        if not (reg_path.exists() and matrix_path.exists()):
            st.info("Запустите ноутбук 08b_acquisition_regression.ipynb: он сохранит acquisition_*.csv в data/processed/.")
        else:
            reg, matrix = pd.read_csv(reg_path), pd.read_csv(matrix_path)
            st.subheader("Разница с каналом organic")
            titles = reg.drop_duplicates("исход").set_index("исход")["название"].to_dict()
            outcome = st.radio("Исход", list(titles), format_func=lambda k: titles[k], horizontal=True, key="acq_outcome")
            r = reg[(reg.исход == outcome) & (reg.канал != "organic")].copy()
            if outcome == "log_spend":          # разница логарифмов -> проценты
                est, lo, hi = [(np.exp(r[c]) - 1) * 100 for c in ("разница с organic", "ci_low", "ci_high")]
                unit = "%, размер трат среди купивших"
            else:                                # доли -> процентные пункты
                est, lo, hi = [r[c] * 100 for c in ("разница с organic", "ci_low", "ci_high")]
                unit = "п.п."
            r["est"], r["lo"], r["hi"] = est, lo, hi
            r = r.sort_values("est")
            colors = ["#2ca02c" if l > 0 else "#d62728" if h < 0 else "#9aa8b4" for l, h in zip(r.lo, r.hi)]
            fig = go.Figure(go.Scatter(
                x=r.est, y=r.канал, mode="markers", marker=dict(size=11, color=colors),
                error_x=dict(type="data", symmetric=False, array=r.hi - r.est, arrayminus=r.est - r.lo, color="#8a99a6"),
                customdata=np.c_[r["клиентов в канале"], r.lo, r.hi],
                hovertemplate="%{y}<br>разница: %{x:.1f}<br>интервал: %{customdata[1]:.1f} … %{customdata[2]:.1f}"
                              "<br>клиентов в канале: %{customdata[0]:,.0f}<extra></extra>"))
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=380, xaxis_title=f"разница с organic, {unit}", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, **W)
            st.subheader("Размер канала и качество клиентов")
            verdict_colors = {"лучше organic": "#2ca02c", "хуже organic": "#d62728", "неотличим / смешанно": "#9aa8b4"}
            fig2 = px.scatter(matrix, x="клиентов", y="активация, п.п.", text="канал", color="вывод", size="клиентов",
                              log_x=True, color_discrete_map=verdict_colors,
                              hover_data=["покупка, п.п.", "удержание, п.п.", "размер трат, %"])
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_traces(textposition="top center")
            fig2.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="разница с organic по активации, п.п.")
            st.plotly_chart(fig2, **W)
            st.dataframe(matrix.round(1), **W, hide_index=True)
    # ---------------------------------------------------------------- вкладка 2: комбинации
    with tab_combo:
        combo_path = DIR / "acquisition_combos.csv"
        if not combo_path.exists():
            st.info("Запустите ноутбук 08b_acquisition_regression.ipynb: он сохранит acquisition_combos.csv.")
        else:
            combos = pd.read_csv(combo_path).rename(columns={"set_label": "набор источников"})
            st.subheader("Активация в первые 30 дней по набору источников")
            st.caption("Самые частые наборы источников. Клиенты «organic + продукт» активнее, чем «только organic» и «только продукт».")
            combos = combos.sort_values("клиентов", ascending=False)
            fig3 = px.bar(combos, x="активированы", y="набор источников", orientation="h", text_auto=".0%",
                          hover_data=["клиентов", "купили", "удержание"])
            fig3.update_layout(yaxis=dict(autorange="reversed", title=""), height=460, xaxis_tickformat=".0%",
                               xaxis_title="доля активированных", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, **W)
            st.dataframe(combos.round(3), **W, hide_index=True)

    # ---------------------------------------------------------------- вкладка 3: прежний обзор (ноутбук 08)
    with tab_basic:
        cs = load_csv("channel_summary.csv", index_col=0)
        raw = load_csv("channel_raw_profile.csv", index_col=0)
        idx_cols = [c for c in cs.columns if c.startswith("индекс")]
        st.subheader("Индекс качества каналов (1 = средний клиент)")
        order = cs.sort_values("средний индекс", ascending=False).index
        fig = px.imshow(cs.loc[order, idx_cols], text_auto=".2f", color_continuous_scale="RdYlGn", color_continuous_midpoint=1, aspect="auto")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, **W)
        left, right = st.columns(2)
        with left:
            fig2 = px.bar(cs.loc[order].reset_index(), x="средний индекс", y="index", orientation="h", text=cs.loc[order, "средний индекс"].round(2))
            fig2.add_vline(x=1, line_color="black")
            fig2.update_layout(yaxis=dict(autorange="reversed", title=""), height=380, margin=dict(l=0, r=0, t=30, b=0), title="Средний индекс")
            st.plotly_chart(fig2, **W)
        with right:
            fig3 = px.scatter(cs.reset_index(), x="клиентов", y="индекс: траты", text="index", size="клиентов", log_x=True)
            fig3.add_hline(y=1, line_dash="dash", line_color="gray")
            fig3.update_traces(textposition="top center")
            fig3.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="Размер канала и ценность клиентов")
            st.plotly_chart(fig3, **W)
        st.subheader("Показатели по каналам")
        st.dataframe(raw.round(3), **W)
        st.warning("Сырое сравнение обманывает: клиенты с несколькими источниками и «старые» клиенты активнее сами по себе. "
                   "Выводы описательные («при прочих равных»), а не причинные.")
