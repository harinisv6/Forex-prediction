import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ta
import joblib
import os
import numpy as np
import streamlit.components.v1 as components

from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_autorefresh import st_autorefresh

# Safely import MetaTrader5 (Windows only; mocked or disabled on Linux cloud)
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="AI Forex Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================
# LOGIN GATE
# ====================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.query_params.get("logged_in") == "true":
    st.session_state.logged_in = True

if not st.session_state.logged_in:
    st.markdown("""
        <style>
        #MainMenu {visibility:hidden;}
        header {visibility:hidden;}
        footer {visibility:hidden;}
        section[data-testid="stSidebar"]{display:none;}
        .block-container{padding:0rem; max-width:100%;}
        iframe{width:100% !important;}
        </style>
    """, unsafe_allow_html=True)

    with open("templates/login.html", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=1000, scrolling=False)
    st.stop()

# ====================================
# SETTINGS
# ====================================
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
LOG_FILE = "prediction_log.csv"
MODEL_DIR = "models"

pair = st.sidebar.selectbox("Currency Pair", PAIRS)
refresh = st.sidebar.slider("Refresh every (sec)", 5, 60, 10)
candle_count = st.sidebar.slider("Candles to display", 50, 300, 100)

st_autorefresh(interval=refresh * 1000, key="autorefresh")


# ====================================
# MT5 — persistent backend connection
# ====================================
@st.cache_resource
def init_mt5():
    if not MT5_AVAILABLE:
        return False, "MetaTrader5 package not available on this OS (Linux cloud)."
    if not mt5.initialize():
        return False, str(mt5.last_error())
    return True, None


connected, mt5_error = init_mt5()

if not connected:
    st.warning(f"⚠️ MT5 connection status: {mt5_error}")
    st.info("Running in cloud/dashboard preview mode (Live MT5 data disabled).")


# ====================================
# LOAD MODELS (cached)
# ====================================
@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
    reg_1h = joblib.load(os.path.join(MODEL_DIR, "xgb_1h.pkl"))
    reg_1d = joblib.load(os.path.join(MODEL_DIR, "xgb_1d.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return clf, reg_1h, reg_1d, features


try:
    clf, reg_1h, reg_1d, FEATURES = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.info(f"Make sure all .pkl files are inside the '{MODEL_DIR}/' folder.")
    st.stop()


# ====================================
# FETCH LIVE DATA FROM MT5 (or fallback)
# ====================================
@st.cache_data(ttl=refresh)
def get_data(pair: str, count: int) -> pd.DataFrame:
    if MT5_AVAILABLE and connected:
        rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, count)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df["volume"] = df["tick_volume"] if "tick_volume" in df.columns else 0
            return df.reset_index(drop=True)

    # Fallback dummy data generator for cloud testing if MT5 is unreachable
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in range(count)]
    times.reverse()
    np.random.seed(42)
    base_price = 1.1000 if "EUR" in pair else (1.3000 if "GBP" in pair else (150.0 if "JPY" in pair else 0.6500))
    closes = base_price + np.cumsum(np.random.randn(count) * 0.0005)
    df = pd.DataFrame({
        "time": times,
        "open": closes + np.random.randn(count) * 0.0002,
        "high": closes + abs(np.random.randn(count) * 0.0004),
        "low": closes - abs(np.random.randn(count) * 0.0004),
        "close": closes,
        "volume": np.random.randint(100, 1000, count)
    })
    return df.reset_index(drop=True)


# ====================================
# FEATURE ENGINEERING
# ====================================
def create_features(df: pd.DataFrame, current_pair: str) -> pd.DataFrame:
    df = df.copy()

    df["open_lag1"] = df["open"].shift(1)
    df["high_lag1"] = df["high"].shift(1)
    df["low_lag1"] = df["low"].shift(1)
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)

    df["returns"] = df["close"].pct_change(fill_method=None)
    df["mom3"] = df["close"].pct_change(3, fill_method=None)
    df["mom5"] = df["close"].pct_change(5, fill_method=None)
    df["mom10"] = df["close"].pct_change(10, fill_method=None)

    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"]
    ).average_true_range()

    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["volatility"] = df["returns"].rolling(5).std()

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.dayofweek

    for p in PAIRS:
        df[f"pair_{p}"] = 1 if p == current_pair else 0

    return df.dropna().reset_index(drop=True)


# ====================================
# ALIGN FEATURES TO TRAINED MODEL
# ====================================
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES]


# ====================================
# PREDICT
# ====================================
def predict(raw: pd.DataFrame, feat_df: pd.DataFrame):
    X = align_features(feat_df.copy())
    row = X.iloc[[-1]]

    prob = float(clf.predict_proba(row)[0][1])
    ret_1h = float(reg_1h.predict(row)[0])
    ret_1d = float(reg_1d.predict(row)[0])

    current = float(raw["close"].iloc[-1])
    now_time = datetime.now()

    p1h = current * (1 + ret_1h)
    p1d = current * (1 + ret_1d)

    return current, p1h, p1d, prob, now_time


# ====================================
# PREDICTION LOG
# ====================================
def log_prediction(pair, current, p1h, p1d, prob, now_time):
    entry = {
        "timestamp": now_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pair": pair,
        "current": round(current, 5),
        "pred_1h": round(p1h, 5),
        "pred_1d": round(p1d, 5),
        "target_1h": (now_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
        "target_1d": (now_time + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
        "prob": round(prob, 4),
    }
    df_new = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df_log = pd.concat([pd.read_csv(LOG_FILE), df_new], ignore_index=True)
    else:
        df_log = df_new
    df_log.to_csv(LOG_FILE, index=False)


# ====================================
# ACCURACY METRICS
# ====================================
def compute_metrics():
    if not os.path.exists(LOG_FILE):
        return None

    df_log = pd.read_csv(LOG_FILE)
    df_log = df_log[df_log["pair"] == pair].copy()
    if df_log.empty:
        return None

    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], format="ISO8601")
    df_log = df_log.sort_values("timestamp").reset_index(drop=True)

    now = datetime.now()
    results = {}

    df_1h = df_log[df_log["timestamp"] < now - timedelta(hours=1)].copy()
    if len(df_1h) >= 5:
        df_1h["actual"] = df_1h["current"].shift(-1)
        df_1h = df_1h.dropna(subset=["actual"])
        if len(df_1h) >= 2:
            y_true = df_1h["actual"].values
            y_pred = df_1h["pred_1h"].values
            if len(y_true) > 0 and len(y_true) == len(y_pred):
                results["1H"] = {
                    "MAE": round(mean_absolute_error(y_true, y_pred), 6),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
                    "R2": round(float(r2_score(y_true, y_pred)), 4),
                    "MAPE": round(float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100, 4),
                    "n": len(df_1h),
                    "df": df_1h,
                }

    df_1d = df_log[df_log["timestamp"] < now - timedelta(hours=24)].copy()
    if len(df_1d) >= 5:
        df_1d["actual"] = df_1d["current"].shift(-24)
        df_1d = df_1d.dropna(subset=["actual"])
        if len(df_1d) >= 2:
            y_true = df_1d["actual"].values
            y_pred = df_1d["pred_1d"].values
            if len(y_true) > 0 and len(y_true) == len(y_pred):
                results["1D"] = {
                    "MAE": round(mean_absolute_error(y_true, y_pred), 6),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
                    "R2": round(float(r2_score(y_true, y_pred)), 4),
                    "MAPE": round(float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100, 4),
                    "n": len(df_1d),
                    "df": df_1d,
                }

    return results if results else None


# ====================================
# CHART BUILDER
# ====================================
def build_forecast_chart(
        raw, feat_df, current, forecast_price,
        horizon_label, target_time, now_time,
        line_color, atr_val, signal, sig_color, sig_symbol,
        chart_height=520,
):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=raw["time"],
        open=raw["open"],
        high=raw["high"],
        low=raw["low"],
        close=raw["close"],
        name=pair,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    fig.add_trace(go.Scatter(
        x=raw["time"], y=feat_df["ma5"],
        mode="lines", name="MA 5",
        line=dict(color="#FFA726", width=1.2),
    ))

    fig.add_trace(go.Scatter(
        x=raw["time"], y=feat_df["ma10"],
        mode="lines", name="MA 10",
        line=dict(color="#42A5F5", width=1.2),
    ))

    band_scale = 1.0 if "1H" in horizon_label else 2.0
    fig.add_trace(go.Scatter(
        x=[now_time, target_time],
        y=[current, forecast_price],
        mode="lines+markers+text",
        name=f"{horizon_label} Forecast",
        line=dict(color=line_color, width=2.5, dash="dot"),
        marker=dict(size=10, symbol="circle", color=line_color),
        text=[
            f"  {now_time.strftime('%H:%M')}  {current:.5f}",
            f"  {target_time.strftime('%d %b %H:%M')}  {forecast_price:.5f}",
        ],
        textposition="middle right",
        textfont=dict(size=11, color=line_color),
    ))

    r = int(line_color[1:3], 16)
    g = int(line_color[3:5], 16)
    b = int(line_color[5:7], 16)
    fig.add_trace(go.Scatter(
        x=[now_time, target_time, target_time, now_time],
        y=[
            current + atr_val * 0.5,
            forecast_price + atr_val * band_scale,
            forecast_price - atr_val * band_scale,
            current - atr_val * 0.5,
        ],
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(width=0),
        name="Confidence band",
        hoverinfo="skip",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=120, t=40, b=10),
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
    )
    return fig


# ====================================
# RUN PIPELINE
# ====================================
raw = get_data(pair, candle_count)
feat_df = create_features(raw, pair)
current, p1h, p1d, prob, now_time = predict(raw, feat_df)
log_prediction(pair, current, p1h, p1d, prob, now_time)

t1h = now_time + timedelta(hours=1)
t1d = now_time + timedelta(hours=24)
atr_val = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else current * 0.001

if prob > 0.65:
    signal, sig_color, sig_symbol = "BUY 📈", "#26a69a", "▲"
elif prob < 0.35:
    signal, sig_color, sig_symbol = "SELL 📉", "#ef5350", "▼"
else:
    signal, sig_color, sig_symbol = "NO TRADE ⚖️", "#9E9E9E", "■"

delta_1h = f"{((p1h - current) / current * 100):+.4f}%"
delta_1d = f"{((p1d - current) / current * 100):+.4f}%"

metrics = compute_metrics()

# UI Layout Rendering
st.markdown(
    f"**Now:** `{now_time.strftime('%d %b %Y  %H:%M:%S')}`  │  "
    f"**1H target:** `{t1h.strftime('%d %b %Y  %H:%M')}`  │  "
    f"**1D target:** `{t1d.strftime('%d %b %Y  %H:%M')}`"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"{current:.5f}")
col2.metric(f"1H Forecast", f"{p1h:.5f}", delta=delta_1h)
col3.metric(f"1D Forecast", f"{p1d:.5f}", delta=delta_1d)
col4.metric("Signal", signal, delta=f"confidence {prob:.1%}")
col5.metric("1H MAE", f"{metrics['1H']['MAE']:.5f}" if metrics and "1H" in metrics else "—")

st.divider()

# 1H Forecast Chart
st.subheader(f"🕐 1-Hour Forecast — Target: {t1h.strftime('%d %b %Y  %H:%M')}")
fig_1h = build_forecast_chart(raw, feat_df, current, p1h, "1H", t1h, now_time, "#00E5FF", atr_val, signal, sig_color,
                              sig_symbol)
st.plotly_chart(fig_1h, use_container_width=True)

st.divider()

# 1D Forecast Chart (Restored!)
st.subheader(f"📅 1-Day Forecast — Target: {t1d.strftime('%d %b %Y  %H:%M')}")
fig_1d = build_forecast_chart(raw, feat_df, current, p1d, "1D", t1d, now_time, "#FFD600", atr_val, signal, sig_color,
                              sig_symbol)
st.plotly_chart(fig_1d, use_container_width=True)

st.divider()

# RSI Chart
st.subheader("📉 RSI (14)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=raw["time"], y=feat_df["rsi"], mode="lines", line=dict(color="#42A5F5", width=1.5)))
fig_rsi.update_layout(template="plotly_dark", height=180, margin=dict(l=10, r=120, t=10, b=10), plot_bgcolor="#131722",
                      paper_bgcolor="#131722")
st.plotly_chart(fig_rsi, use_container_width=True)

# MACD Chart
st.subheader("📊 MACD")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=raw["time"], y=feat_df["macd"], mode="lines", line=dict(color="#CE93D8", width=1.5)))
fig_macd.update_layout(template="plotly_dark", height=160, margin=dict(l=10, r=120, t=10, b=10), plot_bgcolor="#131722",
                       paper_bgcolor="#131722")
st.plotly_chart(fig_macd, use_container_width=True)
