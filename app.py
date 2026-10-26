import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import MetaTrader5 as mt5
import ta
import joblib
import os
import numpy as np
import streamlit.components.v1 as components

from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_autorefresh import st_autorefresh

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
PAIRS        = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
LOG_FILE     = "prediction_log.csv"
MODEL_DIR    = "models"

pair         = st.sidebar.selectbox("Currency Pair", PAIRS)
refresh      = st.sidebar.slider("Refresh every (sec)", 5, 60, 10)
candle_count = st.sidebar.slider("Candles to display", 50, 300, 100)

st_autorefresh(interval=refresh * 1000, key="autorefresh")


# ====================================
# MT5 — persistent backend connection
# ====================================
@st.cache_resource
def init_mt5():
    if not mt5.initialize():
        return False, str(mt5.last_error())
    return True, None


connected, mt5_error = init_mt5()

if not connected:
    st.error(f"❌ MT5 connection failed: {mt5_error}")
    st.info("Make sure MetaTrader 5 is open and logged in.")
    st.stop()

account = mt5.account_info()
st.sidebar.success("✅ MT5 Connected")
st.sidebar.caption(f"Login : {account.login}")
st.sidebar.caption(f"Server: {account.server}")
st.sidebar.caption(f"Balance: {account.balance:.2f} {account.currency}")


# ====================================
# LOAD MODELS (cached)
# ====================================
@st.cache_resource
def load_models():
    clf      = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
    reg_1h   = joblib.load(os.path.join(MODEL_DIR, "xgb_1h.pkl"))
    reg_1d   = joblib.load(os.path.join(MODEL_DIR, "xgb_1d.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return clf, reg_1h, reg_1d, features


try:
    clf, reg_1h, reg_1d, FEATURES = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.info(f"Make sure all .pkl files are inside the '{MODEL_DIR}/' folder.")
    st.stop()


# ====================================
# FETCH LIVE DATA FROM MT5
# ====================================
@st.cache_data(ttl=refresh)
def get_data(pair: str, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, count)
    if rates is None or len(rates) == 0:
        st.error(f"No data for {pair}: {mt5.last_error()}")
        st.stop()
    df = pd.DataFrame(rates)
    df["time"]   = pd.to_datetime(df["time"], unit="s")
    df["volume"] = df["tick_volume"] if "tick_volume" in df.columns else 0
    return df.reset_index(drop=True)


# ====================================
# FEATURE ENGINEERING
# ====================================
def create_features(df: pd.DataFrame, current_pair: str) -> pd.DataFrame:
    df = df.copy()

    df["open_lag1"]  = df["open"].shift(1)
    df["high_lag1"]  = df["high"].shift(1)
    df["low_lag1"]   = df["low"].shift(1)
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)

    df["returns"]    = df["close"].pct_change(fill_method=None)
    df["mom3"]       = df["close"].pct_change(3,  fill_method=None)
    df["mom5"]       = df["close"].pct_change(5,  fill_method=None)
    df["mom10"]      = df["close"].pct_change(10, fill_method=None)

    df["rsi"]  = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["atr"]  = ta.volatility.AverageTrueRange(
                     df["high"], df["low"], df["close"]
                 ).average_true_range()

    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"]   = bb.bollinger_hband() - bb.bollinger_lband()
    df["ma5"]        = df["close"].rolling(5).mean()
    df["ma10"]       = df["close"].rolling(10).mean()
    df["volatility"] = df["returns"].rolling(5).std()

    df["hour"] = df["time"].dt.hour
    df["day"]  = df["time"].dt.dayofweek

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
    X   = align_features(feat_df.copy())
    row = X.iloc[[-1]]

    prob   = float(clf.predict_proba(row)[0][1])
    ret_1h = float(reg_1h.predict(row)[0])
    ret_1d = float(reg_1d.predict(row)[0])

    current  = float(raw["close"].iloc[-1])   # real last close, not lagged
    now_time = datetime.now()

    p1h = current * (1 + ret_1h)
    p1d = current * (1 + ret_1d)

    return current, p1h, p1d, prob, now_time


# ====================================
# PREDICTION LOG
# ====================================
def log_prediction(pair, current, p1h, p1d, prob, now_time):
    entry = {
        "timestamp":  now_time.strftime("%Y-%m-%dT%H:%M:%S"),  # no microseconds → always consistent
        "pair":       pair,
        "current":    round(current, 5),
        "pred_1h":    round(p1h, 5),
        "pred_1d":    round(p1d, 5),
        "target_1h":  (now_time + timedelta(hours=1) ).strftime("%Y-%m-%d %H:%M"),
        "target_1d":  (now_time + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
        "prob":       round(prob, 4),
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
    """
    Compares past predictions to the next observed price.
    Returns None (never crashes) if not enough data yet.
    """
    if not os.path.exists(LOG_FILE):
        return None

    df_log = pd.read_csv(LOG_FILE)

    df_log = df_log[df_log["pair"] == pair].copy()
    if df_log.empty:
        return None

    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], format="ISO8601")
    df_log = df_log.sort_values("timestamp").reset_index(drop=True)

    now     = datetime.now()
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
                    "MAE":  round(mean_absolute_error(y_true, y_pred), 6),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
                    "R2":   round(float(r2_score(y_true, y_pred)), 4),
                    "MAPE": round(float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100, 4),
                    "n":    len(df_1h),
                    "df":   df_1h,
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
                    "MAE":  round(mean_absolute_error(y_true, y_pred), 6),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
                    "R2":   round(float(r2_score(y_true, y_pred)), 4),
                    "MAPE": round(float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100, 4),
                    "n":    len(df_1d),
                    "df":   df_1d,
                }

    return results if results else None


# ====================================
# CHART BUILDER — reused for 1H & 1D
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
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
        hovertemplate=(
            "<b>%{x|%d %b %Y %H:%M}</b><br>"
            "O: %{open:.5f}  H: %{high:.5f}<br>"
            "L: %{low:.5f}   C: %{close:.5f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=raw["time"], y=feat_df["ma5"],
        mode="lines", name="MA 5",
        line=dict(color="#FFA726", width=1.2),
        hovertemplate="%{x|%d %b %H:%M}  MA5: %{y:.5f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=raw["time"], y=feat_df["ma10"],
        mode="lines", name="MA 10",
        line=dict(color="#42A5F5", width=1.2),
        hovertemplate="%{x|%d %b %H:%M}  MA10: %{y:.5f}<extra></extra>",
    ))

    band_scale = 1.0 if "1H" in horizon_label else 2.0
    fig.add_trace(go.Scatter(
        x=[now_time, target_time],
        y=[current,  forecast_price],
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
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Forecast: %{y:.5f}<extra></extra>",
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

    fig.add_hline(
        y=current,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        line_width=1,
        annotation_text=f"  NOW  {current:.5f}",
        annotation_position="right",
        annotation_font=dict(size=11, color="white"),
    )

    fig.add_hline(
        y=forecast_price,
        line_dash="dot",
        line_color=line_color,
        line_width=1,
        annotation_text=f"  {horizon_label} Target  {forecast_price:.5f}",
        annotation_position="right",
        annotation_font=dict(size=11, color=line_color),
    )

    fig.add_annotation(
        x=raw["time"].iloc[-1],
        y=raw["low"].iloc[-1] * (0.9994 if "JPY" not in pair else 0.9998),
        text=f"{sig_symbol} {signal}",
        showarrow=False,
        font=dict(size=13, color=sig_color),
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor=sig_color,
        borderwidth=1,
        borderpad=5,
    )

    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=11),
        ),
        margin=dict(l=10, r=120, t=40, b=10),
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
    )
    fig.update_xaxes(
        tickformat="%d %b\n%H:%M", nticks=14,
        gridcolor="rgba(255,255,255,0.05)", showgrid=True,
    )
    fig.update_yaxes(
        tickformat=".5f",
        gridcolor="rgba(255,255,255,0.05)", showgrid=True, side="right",
    )
    return fig


# ====================================
# RUN PIPELINE
# ====================================
raw     = get_data(pair, candle_count)
feat_df = create_features(raw, pair)
current, p1h, p1d, prob, now_time = predict(raw, feat_df)
log_prediction(pair, current, p1h, p1d, prob, now_time)

t1h     = now_time + timedelta(hours=1)
t1d     = now_time + timedelta(hours=24)
atr_val = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else current * 0.001

if prob > 0.65:
    signal, sig_color, sig_symbol = "BUY 📈",      "#26a69a", "▲"
elif prob < 0.35:
    signal, sig_color, sig_symbol = "SELL 📉",     "#ef5350", "▼"
else:
    signal, sig_color, sig_symbol = "NO TRADE ⚖️", "#9E9E9E", "■"

delta_1h = f"{((p1h - current) / current * 100):+.4f}%"
delta_1d = f"{((p1d - current) / current * 100):+.4f}%"

metrics = compute_metrics()


# ====================================
# HEADER — time anchors
# ====================================
st.markdown(
    f"**Now:** `{now_time.strftime('%d %b %Y  %H:%M:%S')}`  │  "
    f"**1H target:** `{t1h.strftime('%d %b %Y  %H:%M')}`  │  "
    f"**1D target:** `{t1d.strftime('%d %b %Y  %H:%M')}`"
)


# ====================================
# METRIC CARDS
# ====================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",                                  f"{current:.5f}")
col2.metric(f"1H Forecast  [{t1h.strftime('%H:%M')}]",       f"{p1h:.5f}", delta=delta_1h)
col3.metric(f"1D Forecast  [{t1d.strftime('%d %b %H:%M')}]", f"{p1d:.5f}", delta=delta_1d)
col4.metric("Signal",                                         signal,       delta=f"confidence {prob:.1%}")
col5.metric(
    "1H MAE",
    f"{metrics['1H']['MAE']:.5f}" if metrics and "1H" in metrics else "—",
    delta=f"n={metrics['1H']['n']} bars" if metrics and "1H" in metrics else "collecting…",
)

st.divider()


# ====================================
# ACCURACY METRICS SECTION
# ====================================
st.subheader("📐 Model Accuracy Metrics")

if metrics:
    horizons = [k for k in ["1H", "1D"] if k in metrics]
    acc_cols = st.columns(len(horizons))

    for i, horizon in enumerate(horizons):
        m = metrics[horizon]
        with acc_cols[i]:
            st.markdown(f"#### {horizon} Prediction Accuracy  *(n={m['n']})*")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("MAE",  f"{m['MAE']:.5f}",   help="Mean Absolute Error — lower is better")
            r2.metric("RMSE", f"{m['RMSE']:.5f}",  help="Root Mean Squared Error")
            r3.metric("R²",   f"{m['R2']:.4f}",    help="R² score — closer to 1.0 is better")
            r4.metric("MAPE", f"{m['MAPE']:.3f}%", help="Mean Absolute % Error")

            df_m     = m["df"].copy().tail(30)
            pred_col = f"pred_{horizon.lower()}"
            df_m["abs_err"] = (df_m["actual"] - df_m[pred_col]).abs()

            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=df_m["timestamp"], y=df_m["abs_err"],
                mode="lines", fill="tozeroy",
                line=dict(color="#CE93D8", width=1.5),
                fillcolor="rgba(206,147,216,0.15)",
                hovertemplate="%{x|%d %b %H:%M}<br>Abs Error: %{y:.5f}<extra></extra>",
            ))
            fig_spark.update_layout(
                template="plotly_dark", height=130,
                margin=dict(l=0, r=0, t=22, b=0),
                plot_bgcolor="#131722", paper_bgcolor="#131722",
                showlegend=False,
                title=dict(text="Absolute Error over Time", font=dict(size=11)),
            )
            fig_spark.update_xaxes(
                tickformat="%d %b", nticks=5,
                gridcolor="rgba(255,255,255,0.05)",
            )
            fig_spark.update_yaxes(
                tickformat=".5f",
                gridcolor="rgba(255,255,255,0.05)", side="right",
            )
            st.plotly_chart(fig_spark, use_container_width=True)
else:
    st.info(
        "📊 Accuracy metrics appear automatically after **1 hour** of logged predictions. "
        "Keep the dashboard running and they will fill in on their own."
    )

st.divider()


# ====================================
# 1H FORECAST CHART
# ====================================
st.subheader(f"🕐 1-Hour Forecast — Target: {t1h.strftime('%d %b %Y  %H:%M')}")
st.caption(f"Predicted price 1 hour from now: **{p1h:.5f}**  ({delta_1h})")

fig_1h = build_forecast_chart(
    raw, feat_df, current,
    forecast_price=p1h,
    horizon_label="1H",
    target_time=t1h,
    now_time=now_time,
    line_color="#00E5FF",
    atr_val=atr_val,
    signal=signal, sig_color=sig_color, sig_symbol=sig_symbol,
    chart_height=500,
)
st.plotly_chart(fig_1h, use_container_width=True)

st.divider()


# ====================================
# 1D FORECAST CHART
# ====================================
st.subheader(f"📅 1-Day Forecast — Target: {t1d.strftime('%d %b %Y  %H:%M')}")
st.caption(f"Predicted price 24 hours from now: **{p1d:.5f}**  ({delta_1d})")

fig_1d = build_forecast_chart(
    raw, feat_df, current,
    forecast_price=p1d,
    horizon_label="1D",
    target_time=t1d,
    now_time=now_time,
    line_color="#FFD600",
    atr_val=atr_val,
    signal=signal, sig_color=sig_color, sig_symbol=sig_symbol,
    chart_height=500,
)
st.plotly_chart(fig_1d, use_container_width=True)

st.divider()


# ====================================
# RSI CHART
# ====================================
st.subheader("📉 RSI (14)")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=raw["time"], y=feat_df["rsi"],
    mode="lines", name="RSI",
    line=dict(color="#42A5F5", width=1.5),
    hovertemplate="%{x|%d %b %H:%M}  RSI: %{y:.2f}<extra></extra>",
))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)",
                  annotation_text="Overbought 70", annotation_position="right",
                  annotation_font=dict(size=10, color="#ef5350"))
fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)",
                  annotation_text="Oversold 30",   annotation_position="right",
                  annotation_font=dict(size=10, color="#26a69a"))
fig_rsi.update_layout(
    template="plotly_dark", height=180,
    margin=dict(l=10, r=120, t=10, b=10),
    plot_bgcolor="#131722", paper_bgcolor="#131722",
    showlegend=False, hovermode="x unified",
)
fig_rsi.update_xaxes(tickformat="%d %b\n%H:%M", nticks=14,
                     gridcolor="rgba(255,255,255,0.05)")
fig_rsi.update_yaxes(range=[0, 100], tickformat=".0f",
                     gridcolor="rgba(255,255,255,0.05)", side="right")
st.plotly_chart(fig_rsi, use_container_width=True)


# ====================================
# MACD CHART
# ====================================
st.subheader("📊 MACD")

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(
    x=raw["time"], y=feat_df["macd"],
    mode="lines", name="MACD",
    line=dict(color="#CE93D8", width=1.5),
    hovertemplate="%{x|%d %b %H:%M}  MACD: %{y:.6f}<extra></extra>",
))
fig_macd.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
fig_macd.update_layout(
    template="plotly_dark", height=160,
    margin=dict(l=10, r=120, t=10, b=10),
    plot_bgcolor="#131722", paper_bgcolor="#131722",
    showlegend=False, hovermode="x unified",
)
fig_macd.update_xaxes(tickformat="%d %b\n%H:%M", nticks=14,
                      gridcolor="rgba(255,255,255,0.05)")
fig_macd.update_yaxes(tickformat=".6f",
                      gridcolor="rgba(255,255,255,0.05)", side="right")
st.plotly_chart(fig_macd, use_container_width=True)

st.divider()


# ====================================
# FORECAST SUMMARY TABLE
# ====================================
st.subheader("🎯 Forecast Summary")

summary = pd.DataFrame([
    {
        "Horizon":      "1 Hour",
        "Predicted At": now_time.strftime("%d %b %Y %H:%M:%S"),
        "Target Time":  t1h.strftime("%d %b %Y  %H:%M"),
        "Current":      f"{current:.5f}",
        "Forecast":     f"{p1h:.5f}",
        "Change":       delta_1h,
        "Direction":    "▲ UP" if p1h > current else "▼ DOWN",
        "Signal":       signal,
        "Confidence":   f"{prob:.1%}",
        "MAE":          f"{metrics['1H']['MAE']:.5f}"   if metrics and "1H" in metrics else "—",
        "RMSE":         f"{metrics['1H']['RMSE']:.5f}"  if metrics and "1H" in metrics else "—",
        "R²":           f"{metrics['1H']['R2']:.4f}"    if metrics and "1H" in metrics else "—",
        "MAPE":         f"{metrics['1H']['MAPE']:.3f}%" if metrics and "1H" in metrics else "—",
    },
    {
        "Horizon":      "1 Day (24H)",
        "Predicted At": now_time.strftime("%d %b %Y %H:%M:%S"),
        "Target Time":  t1d.strftime("%d %b %Y  %H:%M"),
        "Current":      f"{current:.5f}",
        "Forecast":     f"{p1d:.5f}",
        "Change":       delta_1d,
        "Direction":    "▲ UP" if p1d > current else "▼ DOWN",
        "Signal":       signal,
        "Confidence":   f"{prob:.1%}",
        "MAE":          f"{metrics['1D']['MAE']:.5f}"   if metrics and "1D" in metrics else "—",
        "RMSE":         f"{metrics['1D']['RMSE']:.5f}"  if metrics and "1D" in metrics else "—",
        "R²":           f"{metrics['1D']['R2']:.4f}"    if metrics and "1D" in metrics else "—",
        "MAPE":         f"{metrics['1D']['MAPE']:.3f}%" if metrics and "1D" in metrics else "—",
    },
])
st.dataframe(summary, use_container_width=True, hide_index=True)


# ====================================
# PREDICTION LOG VIEWER
# ====================================
with st.expander("🗂️ Recent Prediction Log (last 20)"):
    if os.path.exists(LOG_FILE):
        df_log = (
            pd.read_csv(LOG_FILE)
            .query("pair == @pair")
            .tail(20)
            .iloc[::-1]
        )
        st.dataframe(
            df_log.style.format({
                "current": "{:.5f}",
                "pred_1h": "{:.5f}",
                "pred_1d": "{:.5f}",
                "prob":    "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions logged yet — will appear after the first run.")
