# app.py

import os
import pickle
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
import yfinance as yf

# ======================
# GLOBALS / SETTINGS
# ======================
SEQ_LEN = 60
DEVICE = torch.device("cpu")  # Streamlit Cloud CPU
TICKERS = ["AAPL", "MSFT", "AMZN"]

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "return", "ma_10", "ma_20", "RSI", "MACD", "ATR", "OBV"
]


# ======================
# MODEL DEFINITION
# ======================
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ======================
# TECHNICAL INDICATORS
# (same logic as training notebook)
# ======================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = df[col].astype(float)

    # daily return
    df["return"] = df["close"].pct_change()

    # moving averages
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    # RSI(14)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ATR(14)
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()

    # OBV
    obv = [0]
    close_vals = df["close"].values
    vol_vals = df["volume"].values
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i - 1]:
            obv.append(obv[-1] + vol_vals[i])
        elif close_vals[i] < close_vals[i - 1]:
            obv.append(obv[-1] - vol_vals[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    df = df.dropna().reset_index(drop=True)
    return df


# ======================
# DATA LOADER (SAFE, CACHED)
# ======================
@st.cache_data(ttl=3600)
def load_price_data_v2(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    """Download price data and normalise column names."""
    data = yf.download(ticker, start=start_dt, end=end_dt)

    if data is None or data.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for {ticker}. "
            "Try a different date range or clear cache and reload."
        )

    data = data.dropna()
    data = data.rename(columns=str.lower)  # open/high/low/close/adj close/volume
    data.reset_index(inplace=True)

    # normalise date column
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    elif "date" not in data.columns and "index" in data.columns:
        data.rename(columns={"index": "date"}, inplace=True)

    return data


@st.cache_resource
def load_model_and_scaler(ticker: str, input_size: int):
    model_path = f"models/lstm_{ticker}.pth"
    scaler_path = f"artifacts/scaler_{ticker}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMRegression(input_size=input_size)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, scaler


def build_last_sequence(df_ind: pd.DataFrame, scaler, seq_len: int = SEQ_LEN):
    """Build final 60-day sequence and return (X_seq, (last_log_close, last_date, last_price))."""
    if len(df_ind) < seq_len:
        return None, None

    df = df_ind.copy()
    df["log_close"] = np.log(df["close"])

    recent = df.tail(seq_len)
    X = scaler.transform(recent[FEATURE_COLS])
    X_seq = np.expand_dims(X, axis=0)  # (1, seq_len, n_features)

    last_log_close = recent["log_close"].iloc[-1]
    last_date = recent["date"].iloc[-1]
    last_price = recent["close"].iloc[-1]
    return X_seq, (last_log_close, last_date, last_price)


def predict_5day_price(model, X_seq, last_log_close):
    X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_5d_log_return = model(X_t).cpu().numpy().flatten()[0]

    pred_log_close_5d = last_log_close + pred_5d_log_return
    pred_price_5d = float(np.exp(pred_log_close_5d))
    return pred_5d_log_return, pred_price_5d


# ======================
# STREAMLIT UI SETUP
# ======================
st.set_page_config(
    page_title="LSTM Stock Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LSTM-based Stock Price Prediction (5-Day Horizon)")
st.markdown(
    """
This web app exposes an LSTM model trained on multiple stocks with technical indicators.

**Model design (for the professor):**
- Input: last **60 days** of price & indicators  
- Target: **5-day ahead log return** of the closing price  
- Features: OHLCV, daily return, MA(10/20), RSI, MACD, ATR, OBV  
"""
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Choose ticker", TICKERS, index=0)
    years_back = st.slider("History window (years)", 1, 5, 3)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years_back)
    st.caption("Predictions are for ~5 trading days ahead based on the latest available data.")

# Load raw price data once, reused across tabs
try:
    df_raw = load_price_data_v2(ticker, start_dt, end_dt)
except Exception as e:
    st.error(str(e))
    st.stop()

# ======================
# TABS
# ======================
tab_info, tab_train, tab_pred = st.tabs(
    ["ℹ️ Instructions", "📉 Training & Performance", "📈 Prediction"]
)

# -------- TAB 1: INSTRUCTIONS --------
with tab_info:
    st.subheader("User Instructions")
    st.markdown(
        """
### What this app does
- Uses an LSTM regression model trained on **AAPL, MSFT, and AMZN**.
- Takes the last **60 days** of prices and technical indicators and predicts the **5-day ahead log return**.
- Converts that to both a **percentage return** and an **implied price** 5 days ahead.

### How to use it
1. Select a **ticker** and **history window** in the sidebar.  
2. Go to the **Prediction** tab to see:
   - Recent historical data  
   - Predicted 5-day return and price  
   - Price history + 5-day forecast chart  
3. Go to the **Training & Performance** tab to inspect training curves and summary metrics.

### Notes
- Models and scalers were trained offline and loaded here.
- This interface is designed to satisfy the course requirement for an interactive web app
  that displays history, predictions, and training diagnostics.
"""
    )

# -------- TAB 2: TRAINING & PERFORMANCE --------
with tab_train:
    st.subheader(f"Training vs Validation Loss – {ticker}")

    losses_path = f"artifacts/losses_{ticker}.pkl"
    if os.path.exists(losses_path):
        with open(losses_path, "rb") as f:
            losses = pickle.load(f)
        train_losses = losses.get("train_losses", [])
        val_losses = losses.get("val_losses", [])

        if train_losses and len(train_losses) == len(val_losses):
            epochs = list(range(1, len(train_losses) + 1))
            loss_df = pd.DataFrame(
                {"epoch": epochs, "train_loss": train_losses, "val_loss": val_losses}
            ).set_index("epoch")
            st.line_chart(loss_df)
            st.caption("Lower validation loss indicates better generalization.")
        else:
            st.info("Loss data found, but format is unexpected.")
    else:
        st.info("No saved training/validation curves for this ticker.")

    st.markdown("---")
    st.subheader("Overall LSTM Performance (All Tickers)")
    results_path = "artifacts/results_summary.csv"
    if os.path.exists(results_path):
        st.dataframe(pd.read_csv(results_path))
    else:
        st.info(
            "Summary results table not found. "
            "You can export it from the training notebook as 'artifacts/results_summary.csv'."
        )

# -------- TAB 3: PREDICTION --------
with tab_pred:
    st.subheader(f"Prediction for {ticker}")

    st.markdown("**Recent historical data**")
    st.dataframe(df_raw.tail(10), use_container_width=True)

    # Add indicators
    df_ind = add_indicators(df_raw)
    if len(df_ind) < SEQ_LEN:
        st.warning("Not enough data after indicators for 60-day sequence. Increase date range.")
        st.stop()

    input_size = len(FEATURE_COLS)

    # Load model and scaler
    try:
        model, scaler = load_model_and_scaler(ticker, input_size)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Build final sequence
    X_seq, meta = build_last_sequence(df_ind, scaler, SEQ_LEN)
    if X_seq is None or meta is None:
        st.warning("Could not build a full 60-day sequence.")
        st.stop()

    last_log_close, last_date, last_price = meta

    # Predict
    pred_log_return_5d, pred_price_5d = predict_5day_price(model, X_seq, last_log_close)

    # Convert to percent return
    pred_pct_return_5d = (np.exp(pred_log_return_5d) - 1) * 100

    # Horizon dates and daily path
    horizon_date = last_date + pd.Timedelta(days=5)
    daily_log_ret = pred_log_return_5d / 5.0
    horizon_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
    forecast_prices = [
        float(last_price) * float(np.exp(daily_log_ret * i))
        for i in range(1, 6)
    ]

    # Scalars for metrics
    last_price_float = float(np.asarray(last_price).reshape(-1)[0])
    pred_ret_float = float(pred_pct_return_5d)
    pred_price_float = float(pred_price_5d)
    last_date_str = str(pd.to_datetime(last_date).date())
    horizon_date_str = str(pd.to_datetime(horizon_date).date())

    col1, col2, col3 = st.columns(3)
    col1.metric("Last close", f"${last_price_float:,.2f}", f"as of {last_date_str}")
    col2.metric("Predicted 5-day return", f"{pred_ret_float:,.2f}%")
    col3.metric(
        "Predicted price in ~5 days",
        f"${pred_price_float:,.2f}",
        f"by {horizon_date_str}",
    )

    st.markdown("#### Approximate day-by-day forecast")
    forecast_table = pd.DataFrame(
        {
            "date": [d.date() for d in horizon_dates],
            "predicted_price": forecast_prices,
        }
    )
    st.dataframe(forecast_table, use_container_width=True)

    st.markdown("### Price history and 5-day forecast")

    # ---- History: last 5 closes ----
    last5 = df_ind.tail(5).copy()
    hist_dates = pd.to_datetime(last5["date"]).dt.normalize().tolist()
    hist_prices = last5["close"].astype(float).tolist()

    hist_df = pd.DataFrame(
        {
            "date": hist_dates,
            "price": hist_prices,
            "series": ["History (close)"] * len(hist_prices),
        }
    )

    # ---- Forecast: next 5 days ----
    forecast_dates_norm = [pd.to_datetime(d).normalize() for d in horizon_dates]
    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates_norm,
            "price": list(forecast_prices),
            "series": ["Forecast"] * len(forecast_prices),
        }
    )

    plot_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(format="%b %d", labelAngle=-45),
            ),
            y=alt.Y("price:Q", title="Price (USD)"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date:T", "series:N", "price:Q"],
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "History shows the last 5 closing prices. "
        "Forecast shows an approximate 5-day price path, assuming the 5-day "
        "log return is distributed equally across the next 5 days."
    )
