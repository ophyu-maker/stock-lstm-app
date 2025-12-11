# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
from datetime import date, timedelta

# ==========
# SETTINGS
# ==========
SEQ_LEN = 60
DEVICE = torch.device("cpu")  # Streamlit Cloud: stay on CPU

TICKERS = ["AAPL", "MSFT", "AMZN"]   # the tickers you trained on

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "return", "ma_10", "ma_20", "RSI", "MACD", "ATR", "OBV"
]

# ==========
# MODEL DEF
# ==========
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, H)
        out = out[:, -1, :]     # last timestep
        out = self.fc(out)      # (B, 1)
        return out

# ==========
# INDICATORS
# ==========
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()

    # MAs
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

    # MACD (12, 26)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ATR(14)
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    df = df.dropna().reset_index()
    df.rename(columns={"Date": "date"}, inplace=True, errors="ignore")
    if "date" not in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)

    return df

# ==========
# HELPERS
# ==========
@st.cache_data
def load_price_data(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    data = yf.download(ticker, start=start_dt, end=end_dt)
    data = data.dropna()
    data = data.rename(columns=str.lower)  # open, high, low, close, adj close, volume
    data.reset_index(inplace=True)
    data.rename(columns={"date": "date"}, inplace=True)
    return data

@st.cache_resource
def load_model_and_scaler(ticker: str, input_size: int):
    model_path = f"models/lstm_{ticker}.pth"
    scaler_path = f"artifacts/scaler_{ticker}.pkl"

    # load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # load model
    model = LSTMRegression(input_size=input_size)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, scaler

def build_last_sequence(df_ind: pd.DataFrame, scaler, seq_len: int):
    # we assume df_ind already has indicators & no NaNs
    if len(df_ind) < seq_len:
        return None, None

    df_ind = df_ind.copy()
    df_ind["log_close"] = np.log(df_ind["close"])

    recent = df_ind.iloc[-seq_len:]  # last seq_len rows

    X = scaler.transform(recent[FEATURE_COLS])
    X_seq = np.expand_dims(X, axis=0)  # (1, seq_len, n_features)

    last_log_close = recent["log_close"].iloc[-1]
    last_date = recent["date"].iloc[-1]
    last_price = recent["close"].iloc[-1]

    return X_seq, (last_log_close, last_date, last_price)

def predict_5day_price(model, X_seq, last_log_close):
    X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_return_5d = model(X_t).cpu().numpy().flatten()[0]

    pred_log_close_5d = last_log_close + pred_return_5d
    pred_price_5d = float(np.exp(pred_log_close_5d))
    return pred_return_5d, pred_price_5d

# ==========
# STREAMLIT UI
# ==========
st.set_page_config(page_title="LSTM Stock Prediction", page_icon="📊", layout="wide")

st.title("📊 LSTM-based Stock Prediction (5-Day Horizon)")

st.markdown(
    """
This app uses your **PyTorch LSTM model** with technical indicators to predict the
**5-day ahead log return** and implied **future price** for selected stocks.

The model was trained on historical data with:
- Lookback window: **60 days**
- Target: **5-day log return** of the closing price  
"""
)

with st.sidebar:
    st.header("Settings")

    ticker = st.selectbox("Ticker", options=TICKERS, index=0)
    years_back = st.slider("History window (years)", 1, 5, 3)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years_back)

    st.caption("Note: model expects the same features & scaling as used in training.")

# === Load data ===
with st.spinner(f"Downloading data for {ticker}..."):
    df_raw = load_price_data(ticker, start_dt, end_dt)

if df_raw.empty:
    st.error("No data returned. Try a different date range.")
    st.stop()

st.subheader(f"Recent data for {ticker}")
st.dataframe(df_raw.tail(10))

# === Indicators ===
df_ind = add_indicators(df_raw)

if len(df_ind) < SEQ_LEN:
    st.warning("Not enough data after adding indicators for the chosen date range.")
    st.stop()

input_size = len(FEATURE_COLS)

# === Load model + scaler ===
with st.spinner("Loading model & scaler..."):
    model, scaler = load_model_and_scaler(ticker, input_size)

# === Build last sequence ===
X_seq, meta = build_last_sequence(df_ind, scaler, SEQ_LEN)
if X_seq is None:
    st.warning("Not enough rows to build a full sequence.")
    st.stop()

last_log_close, last_date, last_price = meta

# === Predict ===
with st.spinner("Predicting 5-day ahead return and price..."):
    pred_return_5d, pred_price_5d = predict_5day_price(model, X_seq, last_log_close)

# Convert 5-day log return to % return
pred_pct_return_5d = (np.exp(pred_return_5d) - 1) * 100

# Assume horizon date = last_date + 5 days (approx)
horizon_date = last_date + pd.Timedelta(days=5)

st.subheader("Prediction")

col1, col2, col3 = st.columns(3)
col1.metric("Last close", f"${last_price:,.2f}", f"as of {last_date.date()}")
col2.metric("Predicted 5-day return", f"{pred_pct_return_5d:,.2f}%")
col3.metric("Predicted price in 5 days", f"${pred_price_5d:,.2f}", f"by ~{horizon_date.date()}")

# === Plot historical + forecast point ===
st.subheader("Price history and 5-day ahead forecast")

plot_df = df_ind[["date", "close"]].copy()
plot_df.set_index("date", inplace=True)

# Add forecast point
plot_df.loc[horizon_date, "close_forecast"] = pred_price_5d

st.line_chart(plot_df)

st.caption(
    "Model output is a 5-day ahead log return. We convert that to a % return and implied future price "
    "by applying it to the last observed closing price."
)
