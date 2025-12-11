###############################################################################
# 📌 LSTM Stock Prediction Web App (5-Day Horizon)
# Fixed version with improved data loader that prevents caching empty results.
###############################################################################

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from datetime import date, timedelta

import torch
import torch.nn as nn
from pathlib import Path
import pickle


###############################################################################
# ----------------------------- Model Definition ------------------------------
###############################################################################

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


###############################################################################
# ---------------------------- SAFE DATA LOADER -------------------------------
###############################################################################

@st.cache_data(ttl=3600)   # cache for 1 hour; prevents stale data
def load_price_data_v2(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    """
    Download price data from Yahoo Finance with protection:
    - Rejects empty data
    - Prevents caching broken results
    """
    data = yf.download(ticker, start=start_dt, end=end_dt)

    if data is None or data.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for {ticker}. "
            "This may be a temporary issue. Try a different date range or clear cache."
        )

    data = data.dropna()
    data = data.rename(columns=str.lower)
    data.reset_index(inplace=True)

    # Ensure the date column is named "date"
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    elif "date" not in data.columns and "index" in data.columns:
        data.rename(columns={"index": "date"}, inplace=True)

    return data


###############################################################################
# --------------------------- Technical Indicators -----------------------------
###############################################################################

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)

    df = df.dropna().reset_index(drop=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, f=12, s=26):
    exp1 = series.ewm(span=f, adjust=False).mean()
    exp2 = series.ewm(span=s, adjust=False).mean()
    return exp1 - exp2


def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.to_frame()
    tr["hc"] = high_close
    tr["lc"] = low_close
    tr = tr.max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(df):
    direction = np.where(df["close"] > df["close"].shift(), 1, -1)
    direction[0] = 0
    return (direction * df["volume"]).cumsum()


###############################################################################
# ----------------------- Sequence Builder for LSTM ---------------------------
###############################################################################

SEQ_LEN = 60

def build_last_sequence(df, scaler, seq_len=SEQ_LEN):
    last_block = df.drop(columns=["date"]).tail(seq_len)
    arr = scaler.transform(last_block)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


###############################################################################
# --------------------------------- UI SETUP ----------------------------------
###############################################################################

st.set_page_config(page_title="LSTM Stock Forecaster", layout="wide")

st.title("📈 LSTM-based Stock Price Prediction (5-Day Horizon)")
st.write(
    "This web app exposes an LSTM model trained on multiple stocks "
    "with technical indicators. Predictions estimate **5-day ahead returns**."
)

###############################################################################
# Sidebar controls
###############################################################################

st.sidebar.header("Settings")

ticker = st.sidebar.selectbox("Choose ticker", ["AAPL", "AMZN", "MSFT"])

years = st.sidebar.slider("History window (years)", 1, 10, 3)
end_dt = date.today()
start_dt = end_dt - timedelta(days=365 * years)

st.sidebar.write(
    "Predictions are for ~5 trading days ahead based on the latest available data."
)


###############################################################################
# ------------------------------- Load Data -----------------------------------
###############################################################################

try:
    df_raw = load_price_data_v2(ticker, start_dt, end_dt)
except Exception as e:
    st.error(str(e))
    st.stop()

df = add_indicators(df_raw)
if df.empty:
    st.error("Indicators produced an empty dataset. Try increasing date range.")
    st.stop()

###############################################################################
# --------------------------- Load Model + Scaler ------------------------------
###############################################################################

MODEL_PATH = Path("models") / f"lstm_{ticker}.pth"
SCALER_PATH = Path("models") / f"scaler_{ticker}.pkl"

device = torch.device("cpu")

model = LSTMModel(input_size=df.drop(columns=["date"]).shape[1])
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

###############################################################################
# ----------------------------- Make Prediction -------------------------------
###############################################################################

seq_tensor = build_last_sequence(df, scaler)
with torch.no_grad():
    pred_log_return = model(seq_tensor).item()

last_price = df["close"].iloc[-1]
predicted_5d_price = last_price * np.exp(pred_log_return)

###############################################################################
# ------------------------- Display Prediction Results -------------------------
###############################################################################

st.header(f"Prediction for {ticker}")

st.metric("Last close", f"${last_price:,.2f}", delta=None)
st.metric("Predicted price in ~5 days", f"${predicted_5d_price:,.2f}")

###############################################################################
# ----------------- Add 5-day forecast curve (Altair chart) -------------------
###############################################################################

daily_growth = np.exp(pred_log_return / 5)
forecast_prices = [last_price * (daily_growth ** i) for i in range(1, 6)]
forecast_dates = [df["date"].iloc[-1].date() + timedelta(days=i) for i in range(1, 6)]

hist_last5 = df[["date", "close"]].tail(5)
hist_last5["series"] = "History"
hist_last5.rename(columns={"close": "price"}, inplace=True)

forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "price": forecast_prices,
    "series": ["Forecast"] * 5,
})

chart_df = pd.concat([hist_last5, forecast_df], ignore_index=True)

chart = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("price:Q", title="Price (USD)"),
        color="series:N",
        tooltip=["date:T", "series:N", "price:Q"]
    )
    .properties(height=350)
)

st.subheader("Price history and 5-day forecast")
st.altair_chart(chart, use_container_width=True)

