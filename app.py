# app.py
import os
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
import yfinance as yf

# ======================
# GLOBAL SETTINGS
# ======================
SEQ_LEN = 60
DEVICE = torch.device("cpu")  # Streamlit Cloud: CPU only
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
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, H)
        out = out[:, -1, :]     # last timestep
        out = self.fc(out)      # (B, 1)
        return out

# ======================
# TECHNICAL INDICATORS
# ======================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure correct dtypes
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")

    df = df.dropna(subset=["close", "volume", "high", "low"])

    # Simple daily return
    df["return"] = df["close"].pct_change()

    # Moving averages
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

    # OBV using numpy arrays (avoids ambiguous Series truth values)
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

    # Make sure we have a 'date' column
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    elif "index" in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)

    return df


# ======================
# HELPERS
# ======================
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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load model
    model = LSTMRegression(input_size=input_size)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, scaler

def build_last_sequence(df_ind: pd.DataFrame, scaler, seq_len: int):
    if len(df_ind) < seq_len:
        return None, None

    df_ind = df_ind.copy()
    df_ind["log_close"] = np.log(df_ind["close"])

    recent = df_ind.iloc[-seq_len:]

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

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(
    page_title="LSTM Stock Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LSTM-based Stock Price Prediction (5-Day Horizon)")

st.markdown(
    """
This web app exposes an LSTM model trained on multiple stocks with technical indicators.

**Model design:**
- Input: last **60 days** of price & indicators  
- Target: **5-day ahead log return** of the closing price  
- Features: OHLCV, daily return, MA(10/20), RSI, MACD, ATR, OBV  

Use the tabs below to switch between **Instructions**, **Training & Performance**, and **Prediction**.
"""
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Choose ticker", TICKERS, index=0)
    years_back = st.slider("History window (years)", min_value=1, max_value=5, value=3)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years_back)
    st.caption("Predictions are for ~5 trading days ahead based on the latest available data.")

# Tabs
tab_pred, tab_train, tab_info = st.tabs([ "ℹ️ Instructions","📈 Prediction", "📉 Training & Performance"])

# ======================
# TAB 1: INSTRUCTIONS
# ======================
with tab_info:
    st.subheader("User Instructions")

    st.markdown(
        """
### What this app does

- Uses an LSTM regression model trained on **AAPL, MSFT, and AMZN**.
- The model takes the last **60 days** of prices and technical indicators and predicts the **5-day ahead log return**.
- The predicted log return is converted to:
  - a **percentage return**, and  
  - an **implied future price** 5 days from the last available date.

### How to use it

1. **Select a ticker** in the sidebar (AAPL, MSFT, AMZN).
2. Choose the **history window** (number of years of past data to load).
3. Go to the **Prediction** tab:
   - Review recent historical prices in the table.
   - See the model’s predicted 5-day return and future price.
   - Inspect the price chart with the forecast point appended.
4. Go to the **Training & Performance** tab:
   - View the **training vs validation loss curves** for the selected ticker.
   - View the **MAE/RMSE summary table** (if provided from training notebook).

### Notes for evaluation

- This interface is designed for the course project requirement:
  - Web-based interactive app
  - Ticker input and visible predictions
  - Past history display
  - Training/validation curves
  - Additional figures/tables summarizing results
- All models and scalers were pre-trained offline in a Colab notebook and loaded here.
"""
    )
# ======================
# TAB 2: TRAINING & PERFORMANCE
# ======================
with tab_train:
    st.subheader(f"Training vs Validation Loss – {ticker}")

    losses_path = f"artifacts/losses_{ticker}.pkl"
    if os.path.exists(losses_path):
        with open(losses_path, "rb") as f:
            losses = pickle.load(f)
        train_losses = losses.get("train_losses", [])
        val_losses = losses.get("val_losses", [])

        if len(train_losses) > 0 and len(val_losses) == len(train_losses):
            epochs = list(range(1, len(train_losses) + 1))
            loss_df = pd.DataFrame(
                {
                    "epoch": epochs,
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                }
            ).set_index("epoch")

            st.line_chart(loss_df)
            st.caption("Lower validation loss over epochs indicates better generalization.")
        else:
            st.info("Loss data found but not in the expected format.")
    else:
        st.info("No saved training/validation curves for this ticker.")

    st.markdown("---")
    st.subheader("Overall LSTM Performance (All Tickers)")

    results_path = "artifacts/results_summary.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df)
    else:
        st.info("Summary results table not found. You can generate it from the training notebook as 'artifacts/results_summary.csv'.")
# ======================
# TAB 3: PREDICTION
# ======================
with tab_pred:
    st.subheader(f"Prediction for {ticker}")

    # Load price data
    with st.spinner(f"Downloading price data for {ticker}..."):
        df_raw = load_price_data(ticker, start_dt, end_dt)

    if df_raw.empty:
        st.error("No data returned. Try a different date range.")
    else:
        st.markdown("**Recent historical data**")
        st.dataframe(df_raw.tail(10))

        # Add indicators
        df_ind = add_indicators(df_raw)

        if len(df_ind) < SEQ_LEN:
            st.warning("Not enough data after adding indicators for this history window.")
        else:
            input_size = len(FEATURE_COLS)

            # Load model + scaler
            try:
                with st.spinner("Loading trained LSTM model and scaler..."):
                    model, scaler = load_model_and_scaler(ticker, input_size)
            except FileNotFoundError as e:
                st.error(str(e))
            else:
                # Build last sequence
                X_seq, meta = build_last_sequence(df_ind, scaler, SEQ_LEN)
                if X_seq is None:
                    st.warning("Not enough rows to build a full 60-day sequence.")
                else:
                    last_log_close, last_date, last_price = meta

                    # Predict
                    with st.spinner("Predicting 5-day ahead return and price..."):
                        pred_return_5d, pred_price_5d = predict_5day_price(
                            model, X_seq, last_log_close
                        )

                    pred_pct_return_5d = (np.exp(pred_return_5d) - 1) * 100
                    horizon_date = last_date + pd.Timedelta(days=5)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Last close", f"${last_price:,.2f}", f"as of {last_date.date()}")
                    col2.metric("Predicted 5-day return", f"{pred_pct_return_5d:,.2f}%")
                    col3.metric("Predicted price in ~5 days", f"${pred_price_5d:,.2f}", f"by {horizon_date.date()}")

                    st.markdown("### Price history and 5-day forecast")

                    plot_df = df_ind[["date", "close"]].copy()
                    plot_df.set_index("date", inplace=True)
                    plot_df.loc[horizon_date, "close_forecast"] = pred_price_5d

                    st.line_chart(plot_df)

