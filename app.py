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
import altair as alt

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
# TECHNICAL INDICATORS  (same as your working version)
# ======================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators. Assumes columns:
    date, open, high, low, close, volume
    """
    df = df.copy()

    # Ensure numeric for calculations
    for col in ["close", "volume", "high", "low"]:
        df[col] = df[col].astype(float)

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

    # OBV using numpy arrays
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

    # Drop NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)

    return df

# ======================
# HELPERS
# ======================

# NEW loader with ttl and empty-data protection
@st.cache_data(ttl=3600)
def load_price_data_v2(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    """
    Download OHLCV data and make sure there is a 'date' column.
    Rejects empty responses so we don't cache a broken result.
    """
    data = yf.download(ticker, start=start_dt, end=end_dt)

    if data is None or data.empty:
        raise ValueError(
            f"No data returned from Yahoo Finance for {ticker}. "
            "This might be temporary; try again or adjust the date range."
        )

    data = data.dropna()
    data = data.rename(columns=str.lower)  # open, high, low, close, adj close, volume
    data.reset_index(inplace=True)        # index -> column (usually 'Date')
    # Normalise to 'date'
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
    """Build last 60-day sequence and fetch last date/price."""
    if len(df_ind) < seq_len:
        return None, None

    df_ind = df_ind.copy()

    # Ensure we know which column is date
    date_col = "date"
    if date_col not in df_ind.columns:
        for c in df_ind.columns:
            if np.issubdtype(df_ind[c].dtype, np.datetime64):
                date_col = c
                break

    df_ind["log_close"] = np.log(df_ind["close"])

    recent = df_ind.iloc[-seq_len:]

    X = scaler.transform(recent[FEATURE_COLS])
    X_seq = np.expand_dims(X, axis=0)  # (1, seq_len, n_features)

    last_log_close = recent["log_close"].iloc[-1]
    last_date = recent[date_col].iloc[-1]
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

**Model design (for the professor):**
- Input: last **60 days** of price & indicators  
- Target: **5-day ahead log return** of the closing price  
- Features: OHLCV, daily return, MA(10/20), RSI, MACD, ATR, OBV  
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

# Tabs in order: Instructions -> Training & Performance -> Prediction
tab_info, tab_train, tab_pred = st.tabs(
    ["ℹ️ Instructions", "📉 Training & Performance", "📈 Prediction"]
)

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

1. In the sidebar, select a **ticker** and **history window**.
2. Go to the **Prediction** tab to:
   - Review recent historical prices.
   - See the model’s 5-day-ahead return and future price.
   - Inspect the price chart with a 5-day price path.
3. Go to the **Training & Performance** tab to:
   - See training vs validation loss curves.
   - View the MAE/RMSE summary table across tickers (if provided).

### Notes

- Models and scalers are pre-trained offline and loaded here.
- This interface is designed to meet the course project requirements
  (interactive web app, history, prediction, training curves, and tables).
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
        st.info(
            "Summary results table not found. "
            "You can generate it from the training notebook as 'artifacts/results_summary.csv'."
        )

# ======================
# TAB 3: PREDICTION
# ======================
with tab3:

    st.markdown("### Prediction")

    # Re-load indicator dataframe so it's fresh
    df_ind = add_indicators(df_raw.copy())

    # Must have at least 60 rows
    if len(df_ind) < 60:
        st.error("Not enough data for prediction (need at least 60 trading days). Try increasing your history window.")
        st.stop()

    # Compute the model target (future 5-day log return)
    df_ind = compute_future_return(df_raw, df_ind)

    # Load the model and scaler
    try:
        model = load_model(SAVED_MODEL_PATH)
        scaler = joblib.load(SAVED_SCALER_PATH)
    except Exception as e:
        st.error("Model or scaler file not found. Upload them first.")
        st.stop()

    # Use the last 60 days of features
    recent = df_ind.tail(60)
    X_input = scaler.transform(recent[FEATURE_COLS])
    X_input = X_input.reshape(1, 60, len(FEATURE_COLS))

    # Predict scaled return, convert to real return
    pred_scaled = model.predict(X_input)[0][0]
    pred_return = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Convert predicted return into 5-day price path
    last_close = df_ind["close"].iloc[-1]
    final_price = last_close * np.exp(pred_return)

    # Spread return evenly across 5 days
    daily_return = pred_return / 5
    forecast_prices = [last_close * np.exp(daily_return * (i + 1)) for i in range(5)]

    # Estimated dates (next 5 business days)
    last_date = df_ind["date"].iloc[-1]
    horizon_dates = [last_date + pd.Timedelta(days=i+1) for i in range(5)]

    st.markdown(f"### Prediction for **{ticker}**")
    st.write("Recent historical data")

    # Display last 5 days of historical data
    st.dataframe(df_ind.tail(5)[["date", "open", "high", "low", "close", "volume"]], use_container_width=True)

    st.markdown("### Price history and 5-day forecast")

    # --------- HISTORY (last 5 closes) ----------
    last5 = df_ind.tail(5).copy()

    hist_dates = pd.to_datetime(last5["date"]).dt.normalize().tolist()
    hist_prices = last5["close"].astype(float).tolist()

    hist_df = pd.DataFrame({
        "date": hist_dates,
        "price": hist_prices,
        "series": ["History (close)"] * len(hist_prices),
    })

    # --------- FORECAST (next 5 days) ----------
    forecast_dates_norm = [pd.to_datetime(d).normalize() for d in horizon_dates]
    forecast_df = pd.DataFrame({
        "date": forecast_dates_norm,
        "price": list(forecast_prices),
        "series": ["Forecast"] * len(forecast_prices),
    })

    # Combine history + forecast
    plot_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    # Build Altair line chart
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
        "Forecast shows the predicted 5-day price path, assuming the 5-day "
        "log return is distributed equally across the next 5 days."
    )

