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

st.title("📊LSTM-based Stock Price Prediction (5-Day Horizon)")

st.markdown(
    """
This web app exposes an LSTM model trained on APPL, AMNZ, and MSFT stocks with technical indicators.

**Model design:**
- Input: last **60 days** of price & indicators  
- Target: **5-day ahead log return** of the closing price  
- Features: OHLCV, daily return, MA(10/20), RSI, MACD, ATR, OBV  
"""
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Choose ticker", TICKERS, index=0)
    years_back = st.slider("Historical data range (for table data only)", min_value=1, max_value=5, value=1)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years_back)
    st.caption("Select how many past stock data you want to view in the Recent Historical Data table ")

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

1. In the sidebar, select a **ticker** and **history data range** (to see past stock history on Prediction tab).
2. Go to the **Prediction** tab to:
   - Review recent historical prices.
   - See the model’s 5-day-ahead return and future price.
   - Inspect the price chart with a 5-day price path.
3. Go to the **Training & Performance** tab to:
   - See training vs validation loss curves.
   - Explore prediction diagnostics, including actual vs. predicted returns, residual patterns, and error-volatility relationships.
   - View the MAE/RMSE summary table across tickers.

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
    st.subheader("Prediction Diagnostics")

    # ------------------------------
    # 2. LOAD DIAGNOSTICS CSV
    # ------------------------------
    diag_path = f"artifacts/diagnostics_{ticker}.csv"

    if not os.path.exists(diag_path):
        st.warning("No diagnostics file found for this ticker.")
        st.stop()

    diag_df = pd.read_csv(diag_path, parse_dates=["date"])

    # Extract series
    dates        = diag_df["date"]
    actual       = diag_df["actual_return"]
    pred         = diag_df["pred_return"]
    residuals    = diag_df["residual"]
    abs_error    = diag_df["abs_error"]
    actual_abs   = diag_df["actual_abs"]

    # ------------------------------
    # 3. Actual vs Predicted Line Plot
    # ------------------------------
    st.subheader(f"Actual vs Predicted 5-Day Returns – {ticker}")

    line_df = pd.DataFrame({
        "date": dates,
        "Actual": actual,
        "Predicted": pred
    }).set_index("date")

    st.line_chart(line_df)

    st.caption(
        "Shows how closely the model follows actual 5-day log returns. "
        "Large gaps indicate prediction difficulty during volatile periods."
    )

    # ------------------------------
    # 4. Residual Plot
    # ------------------------------
    st.subheader(f"Residuals Over Test Set (Pred − Actual) – {ticker}")
 

    res_df = pd.DataFrame({
        "index": range(len(residuals)),
        "Residual": residuals
    }).set_index("index")

    st.line_chart(res_df)

    st.caption(
        "Residuals near zero indicate accurate predictions. "
        "Large positive or negative spikes show where the model struggled."
    )

    # ------------------------------
    # 5. Error vs Volatility Scatter
    # ------------------------------
    st.subheader(f"Error vs Volatility – {ticker}")
 
    scatter_df = pd.DataFrame({
        "|Actual Return|": actual_abs,
        "|Error|": abs_error
    })

    st.scatter_chart(scatter_df)

    st.caption(
        "Points higher on the chart indicate large prediction errors. "
        "If errors increase with |actual return|, the model struggles more in volatile periods."
    )

    st.markdown("---")

    # ------------------------------
    # 6. Results table
    # ------------------------------
    st.subheader("Overall LSTM Performance (All Tickers)")

    results_path = "artifacts/results_summary.csv"
    if os.path.exists(results_path):
        st.dataframe(pd.read_csv(results_path))
    else:
        st.info("results_summary.csv not found.")



# ======================
# TAB 3: PREDICTION
# ======================
with tab_pred:
    st.subheader(f"Prediction for {ticker}")

    # Load price data using the safe loader
    try:
        with st.spinner(f"Downloading price data for {ticker}..."):
            df_raw = load_price_data_v2(ticker, start_dt, end_dt)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if df_raw.empty:
        st.error("No data returned. Try a different date range.")
        st.stop()

    st.markdown("**Recent historical data**")

# Option 1: user controls how many rows to see
    num_rows = st.slider(
        "Number of historical rows to display",
        min_value=10,
        max_value=int(len(df_raw)),
        value=min(50, int(len(df_raw))),
        step=10,
    )

    st.dataframe(df_raw.tail(num_rows))


    # Add indicators
    df_ind = add_indicators(df_raw)

    if len(df_ind) < SEQ_LEN:
        st.warning("Not enough data after adding indicators for this history window.")
        st.stop()

    input_size = len(FEATURE_COLS)

    # Load model + scaler
    try:
        with st.spinner("Loading trained LSTM model and scaler..."):
            model, scaler = load_model_and_scaler(ticker, input_size)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Build last sequence
    X_seq, meta = build_last_sequence(df_ind, scaler, SEQ_LEN)
    if X_seq is None or meta is None:
        st.warning("Not enough rows to build a full 60-day sequence.")
        st.stop()

    last_log_close, last_date, last_price = meta

    # Predict
    with st.spinner("Predicting 5-day ahead return and price..."):
        pred_return_5d, pred_price_5d = predict_5day_price(
            model, X_seq, last_log_close
        )

    # Convert to percent & horizon date
    pred_pct_return_5d = (np.exp(pred_return_5d) - 1) * 100
    horizon_date = last_date + pd.Timedelta(days=5)

    # Safe scalars
    last_price_float = float(last_price)
    pred_ret_float = float(pred_pct_return_5d)
    pred_price_float = float(pred_price_5d)

    last_date_str = str(pd.to_datetime(last_date).date())
    horizon_date_str = str(pd.to_datetime(horizon_date).date())

    # ========== 5-day daily path ==========
    daily_log_ret = pred_return_5d / 5.0
    horizon_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
    forecast_prices = [
        last_price_float * float(np.exp(daily_log_ret * i))
        for i in range(1, 6)
    ]

    # ===== Metrics =====
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Last close",
        f"${last_price_float:,.2f}",
        f"as of {last_date_str}",
    )
    col2.metric(
        "Predicted 5-day return",
        f"{pred_ret_float:,.2f}%",
    )
    col3.metric(
        "Predicted price in ~5 days",
        f"${pred_price_float:,.2f}",
        f"by {horizon_date_str}",
    )

    # ===== Day-by-day forecast table =====
    st.markdown("#### Approximate day-by-day forecast (equal daily returns assumption)")
    forecast_table = pd.DataFrame(
        {
            "date": [d.date() for d in horizon_dates],
            "predicted_price": forecast_prices,
        }
    )
    st.dataframe(forecast_table)

    # ============================================================
    # PRICE HISTORY + 5-DAY FORECAST CHART
    # ============================================================
    st.markdown("### Price history and 5-day forecast")

    # ---- Last 5 historical closes ----
    hist_last5 = df_ind[["date", "close"]].copy().tail(5)

    if hist_last5.empty:
        st.info("No historical prices available for the last 5 days.")
    else:
        # Clean up history
        hist_last5["date"] = pd.to_datetime(hist_last5["date"]).dt.normalize()
        hist_last5.rename(columns={"close": "price"}, inplace=True)
        hist_last5["price"] = hist_last5["price"].astype(float)

        # ---- Next 5 forecast prices ----
        forecast_df = pd.DataFrame(
            {
                "date": [pd.to_datetime(d).normalize() for d in horizon_dates],
                "price": forecast_prices,
            }
        )
        forecast_df["price"] = forecast_df["price"].astype(float)

        # Ensure datetime for both
        hist_last5["date"] = pd.to_datetime(hist_last5["date"])
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

     # ---- Build layered Altair chart (simple, with string dates) ----

        # Make nice string labels so x-axis is discrete (no duplicate dates)
        hist_last5["date_str"] = hist_last5["date"].dt.strftime("%b %d")
        forecast_df["date_str"] = forecast_df["date"].dt.strftime("%b %d")

        hist_chart = (
            alt.Chart(hist_last5)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x=alt.X(
                    "date_str:N",
                    title="Date",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y("price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="History close"),
                ],
            )
        )

        forecast_chart = (
            alt.Chart(forecast_df)
            .mark_line(point=True, color="#7fbfff")
            .encode(
                x=alt.X(
                    "date_str:N",
                    title="Date",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y("price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="Forecast"),
                ],
            )
        )

        chart = (hist_chart + forecast_chart).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            "<span style='color:#1f77b4'>■</span> History (last 5 closes) &nbsp;&nbsp; "
            "<span style='color:#7fbfff'>■</span> Forecast (next 5 days)",
            unsafe_allow_html=True,
        )

        st.caption(
            "History shows the last 5 closing prices. "
            "Forecast shows an approximate 5-day price path, assuming the "
            "5-day log return is distributed equally across the next 5 days."
        )
                
           # ============================================================
    # TECHNICAL INDICATOR CHARTS
    # ============================================================
    st.markdown("### Technical indicator charts")

    # Make a safe copy with proper datetime
    ind_df = df_ind.copy()
    if "date" in ind_df.columns:
        ind_df["date"] = pd.to_datetime(ind_df["date"])

    # Checkboxes to toggle which indicators to show
    col_rsi, col_macd, col_ma = st.columns(3)
    show_rsi = col_rsi.checkbox("Show RSI (14)", value=True, key="show_rsi")
    show_macd = col_macd.checkbox("Show MACD (12, 26)", value=True, key="show_macd")
    show_ma = col_ma.checkbox("Show MA10 & MA20", value=True, key="show_ma")

    # ----- RSI chart -----
    if show_rsi and "RSI" in ind_df.columns:
        rsi_chart = (
            alt.Chart(ind_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("RSI:Q", title="RSI (14)"),
                tooltip=["date:T", "RSI:Q"],
            )
            .properties(height=200, title="RSI (14)")
        )
        st.altair_chart(rsi_chart, use_container_width=True)

    # ----- MACD chart -----
    if show_macd and "MACD" in ind_df.columns:
        macd_chart = (
            alt.Chart(ind_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("MACD:Q", title="MACD (12, 26)"),
                tooltip=["date:T", "MACD:Q"],
            )
            .properties(height=200, title="MACD (12, 26)")
        )
        st.altair_chart(macd_chart, use_container_width=True)

    # ----- MA10 & MA20 on price -----
    if show_ma and all(c in ind_df.columns for c in ["ma_10", "ma_20", "close"]):
        ma_df = ind_df[["date", "close", "ma_10", "ma_20"]].melt(
            id_vars=["date"],
            value_vars=["close", "ma_10", "ma_20"],
            var_name="series",
            value_name="value",
        )

        ma_chart = (
            alt.Chart(ma_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Price / Moving Averages"),
                color=alt.Color(
                    "series:N",
                    title="Series",
                    scale=alt.Scale(
                        domain=["close", "ma_10", "ma_20"],
                        range=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    ),
                ),
                tooltip=["date:T", "series:N", "value:Q"],
            )
            .properties(height=250, title="Close vs MA10 & MA20")
        )
        st.altair_chart(ma_chart, use_container_width=True)
