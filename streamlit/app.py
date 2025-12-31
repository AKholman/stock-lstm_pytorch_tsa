import os, sys
import streamlit as st
import torch, joblib, yfinance as yf, pandas as pd
import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src.models.lstm import LSTMModel
from src.data.preprocess import create_sequences

st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="centered")
st.title("📈 Stock Price Prediction using LSTM (PyTorch)")

# ---------------------------
# User input
# ---------------------------
ticker = st.text_input("Enter stock ticker symbol:", "AAPL")
period = st.selectbox("Select data period:", ["1y","3y","5y"], index=2)

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = LSTMModel(input_dim=6).to(device)
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=device))
    model.eval()
    scaler_y = joblib.load("models/scaler_y.pkl")
    return model, scaler_y

model, scaler_y = load_model()

# ---------------------------
# Download & preprocess data
# ---------------------------
df = yf.download(ticker, period=period, interval="1d").rename(columns={"Adj Close":"Adj_Close"})
df["Target"] = df["Adj_Close"].shift(-1)
df.dropna(inplace=True)

FEATURES = ["Open","High","Low","Close","Adj_Close","Volume"]
X = df[FEATURES].values
y = df[["Target"]].values

time_steps = 60
X_seq, y_seq = create_sequences(X, y, time_steps)
X_seq_t = torch.tensor(X_seq, dtype=torch.float32)  # shape: (num_samples, seq_len, input_dim)

# ---------------------------
# Predict all
# ---------------------------
with torch.no_grad():
    y_pred_scaled = model(X_seq_t).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)

# ---------------------------
# Display predictions
# ---------------------------
st.subheader("📈 Predictions")
pred_df = pd.DataFrame({"Date": df.index[-len(y_pred):], "Predicted": y_pred.flatten()})
st.line_chart(pred_df.set_index("Date"))

# ---------------------------
# Next-day forecast
# ---------------------------
last_seq = torch.tensor(X[-time_steps:], dtype=torch.float32).unsqueeze(0)  # add batch dim
with torch.no_grad():
    next_day_scaled = model(last_seq).cpu().numpy()
next_day_price = scaler_y.inverse_transform(next_day_scaled)[0][0]

st.subheader("📅 Next-Day Forecast")
st.write(f"**Predicted price:** ${next_day_price:.2f}")
st.success("✅ Done")
