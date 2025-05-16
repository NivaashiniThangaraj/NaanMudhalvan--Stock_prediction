# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 AI-driven Stock Price Prediction")
st.markdown("Predict stock prices using Random Forest and time series data.")

# --- STEP 1: Get stock symbol and load data ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")

if st.button("Load and Predict"):
    st.info("Fetching data...")
    df = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    df.reset_index(inplace=True)

    if df.empty:
        st.error("No data found for this ticker.")
    else:
        st.success(f"Data for {ticker} loaded successfully!")
        st.subheader("📊 Raw Data")
        st.write(df.tail())

        # --- STEP 2: Feature Engineering ---
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Open-Close'] = df['Open'] - df['Close']
        df['High-Low'] = df['High'] - df['Low']

        # Select features
        features = ['Open', 'High', 'Low', 'Volume', 'Open-Close', 'High-Low', 'Year', 'Month', 'Day', 'DayOfWeek']
        X = df[features]
        y = df['Close']

        # --- STEP 3: Data Normalization ---
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # --- STEP 4: Train/Test Split ---
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- STEP 5: Train Model ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- STEP 6: Make Predictions ---
        y_pred = model.predict(X_test)

        # --- STEP 7: Plot Predictions ---
        st.subheader("📈 Predicted vs Actual Closing Price")
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
        ax.plot(y_test.index, y_pred, label="Predicted", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        # --- STEP 8: Show Metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📋 Model Performance")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")
