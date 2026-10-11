"""Streamlit dashboard:  streamlit run app.py"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import config
from src import data_loader, preprocessing, train, evaluate

st.set_page_config(page_title="LSTM vs GRU Forecasting", layout="wide")
st.title("📈 Stock / Crypto Trend Forecasting — LSTM vs GRU")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Asset", config.CRYPTOS + config.STOCKS)
    custom = st.text_input("…or custom Yahoo ticker")
    ticker = custom.strip() or ticker
    source = st.selectbox("Data source", ["auto", "yahoo", "synthetic"])
    window = st.slider("Look-back window (days)", 20, 120, config.WINDOW, 10)
    epochs = st.slider("Max epochs", 3, 100, 20)
    days = st.slider("Forecast days", 5, 60, config.FORECAST_DAYS)
    go = st.button("Train & compare", type="primary")

df = data_loader.load_data(ticker, source=source)
st.subheader(f"{ticker} — {len(df)} daily records")
st.line_chart(df["Close"])

if go:
    ds = preprocessing.prepare(df, window=window)
    res, preds, fcs = {}, {}, {}
    for kind in ["LSTM", "GRU"]:
        with st.spinner(f"Training {kind}…"):
            model, hist, secs = train.train_model(kind, ds, epochs=epochs, verbose=0)
        actual, p = evaluate.predict_test(model, ds)
        m = evaluate.metrics(actual, p); m["Train time(s)"] = round(secs, 1)
        res[kind], preds[kind] = m, p
        fcs[kind] = evaluate.forecast_future(model, ds, days)

    st.subheader("Metrics")
    st.dataframe(pd.DataFrame(res).T.round(4))
    st.success(f"Best by RMSE: {min(res, key=lambda k: res[k]['RMSE'])}")

    dates = ds.test_dates[-len(actual):]
    chart = pd.DataFrame({"Actual": actual, "LSTM": preds["LSTM"],
                          "GRU": preds["GRU"]}, index=dates)
    st.subheader("Actual vs predicted (test set)"); st.line_chart(chart)

    fidx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    st.subheader(f"{days}-day forecast")
    st.line_chart(pd.DataFrame({k: v for k, v in fcs.items()}, index=fidx))
    st.caption("⚠️ Educational project — not financial advice.")
