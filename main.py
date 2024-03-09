import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
import charts
import prediction

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = {
    "Apple": "AAPL",
    "Alibaba": "BABA",
    "Nvidia": "NVDA",
    "Taiwan Semiconductor": "TSM",
    "Advanced Micro Devices": "AMD",
    "Intel": "INTC",
    "Tesla": "TSLA"
}

selected_stock_name = st.selectbox("Select Stock for Prediction", stocks.keys())
selected_stock = stocks[selected_stock_name]

n_years = st.slider("Years:", 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("")

st.subheader('Raw Data')
st.write(data)

st.subheader(f'Open Price History of {selected_stock_name}')
charts.plot_open_raw_data(data, st)

st.subheader(f'Close Price History of {selected_stock_name}')
charts.plot_close_raw_data(data, st)
