import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from plotly import graph_objs as go

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
data_load_state.text("Loading data...done!")

st.subheader('Raw Data')
st.write(data)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()
