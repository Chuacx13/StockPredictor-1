import streamlit as st
import pandas as pd
import yfinance as yf 
import charts as ch
import prediction as pred
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from plotly import graph_objs as go

# Timeframe used
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# Other Markets used for prediction
# S&P 500 (^GSPC)
# NASDAQ Composite (^IXIC)
# Dow Jones Industrial Average (^DJI)
# CAC 40 (^FCHI)
# DAX PERFORMANCE-INDEX (^GDAXI)
# ALL ORDINARIES (^AORD)
# HANG SENG INDEX (^HSI)
# Nikkei 225 (^N225)

US_EU_list = ["^GSPC", "^IXIC", "^DJI", "^FCHI","^GDAXI"]
US_EU_market_dict = {}

asia_list = ["^AORD", "^HSI", "^N225"]
asia_market_dict = {}

@st.cache_data
def get_market_dict():
    for ticker in US_EU_list:
        data = yf.download(ticker, START, TODAY)
        US_EU_market_dict[ticker] = data

    for ticker in asia_list:
        data = yf.download(ticker, START, TODAY)
        asia_market_dict[ticker] = data

get_market_dict()

# Stocks that will be predicted using other data from other markets
stocks = {
    "Apple": "AAPL",
    "Alibaba": "BABA",
    "Nvidia": "NVDA",
    "Taiwan Semiconductor": "TSM",
    "Advanced Micro Devices": "AMD",
    "Intel": "INTC",
    "Tesla": "TSLA"
}

# Select stocks
selected_stock_name = st.selectbox("Select Stock for Prediction", stocks.keys())
selected_stock = stocks[selected_stock_name]

rf = RandomForestRegressor(n_estimators=1000, random_state=42)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Raw Data
data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("")

st.subheader('Raw Data')
st.write(data)

# Price History
st.subheader(f'Open Price History of {selected_stock_name}')
ch.plot_open_raw_data(data, st)

st.subheader(f'Close Price History of {selected_stock_name}')
ch.plot_close_raw_data(data, st)

# Prepare data for analysis
state = st.text("Munging data")
df_scaled, scaler_y, df = pred.data_munging_stocks(selected_stock, data, US_EU_market_dict, asia_market_dict)

# Split into train and test sets; Fit data to model
state.text("Training model...")
X_scaled_train, X_scaled_test, train, test, rf_trained  = pred.prep_train_test_model(df_scaled, df, rf)

# Assess test and train set using RMSE and Adjusted R^2, 
state.text("Assessing train and test model...")
assessment, predict_train_data, predict_test_data = pred.assess_table(X_scaled_train, X_scaled_test, train, test, rf_trained, selected_stock, scaler_y)

# Calculate profits for train set
state.text("Calculating profits for train data...")
df_train, total_profits_train = pred.calc_profits(predict_train_data, selected_stock)

# Calculate profits for test set
state.text("Calculating profits for test data...")
df_test, total_profits_test = pred.calc_profits(predict_test_data, selected_stock)

# Plot Signal-based Trade vs Buy and Hold Strategy (Train)
st.subheader("Signal-based trade vs Buy and Hold Strategy (Train Set)")
state.text("Comparing strategies...")
ch.plot_hold_vs_trade(df_train, st)

# Plot Signal-based Trade vs Buy and Hold Strategy (Test)
st.subheader("Signal-based trade vs Buy and Hold Strategy (Test Set)")
state.text("Comparing strategies...")
ch.plot_hold_vs_trade(df_test, st)

# Plot Profit/Loss (Train)
st.subheader("Profit/Loss over Time (Train)")
state.text("Plotting profits vs loss...")
ch.plot_profit_loss(df_train, st)

# Plot Profit/Loss (Test)
st.subheader("Profit/Loss over Time (Test)")
state.text("Plotting profits vs loss...")
ch.plot_profit_loss(df_test, st)

st.header("Evaluation")
st.subheader("Train vs Test")

# Evaluation by Sharpe Ratio and Max Drawdown
state.text("Evaluating...")
sharpe_ratio_train = pred.calc_sharpe_ratio(df_train)
max_drawdown_train = pred.calc_max_drawdown(df_train)

sharpe_ratio_test = pred.calc_sharpe_ratio(df_test)
max_drawdown_test = pred.calc_max_drawdown(df_test)

assessment.loc['Daily_Sharpe_Ratio'] = [sharpe_ratio_train[0], sharpe_ratio_test[0]]
assessment.loc['Yearly_Sharpe_Ratio'] = [sharpe_ratio_train[1], sharpe_ratio_test[1]]
assessment.loc['Max_Drawdown'] = [max_drawdown_train, max_drawdown_test]
assessment.loc['Total_Profit'] = [total_profits_train, total_profits_test]

st.dataframe(assessment)
state.text("Prediction Complete")