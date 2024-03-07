import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import date

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# S&P 500 (^GSPC)
# NASDAQ Composite (^IXIC)
# Dow Jones Industrial Average (^DJI)
# CAC 40 (^FCHI)
# DAX PERFORMANCE-INDEX (^GDAXI)
# ALL ORDINARIES (^AORD)
# HANG SENG INDEX (^HSI)
# Nikkei 225 (^N225)

US_EU_market_dict = {
    "^GSPC": [], 
    "^IXIC": [],
    "^DJI": [],
    "^FCHI": [],
    "^GDAXI": []
}

asia_market_dict = {
    "^AORD": [],
    "^HSI": [],
    "^N2255": []
}

for ticker in US_EU_market_dict.keys():
    data = yf.download(ticker, START, TODAY)
    US_EU_market_dict[ticker] = data

for ticker in asia_market_dict.keys():
    data = yf.download(ticker, START, TODAY)
    asia_market_dict[ticker] = data

def data_munging_stocks(target_stock_name, target_stock_data): 
    df = pd.DataFrame(index=target_stock_data.index)
    df[target_stock_name] = target_stock_data['Open'].shift(-1) - target_stock_data['Open']
    df[target_stock_name + "_lag"] = df[target_stock_name].shift(1)

    for ticker in US_EU_market_dict.keys():
        data = US_EU_market_dict[ticker]
        df[ticker] = data['Open'] - data['Open'].shift(1)

    for ticker in asia_market_dict.keys():
        data = asia_market_dict[ticker]
        df[ticker] = data['Open'] - data['Open'].shift(1)

    return df