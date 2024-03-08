import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import date

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Other Markets used for prediction
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
    "^N225": []
}

for ticker in US_EU_market_dict.keys():
    data = yf.download(ticker, START, TODAY)
    US_EU_market_dict[ticker] = data

for ticker in asia_market_dict.keys():
    data = yf.download(ticker, START, TODAY)
    asia_market_dict[ticker] = data

#Process stocks used as predictors
def data_munging_stocks(target_stock_name, target_stock_data): 
    df = pd.DataFrame(index=target_stock_data.index)
    df[target_stock_name] = target_stock_data['Open'].shift(-1) - target_stock_data['Open']
    df[target_stock_name + "_lag"] = df[target_stock_name].shift(1)

    window = 10  
    df[f'{target_stock_name}_SMA_{window}'] = target_stock_data['Close'].rolling(window=window).mean()
    df[f'{target_stock_name}_EMA_{window}'] = target_stock_data['Close'].ewm(span=window, adjust=False).mean()

    window = 50  
    df[f'{target_stock_name}_SMA_{window}'] = target_stock_data['Close'].rolling(window=window).mean()
    df[f'{target_stock_name}_EMA_{window}'] = target_stock_data['Close'].ewm(span=window, adjust=False).mean()

    for ticker in US_EU_market_dict.keys():
        data = US_EU_market_dict[ticker]
        df[ticker] = data['Open'] - data['Open'].shift(1)

    for ticker in asia_market_dict.keys():
        data = asia_market_dict[ticker]
        df[ticker] = data['Close'] - data['Open']

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    scaler_y = MinMaxScaler()
    y = df[[target_stock_name]].copy()
    scaled_y = scaler_y.fit_transform(y)

    scaler_x = MinMaxScaler()
    x = df.drop(columns=[target_stock_name]).copy()
    for col in x.columns:
        x[col] = scaler_x.fit_transform(x[[col]])

    df_scaled = pd.DataFrame(index=df.index, data=x, columns=x.columns)
    df_scaled[target_stock_name] = scaled_y

    return df_scaled, scaler_y, df

def predict_gain(df, scaler): 
    train_index = round(len(df) * 0.7)
    test_index = train_index + 30
    X_train = df.iloc[:train_index, 1:]
    X_test = df.iloc[test_index:, 1:]
    y_train = df.iloc[:train_index, 0]
    y_test = df.iloc[test_index:, 0]
    rf =  RandomForestRegressor(n_estimators=1000)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return unscaled_predictions
    
result = data_munging_stocks("AAPL", yf.download("AAPL", START, TODAY))
print(result[0], result[2])
