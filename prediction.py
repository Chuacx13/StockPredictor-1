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

    window = 20
    df[f'{target_stock_name}_SMA_{window}'] = target_stock_data['Close'].rolling(window=window).mean()
    df[f'{target_stock_name}_EMA_{window}'] = target_stock_data['Close'].ewm(span=window, adjust=False).mean()

    window = 200
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

# Prepare train scaled set, test scaled set, train data, test data and model
def prep_train_test_model(df_scaled, df): 
    proportion_test = 0.8
    gap = 180
    train_index = round(len(df_scaled) * proportion_test)
    test_index = train_index + gap
    X_scaled_train = df_scaled.iloc[:train_index, 1:]
    X_scaled_test = df_scaled.iloc[test_index:, 1:]
    y_scaled_train = df_scaled.iloc[:train_index, 0]
    y_scaled_test = df_scaled.iloc[test_index:, 0]

    train = df.iloc[:train_index, :]
    test = df.iloc[test_index:, :]  

    rf = RandomForestRegressor(n_estimators=1000)
    rf.fit(X_scaled_train, y_scaled_train)
    return X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test, train, test, rf # might not need y
    
def adjusted_metric(scaled_data, data, model, num_of_predictors, yname, scaler_y):  
    predictions = model.predict(scaled_data)
    unscaled_predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    data.loc[:,'predicted_gain'] = unscaled_predictions

    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['predicted_gain'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['predicted_gain'])**2).sum()
    r2 = SSR/SST

    adjustR2 = 1 - (1 - r2)*(data.shape[0] - 1)/(data.shape[0] - num_of_predictors - 1)
    RMSE = (SSE/(data.shape[0] - num_of_predictors - 1))**0.5
    return adjustR2, RMSE

def assess_table(train_scaled_data, test_scaled_data, train_data, test_data, model, yname, scaler_y):
    num_of_predictors = len(train_data.columns) - 1
    r2test, RMSEtest = adjusted_metric(test_scaled_data, test_data, model, num_of_predictors, yname, scaler_y)
    r2train, RMSEtrain = adjusted_metric(train_scaled_data, train_data, model, num_of_predictors, yname, scaler_y)

    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

#Test Output
result = data_munging_stocks("AAPL", yf.download("AAPL", START, TODAY))
gain = prep_train_test_model(result[0], result[2])
final = assess_table(gain[0], gain[1], gain[4], gain[5], gain[6], "AAPL", result[1])
print(final)


