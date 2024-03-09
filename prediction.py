import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import matplotlib.pyplot as plt

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
    df['Close'] = target_stock_data['Close']
    df['Date'] = target_stock_data.index

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
    x = df.drop(columns=[target_stock_name, 'Close']).copy()
    for col in x.columns:
        x[col] = scaler_x.fit_transform(x[[col]])

    df_scaled = pd.DataFrame(index=df.index, data=x, columns=x.columns)
    df_scaled[target_stock_name] = scaled_y

    return df_scaled, scaler_y, df

# Prepare train scaled set, test scaled set, train data, test data and model
def prep_train_test_model(df_scaled, df): 
    proportion_test = 0.7
    gap = 365
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
    data_copy = data.copy()
    data_copy['predicted_gain'] = unscaled_predictions

    SST = ((data_copy[yname] - data_copy[yname].mean())**2).sum()
    SSR = ((data_copy['predicted_gain'] - data_copy[yname].mean())**2).sum()
    SSE = ((data_copy[yname] - data_copy['predicted_gain'])**2).sum()
    r2 = SSR/SST

    adjustR2 = 1 - (1 - r2)*(data_copy.shape[0] - 1)/(data_copy.shape[0] - num_of_predictors - 1)
    RMSE = (SSE/(data_copy.shape[0] - num_of_predictors - 1))**0.5
    return adjustR2, RMSE, data_copy

def assess_table(train_scaled_data, test_scaled_data, train_data, test_data, model, yname, scaler_y):
    num_of_predictors = len(train_scaled_data.columns)
    r2test, RMSEtest, predict_test_data = adjusted_metric(test_scaled_data, test_data, model, num_of_predictors, yname, scaler_y)
    r2train, RMSEtrain, predict_train_data = adjusted_metric(train_scaled_data, train_data, model, num_of_predictors, yname, scaler_y)

    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]

    return assessment, predict_train_data, predict_test_data

def calculate_profits(df, target_stock_name):
    df_copy = df.copy()
    window = 10
    df_copy[f'{target_stock_name}_SMA_{window}'] = df_copy['Close'].rolling(window=window).mean()
    df_copy[f'{target_stock_name}_EMA_{window}'] = df_copy['Close'].ewm(span=window, adjust=False).mean()

    window2 = 50
    df_copy[f'{target_stock_name}_SMA_{window2}'] = df_copy['Close'].rolling(window=window2).mean()
    df_copy[f'{target_stock_name}_EMA_{window2}'] = df_copy['Close'].ewm(span=window2, adjust=False).mean()

    df_copy.dropna(inplace=True)

    # Long if predicted gain is positive, SMA_window > SMA__window2 and EMA_window > EMA_window2, short otherwise
    df_copy['Order'] = [1 if row['predicted_gain'] > 0 and 
                    row[f'{target_stock_name}_SMA_{window}'] > row[f'{target_stock_name}_SMA_{window2}'] and 
                    row[f'{target_stock_name}_EMA_{window}'] > row[f'{target_stock_name}_EMA_{window2}'] 
                    else -1
                    for _, row in df_copy.iterrows()]
    
    df_copy['Profit'] = df_copy[target_stock_name] * df_copy['Order']
    df_copy['Trade'] = df_copy['Profit'].cumsum()
    df_copy['Hold'] = df_copy[target_stock_name].cumsum()
    total_profits = df_copy['Profit'].sum()

    df_copy.set_index('Date', inplace=True)
    return df_copy, total_profits

def plot_profit_loss(df_copy):
    df_copy['Profit'].plot()
    plt.axhline(y=0, color='red')
    plt.xlabel('Date')
    plt.ylabel('Profit/Loss')
    plt.show()
    return

def hold_vs_trade(df_copy):
    plt.plot(df_copy.index, df_copy['Trade'].values, color = 'green', label = 'Signal based strategy')
    plt.plot(df_copy.index, df_copy['Hold'].values, color = 'red', label = 'Buy and Hold strategy')
    plt.xlabel('Wealth')
    plt.ylabel('Profit/Loss')
    plt.legend()
    plt.show()
    return

#Test Output
result = data_munging_stocks("TSLA", yf.download("TSLA", START, TODAY))
gain = prep_train_test_model(result[0], result[2])
final = assess_table(gain[0], gain[1], gain[4], gain[5], gain[6], "TSLA", result[1])
profit = calculate_profits(final[1], "TSLA")
hold_vs_trade(profit[0])
#plot_profit_loss(profit[0])
