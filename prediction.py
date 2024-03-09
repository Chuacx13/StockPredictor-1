import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

#Process stocks used as predictors
@st.cache_data
def data_munging_stocks(target_stock_name, target_stock_data, US_EU_market_dict, asia_market_dict):
    df = target_stock_data[['Date']].copy()
    df[target_stock_name] = target_stock_data['Open'].shift(-1) - target_stock_data['Open']
    df[target_stock_name + "_lag"] = df[target_stock_name].shift(1)
    df['Close'] = target_stock_data['Close']
    
    df['Price'] = target_stock_data['Open']

    for ticker in US_EU_market_dict.keys():
        data = US_EU_market_dict[ticker]
        df[ticker] = data['Open'] - data['Open'].shift(1)

    for ticker in asia_market_dict.keys():
        data = asia_market_dict[ticker]
        df[ticker] = data['Close'] - data['Open']

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

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
@st.cache_data
def prep_train_test_model(df_scaled, df, _rf): 
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

    _rf.fit(X_scaled_train, y_scaled_train)
    return X_scaled_train, X_scaled_test, train, test, _rf 

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

@st.cache_data
def assess_table(train_scaled_data, test_scaled_data, train_data, test_data, _model, yname, _scaler_y):
    
    num_of_predictors = len(train_scaled_data.columns)
    r2test, RMSEtest, predict_test_data = adjusted_metric(test_scaled_data, test_data, _model, num_of_predictors, yname, _scaler_y)
    r2train, RMSEtrain, predict_train_data = adjusted_metric(train_scaled_data, train_data, _model, num_of_predictors, yname, _scaler_y)

    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]

    return assessment, predict_train_data, predict_test_data

@st.cache_data
def calc_profits(df, target_stock_name):
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
    return df_copy, total_profits

@st.cache_data
def calc_sharpe_ratio(df):
    df_copy = df.copy()
    df_copy['Wealth'] = df_copy['Trade'] + df_copy.loc[df_copy.index[0], 'Price']
    df_copy['Return'] =  np.log(df_copy['Wealth']) - np.log(df_copy['Wealth'].shift(1))
    daily_return = df_copy['Return'].dropna()
    daily_sharpe = daily_return.mean() / daily_return.std(ddof=1)
    yearly_sharpe = (252**0.5)*daily_return.mean() / daily_return.std(ddof=1)
    return daily_sharpe, yearly_sharpe

@st.cache_data
def calc_max_drawdown(df):
    df_copy = df.copy()
    df_copy['Wealth'] = df_copy['Trade'] + df_copy.loc[df_copy.index[0], 'Price']
    df_copy['Peak'] = df_copy['Wealth'].cummax()
    df_copy['Drawdown'] = (df_copy['Peak'] - df_copy['Wealth']) / df_copy['Peak']
    max_drawdown = df_copy['Drawdown'].max()
    return max_drawdown

